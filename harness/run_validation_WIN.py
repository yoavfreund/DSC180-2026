#!/usr/bin/env python3
"""Windows companion: run inside Ubuntu WSL2, with Bubblewrap isolation.

CLI: python3 harness/run_validation_WIN.py WORKSPACE --run-name sirio/run-01
MCP: add --serve for a newline-delimited JSON-RPC stdio server for Codex.
Original macOS files are unchanged. No third-party Python packages required.
"""
import argparse
import datetime
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import threading
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRIVATE = Path(os.environ.get("DSC180_PRIVATE_WIN", Path.home() / "dsc180-private-win")).resolve()
EVAL_DIR = REPO / "eval"

# Agent feedback targets (see pipeline/SPEC_WIN.md). Validate has 682 negatives,
# so FP rate moves in steps of ~0.15%.
DEFAULT_FP_RATES = "0.01,0.05,0.1"
MAX_LEARNER_BYTES = 5_000_000
STAGE_LINE = re.compile(r"^stage=(\d+) weak=(\d+)$")
SPEED_REPEATS = 3  # predict timings take the minimum of this many runs


class Validator:
    def __init__(self, workspace, run_name, fp_rates=DEFAULT_FP_RATES, max_calls=20,
                 timeouts=None, require_cascade=False):
        self.workspace = Path(workspace).resolve()
        self.fp_rates = fp_rates
        self.max_calls = max_calls
        self.require_cascade = require_cascade
        self.timeouts = {"make": 120, "train": 1800, "describe": 30, "predict": 300, **(timeouts or {})}
        self.calls = 0
        self.lock = threading.Lock()  # one validation at a time, even if the agent calls in parallel
        self.train_dir = PRIVATE / "staged" / "train"
        self.val_images = PRIVATE / "staged" / "validate" / "images"
        self.val_key = PRIVATE / "staged" / "validate" / "labels.txt"
        for p in (self.train_dir, self.val_images, self.val_key):
            if not p.exists():
                raise SystemExit(f"{p} missing; run `pipeline/stage_WIN.py prepare` first")
        if not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)?", run_name):
            raise ValueError("run_name must be one or two simple names separated by /")
        self.run_dir = PRIVATE / "runs" / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "runs.jsonl"
        if not shutil.which("bwrap"):
            raise SystemExit("Bubblewrap missing: install bubblewrap in Ubuntu WSL first")
        if not Path('/usr/bin/time').is_file():
            raise SystemExit('GNU time missing: install the time package in Ubuntu WSL')
        # Compile trusted code outside the repo; never modify the original eval build.
        self.eval_bin = self.run_dir / "eval"
        subprocess.run(["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                        str(EVAL_DIR / "eval.c"), "-lm", "-o", str(self.eval_bin)], check=True)
        self.n_val_images = sum(1 for _ in self.val_images.glob("*.pgm"))
        self.labels = dict(line.split() for line in self.val_key.read_text().splitlines() if line.strip())
        # A one-image directory (a train image) for measuring predict's start-up cost.
        self.one_image = self.run_dir / "one_image"
        if not self.one_image.exists():
            self.one_image.mkdir()
            first = min(self.train_dir.joinpath("images").glob("*.pgm"))
            shutil.copyfile(first, self.one_image / first.name)

    def run(self):
        """Run one validation pass; returns the text shown to the agent."""
        with self.lock, self._run_lock():
            # Persist the budget across CLI invocations and MCP reconnects.
            records = self.log_path.read_text().splitlines() if self.log_path.exists() else []
            self.calls = max((json.loads(line)["call"] for line in records if line.strip()), default=0)
            if self.calls >= self.max_calls:
                return f"Validation budget used up ({self.max_calls} calls). Finalize your learner."
            self.calls += 1
            build = self.run_dir / f"build_{self.calls:03d}"
            record = {"call": self.calls, "time": datetime.datetime.now().isoformat(timespec="seconds")}
            try:
                record["learner_sha256"] = self._snapshot(build)
                text, record["status"], record["eval"] = self._pipeline(build, record)
            except LearnerError as e:
                text, record["status"] = f"ERROR: {e}", "rejected"
            except Exception:  # a harness bug must not crash the agent's tool call or skip the log
                record["status"], record["error"] = "harness_error", traceback.format_exc()
                text = "ERROR: the validation harness itself failed (logged for the humans). Try again later."
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            return f"[validation call {self.calls}/{self.max_calls}]\n{text}"

    @contextlib.contextmanager
    def _run_lock(self):
        with open(self.run_dir / "validation.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _snapshot(self, build):
        src = self.workspace / "learner"
        if src.is_symlink():
            raise LearnerError("learner/ itself must not be a symlink")
        if not src.is_dir():
            raise LearnerError("no learner/ directory in the workspace")
        total, digest = 0, hashlib.sha256()
        for p in sorted(src.rglob("*")):
            if p.is_symlink():
                raise LearnerError(f"symlinks are not allowed in learner/: {p.relative_to(src)}")
            if p.is_file():
                total += p.stat().st_size
                digest.update(str(p.relative_to(src)).encode() + b"\0" + p.read_bytes())
        if total > MAX_LEARNER_BYTES:
            raise LearnerError("learner/ is over 5 MB; keep only source and build files")
        shutil.copytree(src, build, ignore=shutil.ignore_patterns("model*", "*.o", "train", "predict"))
        (build / "tmp").mkdir(exist_ok=True)
        return digest.hexdigest()

    def _sandboxed(self, step, cmd, build, data_dir, record=None, stdout=None):
        """Run one learner step in the sandbox. Returns (result, wall seconds, CPU seconds);
        with a record, also stores the timing under record["timing"][step]."""
        build, data_dir = build.resolve(), Path(data_dir).resolve()
        def guest_path(arg):
            arg = str(arg)
            for host, guest in ((build, "/work"), (data_dir, "/input")):
                if arg == str(host) or arg.startswith(str(host) + "/"):
                    return guest + arg[len(str(host)):]
            return arg
        # An empty root with explicit mounts. Neither /mnt/c nor the host home,
        # validation key, repo, or run logs exist inside the learner sandbox.
        # Run the command as namespace PID 1. Bubblewrap's default intermediate
        # reaper loses descendant CPU accounting in RUSAGE_CHILDREN on WSL.
        argv = ["bwrap", "--unshare-all", "--as-pid-1", "--die-with-parent", "--new-session",
                "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "HOME", "/work", "--setenv", "TMPDIR", "/tmp",
                "--ro-bind", "/usr", "/usr"]
        for path in ("/bin", "/sbin", "/lib", "/lib64"):
            if Path(path).exists():
                argv += ["--ro-bind", path, path]
        argv += ["--dir", "/etc"]
        # Debian/Ubuntu /usr/bin/cc resolves through /etc/alternatives.
        if Path("/etc/alternatives").is_dir():
            argv += ["--ro-bind", "/etc/alternatives", "/etc/alternatives"]
        if Path("/etc/ld.so.cache").exists():
            argv += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
        argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                 "--bind", str(build), "/work"]
        if data_dir != build:
            argv += ["--ro-bind", str(data_dir), "/input"]
        argv += ["--chdir", "/work", "--", *map(guest_path, cmd)]
        # The trusted GNU time wrapper writes outside the learner's mount.
        # Its per-command peak RSS avoids RUSAGE_CHILDREN's cumulative maximum.
        with tempfile.NamedTemporaryFile(prefix='usage_', suffix='.txt', dir=self.run_dir,
                                         delete=False) as usage:
            usage_path = Path(usage.name)
        argv = ['/usr/bin/time', '-q', '-f', '%M', '-o', str(usage_path), '--', *argv]
        before, start = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
        try:
            r = subprocess.run(argv, cwd=build, stdout=stdout if stdout is not None else subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, timeout=self.timeouts[step])
        except subprocess.TimeoutExpired:
            raise LearnerError(f"{step} timed out after {self.timeouts[step]}s")
        finally:
            usage_text = usage_path.read_text().strip()
            usage_path.unlink(missing_ok=True)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall = time.perf_counter() - start
        cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
        if record is not None:
            record.setdefault("timing", {})[step] = {
                "wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
                "peak_rss_kib": int(usage_text.splitlines()[-1])}
        if any(p.is_symlink() for p in build.rglob("*")):
            raise LearnerError(f"{step} created a symlink; learner output must use regular files")
        return r, wall, cpu

    def _describe(self, build, record):
        """Ask the learner for its cascade layout via ./predict --describe MODEL."""
        r, _, _ = self._sandboxed("describe", ["./predict", "--describe", str(build / "model")], build, build, record)
        stages = [STAGE_LINE.match(line.strip()) for line in r.stdout.splitlines() if line.strip()]
        if r.returncode != 0 or not stages or not all(stages):
            return None
        if ([int(m.group(1)) for m in stages] != list(range(1, len(stages) + 1))
                or any(int(m.group(2)) < 1 for m in stages)):
            return None
        record["cascade"] = [int(m.group(2)) for m in stages]
        return record["cascade"]

    def _predict_speed(self, build, record):
        """Per-image predict cost with start-up and model loading removed: the difference between
        timing all validate images and timing one image, each the minimum of a few repeats."""
        cmd = lambda d: ["./predict", str(build / "model"), str(d)]
        def measure(directory):
            result, _, cpu = self._sandboxed("predict", cmd(directory), build, directory,
                                             stdout=subprocess.DEVNULL)
            if result.returncode:
                raise LearnerError("predict failed during timing (stderr withheld)")
            return cpu
        full = min(measure(self.val_images) for _ in range(SPEED_REPEATS))
        one = min(measure(self.one_image) for _ in range(SPEED_REPEATS))
        speed = {"startup_cpu_ms": round(one * 1e3, 2),
                 "cpu_us_per_image": round(max(full - one, 0) / max(self.n_val_images - 1, 1) * 1e6, 1)}
        record["speed"] = speed
        return speed

    def _weak_evaluated(self, scores_path, cascade, record):
        """Average weak classifiers evaluated per face / non-face. The cascade score convention
        (score in [k-1, k) for an image that stops at stage k) gives each image's exit stage."""
        cum = [sum(cascade[:i + 1]) for i in range(len(cascade))]
        sums, counts = {"1": 0, "0": 0}, {"1": 0, "0": 0}
        for line in scores_path.read_text().splitlines():
            sid, score = line.split()
            value = float(score)
            if not 0 <= value < len(cascade):
                raise LearnerError("cascade scores must lie in [0, number of stages)")
            stage = int(math.floor(value))
            label = self.labels[sid]
            sums[label] += cum[stage]
            counts[label] += 1
        out = {"non_face": round(sums["0"] / counts["0"], 1), "face": round(sums["1"] / counts["1"], 1),
               "total": cum[-1]}
        record["weak_evaluated"] = out
        return out

    def _pipeline(self, build, record):
        r, _, _ = self._sandboxed("make", ["make"], build, build, record)
        if r.returncode != 0:
            return f"make failed:\n{tail(r.stderr or r.stdout)}", "make_failed", None
        for exe in ("train", "predict"):
            if not (build / exe).is_file():
                return f"make did not produce ./{exe}", "make_failed", None

        r, _, _ = self._sandboxed("train", ["./train", str(self.train_dir / "images"),
                                            str(self.train_dir / "labels.txt"), str(build / "model")],
                                  build, self.train_dir, record)
        if r.returncode != 0:
            return f"train exited {r.returncode}:\n{tail(r.stderr)}", "train_failed", None

        cascade = self._describe(build, record) if self.require_cascade else None
        if self.require_cascade and (not cascade or len(cascade) < 2):
            return ("cascade required: `./predict --describe MODEL` must print one line per stage, "
                    "`stage=<k> weak=<number of weak classifiers>`, with at least 2 stages"), "no_cascade", None

        # Trusted output lives outside the learner's writable mount.
        scores = self.run_dir / f"scores_{self.calls:03d}.txt"
        with open(scores, "w") as out:
            r, _, _ = self._sandboxed("predict", ["./predict", str(build / "model"), str(self.val_images)],
                                      build, self.val_images, record, stdout=out)
        if r.returncode != 0:
            # predict reads validate images, so its stderr is not passed back.
            return f"predict exited {r.returncode} (stderr withheld: predict runs on validation data)", \
                "predict_failed", None

        e = subprocess.run([str(self.eval_bin), str(scores), str(self.val_key), "--fp-rates", self.fp_rates],
                           capture_output=True, text=True)
        if e.returncode != 0:
            return "predict output rejected: emit exactly one finite score per image ID", "bad_scores", None

        t, speed = record["timing"], self._predict_speed(build, record)
        lines = [summarize(e.stdout),
                 f"timing: train {t['train']['wall_s']:.1f} s wall ({t['train']['cpu_s']:.1f} s CPU); "
                 f"predict {speed['cpu_us_per_image']:.1f} us CPU per image "
                 f"(start-up and model loading, {speed['startup_cpu_ms']:.1f} ms, excluded)",
                 f"predict batch: {self.n_val_images} images, {t['predict']['wall_s']:.3f} s wall "
                 f"({t['predict']['cpu_s']:.3f} s CPU), including start-up, model loading and image I/O"]
        record['model_bytes'] = (build / 'model').stat().st_size
        lines.append(f"model size: {record['model_bytes']} bytes; peak resident memory: "
                     f"train {t['train']['peak_rss_kib'] / 1024:.2f} MiB, "
                     f"predict {t['predict']['peak_rss_kib'] / 1024:.2f} MiB")
        if cascade:
            weak = self._weak_evaluated(scores, cascade, record)
            lines.append(f"cascade: {len(cascade)} stages, weak classifiers per stage: "
                         + ",".join(map(str, cascade)))
            lines.append(f"weak classifiers evaluated per image: non-faces {weak['non_face']}, "
                         f"faces {weak['face']} (of {weak['total']})")
        return "\n".join(lines), "ok", e.stdout


class LearnerError(Exception):
    pass


def tail(text, lines=15, width=300):
    return "\n".join(line[:width] for line in text.strip().splitlines()[-lines:])


def summarize(eval_out):
    """Round eval's output to 3 decimals: enough to tune by, less to probe with."""
    lines = []
    for line in eval_out.splitlines():
        kv = dict(field.split("=", 1) for field in line.split())
        if "auc" in kv:
            lines.append(f"validation: {kv['n_pos']} faces, {kv['n_neg']} non-faces, AUC {float(kv['auc']):.3f}")
        else:
            prec = kv["precision"] if kv["precision"] == "NA" else f"{float(kv['precision']):.3f}"
            lines.append(f"at FP rate <= {kv['fp_rate']}: detection rate {float(kv['tpr']):.3f}, "
                         f"precision {prec}, false positives {kv['fp']}")
    return "\n".join(lines)


def serve(validator):
    """Minimal synchronous MCP stdio transport; one validation at a time.

    stdout is protocol-only. The caller starts this trusted service outside
    the learner sandbox. No Claude SDK or API key is used by this server.
    """
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:  # initialized/cancelled notifications
            continue
        method = request.get("method")
        response = {"jsonrpc": "2.0", "id": request["id"]}
        if method == "initialize":
            result = {"protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"),
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "dsc180-win", "version": "1.0.0"}}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [{"name": "run_validation",
                      "description": "Build and train learner/, validate privately, return aggregate accuracy, timing and declared cascade statistics. Limited calls; can take 30 minutes.",
                      "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}}]}
        elif method == "tools/call" and request.get("params", {}).get("name") == "run_validation":
            if request.get("params", {}).get("arguments", {}):
                result = {"content": [{"type": "text", "text": "run_validation takes no arguments"}], "isError": True}
            else:
                result = {"content": [{"type": "text", "text": validator.run()}]}
        else:
            response["error"] = {"code": -32601, "message": "Method not found"}
            print(json.dumps(response), flush=True)
            continue
        response["result"] = result
        print(json.dumps(response), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Run one validation pass on a workspace's learner/.")
    ap.add_argument("workspace")
    ap.add_argument("--run-name", default="manual")
    ap.add_argument("--fp-rates", default=DEFAULT_FP_RATES)
    ap.add_argument("--require-cascade", action="store_true")
    ap.add_argument("--max-calls", type=int, default=20)
    ap.add_argument("--serve", action="store_true", help="serve the validation tool over MCP stdio")
    args = ap.parse_args()
    validator = Validator(args.workspace, args.run_name, args.fp_rates, max_calls=args.max_calls,
                          require_cascade=args.require_cascade)
    if args.serve:
        serve(validator)
    else:
        print(validator.run())


if __name__ == "__main__":
    main()
