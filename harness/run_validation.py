#!/usr/bin/env python3
"""Validation harness: the agent's only window onto the validation set.

One call = snapshot the agent's learner/ -> make -> ./train on staged train ->
(optionally) ./predict --describe to check for a cascade -> ./predict on staged
validate images -> eval against the private validate key. Every step is timed.
Every agent-written binary runs under harness/learner.sb (no network, nothing
under /Users readable except its build dir and the one data dir the step
needs). The validate key never enters a sandbox; only the trusted eval binary
reads it.

The agent gets back rounded aggregate metrics only. Never per-image results,
which would let it probe labels. Full-precision results go to a private log.

Used two ways:
  * as an Agent SDK tool:  make_server(Validator(...))
  * by hand:               run_validation.py WORKSPACE [--fp-rates ...]
"""
import argparse
import asyncio
import datetime
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
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRIVATE = Path(os.environ.get("DSC180_PRIVATE", Path.home() / "dsc180-private"))
PROFILE = REPO / "harness" / "learner.sb"
EVAL_DIR = REPO / "eval"

# Agent feedback targets (see pipeline/SPEC.md). Validate has 682 negatives,
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
                raise SystemExit(f"{p} missing; run `pipeline/stage.py prepare` first")
        self.run_dir = PRIVATE / "runs" / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "runs.jsonl"
        subprocess.run(["make", "-s", "-C", str(EVAL_DIR), "eval"], check=True)
        self.eval_bin = EVAL_DIR / "eval"
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
        with self.lock:
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

    def _snapshot(self, build):
        src = self.workspace / "learner"
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
        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "TMPDIR": str(build / "tmp"), "HOME": str(build)}
        argv = ["sandbox-exec", "-D", f"BUILD={build}", "-D", f"INPUT={data_dir}", "-f", str(PROFILE), *cmd]
        before, start = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
        try:
            r = subprocess.run(argv, cwd=build, env=env, stdout=stdout or subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, timeout=self.timeouts[step])
        except subprocess.TimeoutExpired:
            raise LearnerError(f"{step} timed out after {self.timeouts[step]}s")
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall = time.perf_counter() - start
        cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
        if record is not None:
            record.setdefault("timing", {})[step] = {"wall_s": round(wall, 3), "cpu_s": round(cpu, 3)}
        return r, wall, cpu

    def _describe(self, build, record):
        """Ask the learner for its cascade layout via ./predict --describe MODEL."""
        r, _, _ = self._sandboxed("describe", ["./predict", "--describe", str(build / "model")], build, build, record)
        stages = [STAGE_LINE.match(line.strip()) for line in r.stdout.splitlines() if line.strip()]
        if r.returncode != 0 or not stages or not all(stages):
            return None
        record["cascade"] = [int(m.group(2)) for m in stages]
        return record["cascade"]

    def _predict_speed(self, build, record):
        """Per-image predict cost with start-up and model loading removed: the difference between
        timing all validate images and timing one image, each the minimum of a few repeats."""
        cmd = lambda d: ["./predict", str(build / "model"), str(d)]
        full = min(self._sandboxed("predict", cmd(self.val_images), build, self.val_images,
                                   stdout=subprocess.DEVNULL)[2] for _ in range(SPEED_REPEATS))
        one = min(self._sandboxed("predict", cmd(self.one_image), build, self.one_image,
                                  stdout=subprocess.DEVNULL)[2] for _ in range(SPEED_REPEATS))
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
            stage = min(max(int(math.floor(float(score))), 0), len(cascade) - 1)
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

        scores = build / "scores.txt"
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
            return f"predict output rejected by eval: {e.stderr.strip()}", "bad_scores", None

        t, speed = record["timing"], self._predict_speed(build, record)
        lines = [summarize(e.stdout),
                 f"timing: train {t['train']['wall_s']:.1f} s wall ({t['train']['cpu_s']:.1f} s CPU); "
                 f"predict {speed['cpu_us_per_image']:.1f} us CPU per image "
                 f"(start-up and model loading, {speed['startup_cpu_ms']:.1f} ms, excluded)"]
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


def make_server(validator):
    """Wrap a Validator as an in-process Agent SDK MCP server named 'harness'."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool("run_validation",
          "Build learner/ with make, train it on the training set, run predict on the hidden "
          "validation set, and return aggregate metrics (detection rate and precision at fixed "
          "false-positive rates, plus AUC), train time, predict CPU time per image, and the cascade "
          "layout reported by ./predict --describe. Takes no arguments. Limited number of calls.",
          {})
    async def run_validation(args):
        # Training can take minutes; run it in a worker thread so the SDK's event loop stays responsive.
        text = await asyncio.to_thread(validator.run)
        return {"content": [{"type": "text", "text": text}]}

    return create_sdk_mcp_server(name="harness", tools=[run_validation])


def main():
    ap = argparse.ArgumentParser(description="Run one validation pass on a workspace's learner/.")
    ap.add_argument("workspace")
    ap.add_argument("--run-name", default="manual")
    ap.add_argument("--fp-rates", default=DEFAULT_FP_RATES)
    ap.add_argument("--require-cascade", action="store_true")
    args = ap.parse_args()
    print(Validator(args.workspace, args.run_name, args.fp_rates,
                    require_cascade=args.require_cascade).run())


if __name__ == "__main__":
    main()
