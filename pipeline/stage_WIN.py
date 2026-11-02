#!/usr/bin/env python3
"""Stage CBCL splits into a leak-free form and build agent workspaces.

The repo keeps readable train/{face,non-face} and validate/{face,non-face}
folders. Agents never see those. Instead, this script writes a "staged" copy of
each split in which nothing about a file except its pixels says what it is:

  * names are neutral ids (000001.pgm, ...) from a seeded permutation
  * files are rewritten (not copied) with a canonical PGM header
  * every file gets the same fixed mtime (the CBCL originals leak the label:
    faces are dated 1997, non-faces 1999)
  * files are created in id order, so creation time and inode order follow
    the random id, not the class

Each staged split has:
  images/<id>.pgm
  labels.txt     "<id> <label>" (1 = face, 0 = non-face). For validate/test
                 this is the private answer key and never goes to an agent.
  ledger.tsv     id, label, original path (for humans/auditing only)

Usage:
  stage_WIN.py doctor                             check WSL build/sandbox tools
  stage_WIN.py split NAME SRC_DIR OUT_DIR [--seed N] stage any split (incl. test)
  stage_WIN.py prepare [--seed N]                   stage repo train + validate
  stage_WIN.py workspace NAME RUN_ID                create a personal workspace

Locations (override with env vars):
  DSC180_PRIVATE_WIN  default ~/dsc180-private-win   staged splits, harness-only
  DSC180_WORK_WIN     default ~/dsc180-work-win      personal workspaces
"""
import argparse
import io
import json
import os
import random
import shutil
import sys
import subprocess
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRIVATE = Path(os.environ.get("DSC180_PRIVATE_WIN", Path.home() / "dsc180-private-win")).resolve()
WORK = Path(os.environ.get("DSC180_WORK_WIN", Path.home() / "dsc180-work-win")).resolve()

TEAM_SEED = 42
SIDE = 19
HEADER = f"P5\n{SIDE} {SIDE}\n255\n".encode()
FIXED_MTIME = 946684800  # 2000-01-01 00:00:00 UTC
CLASSES = {"face": 1, "non-face": 0}


def read_pgm_pixels(path):
    """Return the raw pixel bytes of a binary 19x19 8-bit PGM, validating the header."""
    data = path.read_bytes()
    tokens, pos = [], 0
    while len(tokens) < 4:
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":  # header comment
            while pos < len(data) and data[pos:pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    pos += 1  # single whitespace byte after maxval
    if tokens != [b"P5", str(SIDE).encode(), str(SIDE).encode(), b"255"]:
        sys.exit(f"{path}: expected P5 {SIDE}x{SIDE} maxval 255, got {tokens}")
    pixels = data[pos:pos + SIDE * SIDE]
    if len(pixels) != SIDE * SIDE:
        sys.exit(f"{path}: truncated pixel data")
    return pixels


def stage_split(name, src, out, seed):
    src, out = Path(src), Path(out)
    if out.exists():
        sys.exit(f"{out} already exists; remove it first to restage")
    items = []
    for cls, label in CLASSES.items():
        files = sorted((src / cls).glob("*.pgm"))
        if not files:
            sys.exit(f"no .pgm files in {src / cls}")
        items += [(p, label) for p in files]
    items.sort(key=lambda it: str(it[0]))
    # String seeds are hashed deterministically, so each split gets its own stable permutation.
    random.Random(f"{seed}:{name}").shuffle(items)

    images = out / "images"
    images.mkdir(parents=True)
    width = max(6, len(str(len(items))))
    ledger = ["id\tlabel\toriginal_path"]
    labels = []
    for i, (path, label) in enumerate(items, start=1):
        sid = f"{i:0{width}d}"
        dst = images / f"{sid}.pgm"
        dst.write_bytes(HEADER + read_pgm_pixels(path))
        os.utime(dst, (FIXED_MTIME, FIXED_MTIME))
        ledger.append(f"{sid}\t{label}\t{path.relative_to(src)}")
        labels.append(f"{sid} {label}")
    (out / "ledger.tsv").write_text("\n".join(ledger) + "\n")
    (out / "labels.txt").write_text("\n".join(labels) + "\n")
    n_pos = sum(label for _, label in items)
    print(f"staged {name}: {len(items)} images ({n_pos} face, {len(items) - n_pos} non-face) -> {out}")


def prepare_data(seed):
    """Read committed image bytes, avoiding Windows text conversion of PGM files.

    Git can mistake a small PGM with no NUL bytes for text and convert LF to
    CRLF on checkout. Archive reads the committed blobs without checkout filters.
    This does not alter the working tree, Git configuration, or source images.
    """
    for split in ("train", "validate"):
        if (PRIVATE / "staged" / split).exists():
            sys.exit(f"{PRIVATE / 'staged' / split} already exists; inspect before restaging")
    archive = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "HEAD", "train", "validate"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    with tempfile.TemporaryDirectory(prefix="dsc180-win-source-") as temp:
        source = Path(temp)
        with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tree:
            tree.extractall(source, filter="data")
        for split in ("train", "validate"):
            stage_split(split, source / split, PRIVATE / "staged" / split, seed)


def doctor():
    if sys.platform != "linux" or "microsoft" not in Path("/proc/sys/kernel/osrelease").read_text().lower():
        sys.exit("Run this companion inside Ubuntu WSL2, not Windows Python.")
    missing = [name for name in ("python3", "cc", "make", "git", "bwrap", "wslpath") if not shutil.which(name)]
    if missing:
        sys.exit("Missing tools: " + ", ".join(missing) +
                 ". Install in Ubuntu: sudo apt-get install -y build-essential bubblewrap")
    probe = subprocess.run(["bwrap", "--unshare-all", "--die-with-parent", "--ro-bind", "/", "/",
                            "--", "/bin/true"], capture_output=True, text=True)
    if probe.returncode:
        sys.exit("Bubblewrap namespaces unavailable; no unsafe fallback.\n" + probe.stderr)
    print("Ready: Ubuntu WSL2, Python, C compiler, make, and Bubblewrap namespaces.")


def write_workspace_config(ws, agent, run_id, distro="Ubuntu", max_calls=20):
    """Configure only this workspace, without editing global Codex settings."""
    config_dir = ws / ".codex"
    config_dir.mkdir(exist_ok=True)
    server_args = ["-d", distro, "--cd", "/", "--", "env", f"DSC180_PRIVATE_WIN={PRIVATE}",
                   "python3", str(REPO / "harness" / "run_validation_WIN.py"), str(ws),
                   "--run-name", f"{agent}/{run_id}", "--require-cascade", "--max-calls", str(max_calls), "--serve"]
    (config_dir / "config.toml").write_text(
        '[mcp_servers.validation]\ncommand = "wsl.exe"\nargs = ' + json.dumps(server_args) +
        '\nstartup_timeout_sec = 60\ntool_timeout_sec = 4500\n')
    # CLI overrides also work before the project is trusted; project config may
    # otherwise be ignored. TOML literal strings avoid Windows quote stripping.
    def toml_literal(value):
        if "'''" in value or "\n" in value or "\r" in value:
            raise ValueError("unsupported newline or triple quote in workspace path")
        return "'''" + value + "'''" if "'" in value else "'" + value + "'"
    overrides = ["mcp_servers.validation.command='wsl.exe'",
                 "mcp_servers.validation.args=[" + ",".join(map(toml_literal, server_args)) + "]",
                 "mcp_servers.validation.startup_timeout_sec=60",
                 "mcp_servers.validation.tool_timeout_sec=4500"]
    def ps_literal(value):
        return "'" + value.replace("'", "''") + "'"
    cli_args = ", ".join(item for value in overrides for item in ("'-c'", ps_literal(value)))
    (ws / "start-codex.ps1").write_text(
        "param([switch]$Check)\n$ErrorActionPreference = 'Stop'\n"
        "$overrides = @(" + cli_args + ")\n"
        "if ($Check) {\n  & codex --cd $PSScriptRoot @overrides mcp get validation --json\n"
        "  exit $LASTEXITCODE\n}\n"
        "& codex --cd $PSScriptRoot @overrides --sandbox workspace-write "
        "'Read AGENTS.md and build my own boosted cascade learner in learner/. Train only on the supplied training data. Begin.'\n"
        "exit $LASTEXITCODE\n")


def build_workspace(agent, run_id, distro="Ubuntu", max_calls=20):
    import re
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (agent, run_id)):
        sys.exit("agent and run IDs must contain only letters, digits, underscores and hyphens")
    staged = PRIVATE / "staged" / "train"
    if not staged.exists():
        sys.exit(f"{staged} missing; run `stage_WIN.py prepare` first")
    ws = WORK / agent / run_id
    if ws.exists():
        sys.exit(f"{ws} already exists")
    # Only train goes in: images + labels. No ledger, no validate, no test, no .git.
    shutil.copytree(staged / "images", ws / "train" / "images", copy_function=shutil.copyfile)
    for p in sorted((ws / "train" / "images").iterdir()):
        os.utime(p, (FIXED_MTIME, FIXED_MTIME))
    shutil.copyfile(staged / "labels.txt", ws / "train" / "labels.txt")
    (ws / "learner").mkdir()
    instructions = """# Your personal boosted cascade learner

Build your own classifier in C for 19x19 grayscale patches. Work in learner/.
Use a Viola-Jones attentional cascade of boosted Haar-like classifiers, or
WaldBoost with early rejection after weak classifiers. A monolithic boosted
classifier does not qualify. Use integral images and variance normalization;
constant patches exist, so avoid division by zero and keep every score finite.

The supplied train/images contains 5,931 binary PGM patches with canonical
header P5\\n19 19\\n255\\n. train/labels.txt maps IDs to labels: 1 is face (2,065),
0 is non-face (3,866). IDs carry no label information. Use only this data.
Use an internal training split before spending hidden-validation calls.
Later cascade stages must train on non-faces surviving earlier stages.
There are few negatives: handle their depletion deliberately and explain how.

## Learner interface (Linux executables, compiled through WSL)

make must produce train and predict in learner/:

    ./train <images_dir> <labels.txt> <model_out>
    ./predict <model> <images_dir>
    ./predict --describe <model>

Prediction prints exactly one `<id> <score>` per .pgm, higher = more face-like.
For an image stopping at one-based stage k, the score must be in [k-1, k).
Use (k-1) + sigmoid(margin), with sigmoid clamped away from 0 and 1 so
floating-point rounding cannot cross a stage boundary. Within the same stage,
order by margin. For WaldBoost each weak classifier is a stage.
--describe reads only the model and prints `stage=<k> weak=<n>` per stage:
at least two stages, consecutive k starting at 1, positive weak counts.
The declared layout and scores must represent the actual algorithm.

## Validation and rules

Use the validation MCP server's run_validation tool. It builds and trains your
learner with a 30-minute training timeout, then privately predicts 364 faces
and 682 non-faces. Feedback is aggregate AUC, detection and precision at
1%, 5%, and 10% FP, timings, and declared cascade statistics. The call budget
persists across reconnects. Validate is easy: sound design and speed matter.
Final test targets are 0.1%, 1%, and 10% FP on a separate, harder test set.

Train only on train/. Do not read other learners, the labeled repository,
private validation/test data, or version history, or download data/packages.
Do not change the validation configuration or inspect its implementation.
Learner processes are sandboxed with no network. This personal Codex editing
session does not reproduce Andrew's agent tool restrictions or audit hooks.

When finished, leave working source in learner/ and summarize the algorithm,
stage layout, handling of scarce negatives, hyperparameters, results and times.
"""
    instructions += (f"\nCompile from Windows using:\n\n"
                     f"    wsl.exe -d {distro} --cd '{ws}' -- make -C learner\n")
    (ws / "AGENTS.md").write_text(instructions)
    write_workspace_config(ws, agent, run_id, distro, max_calls)
    print(f"Linux workspace: {ws}")
    if shutil.which("wslpath"):
        windows_path = subprocess.check_output(["wslpath", "-w", str(ws)], text=True).strip()
        print(f"Open this folder in Windows Codex: {windows_path}")
    print("learner/ is empty. Open a fresh chat in this workspace and read AGENTS.md.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    s = sub.add_parser("split")
    s.add_argument("name")
    s.add_argument("src")
    s.add_argument("out")
    s.add_argument("--seed", type=int, default=TEAM_SEED)
    p = sub.add_parser("prepare")
    p.add_argument("--seed", type=int, default=TEAM_SEED)
    w = sub.add_parser("workspace")
    w.add_argument("agent")
    w.add_argument("run_id")
    w.add_argument("--distro", default="Ubuntu")
    w.add_argument("--max-calls", type=int, default=20)
    args = ap.parse_args()

    if args.cmd == "doctor":
        doctor()
    elif args.cmd == "split":
        stage_split(args.name, args.src, args.out, args.seed)
    elif args.cmd == "prepare":
        prepare_data(args.seed)
    else:
        doctor()
        build_workspace(args.agent, args.run_id, args.distro, args.max_calls)


if __name__ == "__main__":
    main()
