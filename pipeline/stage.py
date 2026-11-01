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
  stage.py split NAME SRC_DIR OUT_DIR [--seed N]   stage any split (incl. test)
  stage.py prepare [--seed N]                      stage repo train + validate
  stage.py workspace AGENT RUN_ID                  build an agent workspace

Locations (override with env vars):
  DSC180_PRIVATE  default ~/dsc180-private   staged splits, harness-only
  DSC180_WORK     default ~/dsc180-work      agent workspaces
"""
import argparse
import os
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRIVATE = Path(os.environ.get("DSC180_PRIVATE", Path.home() / "dsc180-private"))
WORK = Path(os.environ.get("DSC180_WORK", Path.home() / "dsc180-work"))

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


def build_workspace(agent, run_id):
    staged = PRIVATE / "staged" / "train"
    if not staged.exists():
        sys.exit(f"{staged} missing; run `stage.py prepare` first")
    ws = WORK / agent / run_id
    if ws.exists():
        sys.exit(f"{ws} already exists")
    # Only train goes in: images + labels. No ledger, no validate, no test, no .git.
    shutil.copytree(staged / "images", ws / "train" / "images", copy_function=shutil.copyfile)
    for p in sorted((ws / "train" / "images").iterdir()):
        os.utime(p, (FIXED_MTIME, FIXED_MTIME))
    shutil.copyfile(staged / "labels.txt", ws / "train" / "labels.txt")
    (ws / "learner").mkdir()
    print(ws)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
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
    args = ap.parse_args()

    if args.cmd == "split":
        stage_split(args.name, args.src, args.out, args.seed)
    elif args.cmd == "prepare":
        for name in ("train", "validate"):
            stage_split(name, REPO / name, PRIVATE / "staged" / name, args.seed)
    else:
        build_workspace(args.agent, args.run_id)


if __name__ == "__main__":
    main()
