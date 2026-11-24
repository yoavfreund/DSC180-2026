"""Pool the CBCL training and test sets and split them again at random (Yoav, Nov 2026).

The downloaded CBCL test set comes from a different source (every file is cmu_*), so a model
trained on the CBCL training set and scored on it measures distribution shift as well as
generalization. Pooling both and re-splitting makes train, validate and test draws from one
distribution. The counts per class stay what they were:

  split      faces  non-faces
  train      2,065      3,866
  validate     364        682
  test         472     23,573

Every draw is stratified by class AND source (cbcl or cmu), so each split has the same source mix
within a class. Identical images (same pixels) are kept together in one split. The source is read
from the file name: cmu_* is cmu, everything else is cbcl.

  .venv/bin/python make_split.py --cbcl ~/dsc180-private/raw/cbcl-train \\
      --cmu ~/dsc180-private/raw/test --train train --validate validate \\
      --test ~/dsc180-private/pooled/test --manifest split_manifest.tsv

Inputs hold face/ and non-face/ folders of the original PGM files. Every output folder must be
empty or missing. Files are copied byte for byte. The manifest lists every image with its split,
class and source; the same inputs and seed always give the same manifest.
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np

from common.data import SOURCES, read_pgm, source_of

CLASSES = ("face", "non-face")
SPLITS = ("train", "validate", "test")
COUNTS = {"test": {"face": 472, "non-face": 23573}, "validate": {"face": 364, "non-face": 682}}
SEED = 20261123


def allocate(total, sizes):
    """Split `total` across strata in proportion to `sizes` (largest remainder, ties by order)."""
    sizes = np.asarray(sizes, dtype=float)
    exact = total * sizes / sizes.sum()
    out = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - out), kind="stable")[: total - out.sum()]:
        out[i] += 1
    return out


def collect(cbcl, cmu):
    """{(class, source): [(name, path, pixel hash)]} sorted by name; refuses mislabeled sources."""
    pool = {}
    for root, want in ((cbcl, "cbcl"), (cmu, "cmu")):
        for cls in CLASSES:
            for p in sorted(Path(root, cls).glob("*.pgm")):
                if source_of(p.name) != want:
                    raise ValueError(f"{p} is in the {want} folder but its name says {source_of(p.name)}")
                digest = hashlib.sha256(read_pgm(p).tobytes()).hexdigest()
                pool.setdefault((cls, want), []).append((p.name, p, digest))
    for (cls, src), items in pool.items():
        names = [n for n, _, _ in items]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate file names in {cls}/{src}")
    return pool


def assign(pool, seed=SEED, counts=COUNTS):
    """{(class, source, name): split}. Draws test, then validate, from groups of identical images."""
    rng = np.random.default_rng(seed)
    out = {}
    for cls in CLASSES:
        strata = [s for s in SOURCES if (cls, s) in pool]
        sizes = [len(pool[cls, s]) for s in strata]
        quota = {split: dict(zip(strata, allocate(counts[split][cls], sizes))) for split in ("test", "validate")}
        for src in strata:
            groups = {}
            for name, _, digest in pool[cls, src]:
                groups.setdefault(digest, []).append(name)
            units = sorted(groups.values())               # each unit is one set of identical images
            order = rng.permutation(len(units))
            left = {split: int(quota[split][src]) for split in ("test", "validate")}
            for i in order:
                unit = units[i]
                split = next((s for s in ("test", "validate") if len(unit) <= left[s]), "train")
                if split != "train":
                    left[split] -= len(unit)
                for name in unit:
                    out[cls, src, name] = split
            if any(left.values()):
                raise ValueError(f"could not fill the {cls}/{src} quota exactly: {left} left over")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cbcl", type=Path, required=True, help="original CBCL training set (face/, non-face/)")
    ap.add_argument("--cmu", type=Path, required=True, help="original CBCL test set, all cmu_* (face/, non-face/)")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--validate", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    dirs = {"train": args.train, "validate": args.validate, "test": args.test}
    for d in dirs.values():
        if d.exists() and any(d.rglob("*.pgm")):
            ap.error(f"{d} already holds PGM files; move them away first")
    pool = collect(args.cbcl, args.cmu)
    split = assign(pool, args.seed)

    rows = []
    for (cls, src), items in sorted(pool.items()):
        for name, path, _ in items:
            s = split[cls, src, name]
            dest = dirs[s] / cls
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest / name)
            rows.append((s, cls, src, name))
    rows.sort(key=lambda r: (SPLITS.index(r[0]), CLASSES.index(r[1]), r[3]))
    with open(args.manifest, "w") as f:
        f.write(f"# make_split.py seed={args.seed}\nsplit\tclass\tsource\tfile\n")
        f.writelines("\t".join(r) + "\n" for r in rows)

    print(f"{'split':9s} {'class':9s} {'cbcl':>6s} {'cmu':>6s} {'total':>6s}")
    for s in SPLITS:
        for cls in CLASSES:
            n = {src: sum(1 for r in rows if r[:3] == (s, cls, src)) for src in SOURCES}
            print(f"{s:9s} {cls:9s} {n['cbcl']:6d} {n['cmu']:6d} {sum(n.values()):6d}")
    print(f"wrote {len(rows)} rows to {args.manifest}")


if __name__ == "__main__":
    main()
