#!/usr/bin/env python3
"""Brute-force reference for eval.c, used to cross-check it on random inputs.

It recomputes everything the slow, obvious way: every candidate threshold is
tried directly, and AUC is computed over all positive/negative pairs. It shares
no code with eval.c.

  reference.py crosscheck EVAL_BIN [--trials N] [--seed S]
"""
import argparse
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def reference(scores, labels, targets, by_rate):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    n_pos, n_neg = len(pos), len(neg)
    pairs = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    lines = [f"n_pos={n_pos} n_neg={n_neg} auc={pairs / (n_pos * n_neg):.6f}"]
    for t in targets:
        allowed = math.floor(t * n_neg + 1e-9) if by_rate else math.floor(t + 1e-9)
        best = (0, 0, math.inf)  # (tp, fp, threshold): predicting nothing
        for th in sorted(set(scores), reverse=True):
            tp = sum(p >= th for p in pos)
            fp = sum(q >= th for q in neg)
            if fp <= allowed and (tp > best[0] or (tp == best[0] and fp < best[1])):
                best = (tp, fp, th)
        tp, fp, th = best
        head = f"fp_rate={t:g}" if by_rate else f"fp_count={t:g}"
        prec = f"{tp / (tp + fp):.6f}" if tp + fp else "NA"
        lines.append(f"{head} fp_allowed={allowed} threshold={th!r} tp={tp} fp={fp} "
                     f"tpr={tp / n_pos:.6f} fpr={fp / n_neg:.6f} precision={prec}")
    return lines


def parse(line):
    return dict(kv.split("=", 1) for kv in line.split())


def same(a, b):
    """Compare two output lines field by field; thresholds compare as floats."""
    da, db = parse(a), parse(b)
    if da.keys() != db.keys():
        return False
    for k in da:
        if k == "threshold" or k.startswith("fp_rate") or k.startswith("fp_count"):
            if float(da[k]) != float(db[k]):
                return False
        elif da[k] != db[k]:
            return False
    return True


def crosscheck(eval_bin, trials, seed):
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for trial in range(trials):
            n = rng.randint(2, 60)
            labels = [rng.randint(0, 1) for _ in range(n)]
            labels[0], labels[1] = 0, 1  # both classes present
            # Coarse rounding on some trials forces many ties.
            digits = rng.choice([0, 1, 2, 6])
            scores = [round(rng.gauss(0.8 * l, 1.0), digits) for l in labels]
            by_rate = rng.random() < 0.5
            targets = ([rng.choice([0, 0.01, 0.1, 0.25, 0.5, 1.0]) for _ in range(3)] if by_rate
                       else [rng.randint(0, n) for _ in range(3)])
            ids = [f"img{i:04d}" for i in range(n)]
            order = list(range(n))
            rng.shuffle(order)  # score file order must not matter
            (tmp / "key.txt").write_text("".join(f"{ids[i]} {labels[i]}\n" for i in range(n)))
            (tmp / "scores.txt").write_text("".join(f"{ids[i]} {scores[i]!r}\n" for i in order))
            flag = "--fp-rates" if by_rate else "--fp-counts"
            arg = ",".join(f"{t:g}" for t in targets)
            out = subprocess.run([eval_bin, tmp / "scores.txt", tmp / "key.txt", flag, arg],
                                 capture_output=True, text=True)
            got = out.stdout.splitlines()
            want = reference(scores, labels, targets, by_rate)
            if out.returncode != 0 or len(got) != len(want) or not all(map(same, got, want)):
                print(f"MISMATCH trial {trial} ({flag} {arg})", file=sys.stderr)
                print("  eval:     ", *got, out.stderr, sep="\n    ", file=sys.stderr)
                print("  reference:", *want, sep="\n    ", file=sys.stderr)
                return 1
    print(f"crosscheck: {trials} random trials match the brute-force reference")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crosscheck")
    c.add_argument("eval_bin")
    c.add_argument("--trials", type=int, default=500)
    c.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.exit(crosscheck(args.eval_bin, args.trials, args.seed))


if __name__ == "__main__":
    main()
