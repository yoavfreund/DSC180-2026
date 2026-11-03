#!/usr/bin/env python3
"""check_local.py - free local checks to run BEFORE spending a run_validation call.

Run from the workspace root:   python3 check_local.py [--quick]

It splits train/ 80/20 (stratified, seeded), builds learner/, trains on the 80%,
predicts the 20%, and reports:
  - contract problems the real harness would reject (make/train/predict failures,
    --describe format, missing/duplicate/extra ids, non-finite scores),
  - the cascade exit-stage statistics the harness computes (floor(score) = exit stage),
  - AUC and detection rate at 1%/5%/10% false positives on the holdout,
  - a flat-patch test (constant images must give finite scores).

It uses only train/. The holdout is a part of train that the learner does not see
during this check, so the numbers are a rough guide, not the validation result.
Python standard library only.
"""
import argparse
import math
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEARNER = ROOT / "learner"
SCRATCH = ROOT / ".check"
STAGE_LINE = re.compile(r"^stage=(\d+) weak=(\d+)$")
FP_RATES = (0.01, 0.05, 0.10)

problems, warnings = [], []


def ok(msg):
    print(f"OK    {msg}")


def info(msg):
    print(f"INFO  {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"WARN  {msg}")


def fail(msg):
    problems.append(msg)
    print(f"FAIL  {msg}")


def tail(text, lines=12):
    return "\n".join((text or "").strip().splitlines()[-lines:])


def run(cmd, timeout, cwd=LEARNER):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, time.time() - t0
    return r, time.time() - t0


def read_labels(path):
    labels = {}
    for line in path.read_text().splitlines():
        if line.strip():
            sid, lab = line.split()
            labels[sid] = int(lab)
    return labels


def make_split(args):
    """Stratified train/holdout split of train/, copied under .check/."""
    labels = read_labels(ROOT / "train" / "labels.txt")
    rng = random.Random(args.seed)
    fit, hold = [], []
    for lab in (0, 1):
        ids = sorted(i for i, l in labels.items() if l == lab)
        rng.shuffle(ids)
        cut = int(len(ids) * args.holdout)
        hold += ids[:cut]
        fit += ids[cut:]
    if args.quick:
        rng.shuffle(fit)
        fit = fit[: max(400, len(fit) // 4)]
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    for name, ids in (("fit", fit), ("hold", hold)):
        (SCRATCH / name / "images").mkdir(parents=True)
        for sid in ids:
            shutil.copyfile(ROOT / "train" / "images" / f"{sid}.pgm", SCRATCH / name / "images" / f"{sid}.pgm")
    (SCRATCH / "fit" / "labels.txt").write_text("".join(f"{i} {labels[i]}\n" for i in sorted(fit)))
    hold_labels = {i: labels[i] for i in hold}
    # Flat patches: a variance-normalizing learner must not divide by zero.
    (SCRATCH / "flat" / "images").mkdir(parents=True)
    for name, val in (("flat000", 0), ("flat128", 128), ("flat255", 255)):
        (SCRATCH / "flat" / "images" / f"{name}.pgm").write_bytes(b"P5\n19 19\n255\n" + bytes([val]) * 361)
    near = bytearray([128]) * 361
    near[0] = 129
    (SCRATCH / "flat" / "images" / "nearflat.pgm").write_bytes(b"P5\n19 19\n255\n" + bytes(near))
    return fit, hold_labels


def parse_scores(text):
    """Return (scores dict, problems list). Every line must be '<id> <finite number>'."""
    scores, bad, dup = {}, 0, 0
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            bad += 1
            continue
        try:
            val = float(parts[1])
        except ValueError:
            bad += 1
            continue
        if not math.isfinite(val):
            bad += 1
            continue
        if parts[0] in scores:
            dup += 1
        scores[parts[0]] = val
    return scores, bad, dup


def auc(pos, neg):
    """Rank-based AUC with tied scores counted half."""
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        rank_sum += avg_rank * sum(l for _, l in allv[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def detection_at_fp(scores, labels, rate):
    """Best detection rate whose false positives stay within floor(rate * n_neg); ties all in or all out."""
    n_pos = sum(labels.values())
    n_neg = len(labels) - n_pos
    allowed = int(rate * n_neg)
    order = sorted(scores, key=lambda i: -scores[i])
    best_tp = tp = fp = i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        tp += sum(labels[k] for k in order[i:j])
        fp += sum(1 - labels[k] for k in order[i:j])
        if fp <= allowed:
            best_tp = max(best_tp, tp)
        i = j
    return best_tp / n_pos, allowed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", type=float, default=0.2, help="fraction of train held out (default 0.2)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="train on about a quarter of the fit set, for fast iteration")
    ap.add_argument("--train-timeout", type=int, default=1800)
    args = ap.parse_args()

    if not (ROOT / "train" / "labels.txt").exists() or not LEARNER.exists():
        sys.exit("run this from the workspace root (it needs train/ and learner/)")

    # 1. build
    r, secs = run(["make"], 120)
    if r is None or r.returncode != 0:
        fail("make failed:\n" + tail((r.stderr or r.stdout) if r else "timed out after 120 s"))
        return finish()
    for exe in ("train", "predict"):
        if not (LEARNER / exe).exists():
            fail(f"make did not produce ./{exe}")
    if problems:
        return finish()
    ok(f"make ({secs:.1f} s)")

    # 2. split + train
    fit, hold_labels = make_split(args)
    n_hold_pos = sum(hold_labels.values())
    info(f"fit {len(fit)} images, holdout {len(hold_labels)} ({n_hold_pos} faces, {len(hold_labels) - n_hold_pos} non-faces)"
         + ("  [--quick]" if args.quick else ""))
    model = SCRATCH / "model"
    r, secs = run(["./train", str(SCRATCH / "fit" / "images"), str(SCRATCH / "fit" / "labels.txt"), str(model)],
                  args.train_timeout)
    if r is None:
        fail(f"train timed out after {args.train_timeout} s")
        return finish()
    if r.returncode != 0:
        fail(f"train exited {r.returncode}:\n" + tail(r.stderr))
        return finish()
    if not model.exists():
        fail("train did not write the model file")
        return finish()
    ok(f"train ({secs:.1f} s wall)")

    # 3. describe
    r, _ = run(["./predict", "--describe", str(model)], 30)
    stages = [STAGE_LINE.match(l.strip()) for l in (r.stdout.splitlines() if r else []) if l.strip()]
    if r is None or r.returncode != 0 or not stages or not all(stages):
        fail("`./predict --describe MODEL` must print one `stage=<k> weak=<n>` line per stage (and nothing else)")
        return finish()
    cascade = [int(m.group(2)) for m in stages]
    if len(cascade) < 2:
        fail(f"only {len(cascade)} stage reported; a cascade needs at least 2")
        return finish()
    if len(cascade) <= 12:
        ok(f"describe: {len(cascade)} stages, weak per stage {','.join(map(str, cascade))}")
    else:
        ok(f"describe: {len(cascade)} stages, {sum(cascade)} weak classifiers in total")

    # 4. predict on holdout
    r, secs = run(["./predict", str(model), str(SCRATCH / "hold" / "images")], 300)
    if r is None or r.returncode != 0:
        fail("predict failed on the holdout:\n" + tail(r.stderr if r else "timed out"))
        return finish()
    scores, bad, dup = parse_scores(r.stdout)
    if bad:
        fail(f"{bad} output lines are malformed or have non-finite scores (need '<id> <finite number>')")
    if dup:
        fail(f"{dup} ids were scored more than once")
    missing = set(hold_labels) - set(scores)
    extra = set(scores) - set(hold_labels)
    if missing:
        fail(f"{len(missing)} images were not scored (every .pgm must get exactly one line)")
    if extra:
        fail(f"{len(extra)} scored ids do not match any image")
    if problems:
        return finish()
    ok(f"predict scored all {len(scores)} holdout images ({secs / len(scores) * 1e3:.2f} ms/image wall, includes start-up)")

    # 5. cascade statistics, the way the harness reads them
    k = len(cascade)
    cum = [sum(cascade[: i + 1]) for i in range(k)]
    out_of_range = sum(1 for s in scores.values() if s < 0 or s >= k)
    on_integer = sum(1 for s in scores.values() if s == math.floor(s))
    if out_of_range:
        fail(f"{out_of_range} scores fall outside [0, {k}); the harness clamps the exit stage, so the cascade "
             "convention (rejected at stage k -> score in [k, k+1), 0-based) is broken")
    if on_integer:
        warn(f"{on_integer} scores are exact integers; squashing probably saturated (a score of exactly k+1 would "
             "count as exiting one stage later). Scale margins before the sigmoid")
    per_stage = {1: [0] * k, 0: [0] * k}
    for sid, s in scores.items():
        per_stage[hold_labels[sid]][min(max(int(math.floor(s)), 0), k - 1)] += 1
    n_neg = len(hold_labels) - n_hold_pos
    weak = {lab: sum(cum[st] * c for st, c in enumerate(per_stage[lab])) / max(1, sum(per_stage[lab])) for lab in (0, 1)}
    n_pos = n_hold_pos
    if k <= 12:
        info(f"exit stage counts (stage 1..{k})   faces: {per_stage[1]}   non-faces: {per_stage[0]}")
    else:  # many stages (e.g. one weak classifier per stage): show how many images are still alive at checkpoints
        marks = [m for m in (1, 2, 5, 10, 20, 50, 100, 200, 500) if m < k]
        alive = lambda lab, total, m: total - sum(per_stage[lab][:m])
        info(f"images still alive after stage {', '.join(map(str, marks))}, then at the last stage ({k}):")
        info("  non-faces: " + ", ".join(str(alive(0, n_neg, m)) for m in marks) + f", {per_stage[0][k - 1]}  (of {n_neg})")
        info("  faces:     " + ", ".join(str(alive(1, n_pos, m)) for m in marks) + f", {per_stage[1][k - 1]}  (of {n_pos})")
    info(f"weak classifiers evaluated per image: non-faces {weak[0]:.1f}, faces {weak[1]:.1f} (of {cum[-1]})")
    if weak[0] > 0.8 * cum[-1]:
        warn("non-faces evaluate most of the cascade; there is little early rejection, so this is barely a cascade")
    if sum(1 for c in per_stage[0] if c) == 1 and k > 1:
        warn("all non-faces exit at the same stage; check that earlier stages actually reject")

    # 6. accuracy on the holdout
    pos = [s for sid, s in scores.items() if hold_labels[sid]]
    neg = [s for sid, s in scores.items() if not hold_labels[sid]]
    parts = []
    for rate in FP_RATES:
        det, allowed = detection_at_fp(scores, hold_labels, rate)
        parts.append(f"FP<={rate:.0%} ({allowed} allowed): detection {det:.3f}")
    info(f"holdout AUC {auc(pos, neg):.4f}   " + "   ".join(parts))

    # 7. flat patches
    r, _ = run(["./predict", str(model), str(SCRATCH / "flat" / "images")], 60)
    if r is None or r.returncode != 0:
        fail("predict crashed or failed on flat (constant) patches")
    else:
        flat, bad, dup = parse_scores(r.stdout)
        if bad or dup or len(flat) != 4:
            fail("flat (constant) patches must each get exactly one finite score")
        else:
            ok("flat and near-flat patches give finite scores")
    finish()


def finish():
    print()
    if problems:
        print(f"RESULT: FAILED ({len(problems)} problem{'s' if len(problems) != 1 else ''}); the harness would reject or mis-score this")
        sys.exit(1)
    print("RESULT: PASSED" + (f" with {len(warnings)} warning(s)" if warnings else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()