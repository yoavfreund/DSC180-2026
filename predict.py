"""Score a labeled image folder with a finished run, in the shared evaluator's format.

  .venv/bin/python predict.py --run-dir runs/NAME --images DIR --scores OUT.txt --key KEY.txt [--cascade]
  eval/eval OUT.txt KEY.txt --fp-rates 0.01,0.05,0.1

DIR holds face/ and non-face/ PGM folders (validate/, or the test set on test day).
SCORES gets one "<id> <score>" line per image, higher = more face-like: the full
chain sum(alpha * h), or with --cascade the cascade score from RUN_DIR/cascade.json.
KEY gets one "<id> <label>" line per image, 1 = face, 0 = non-face.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common.data import load_labeled_folder
from ensemble import apply_cascade, learner_outputs, running_scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--cascade", action="store_true")
    args = ap.parse_args()

    images, y, ids = load_labeled_folder(args.images)
    entries, H = learner_outputs(args.run_dir, images)
    alphas = np.array([e["alpha"] for e in entries])
    F_cum = running_scores(H, alphas)
    if args.cascade:
        stages = json.loads((Path(args.run_dir) / "cascade.json").read_text())["stages"]
        score, _ = apply_cascade(F_cum, stages, alphas)
    else:
        score = F_cum[-1]
    with open(args.scores, "x") as s, open(args.key, "x") as k:
        for i, sc, lab in zip(ids, score, y):
            s.write(f"{i} {sc:.17g}\n")
            k.write(f"{i} {1 if lab > 0 else 0}\n")
    print(f"wrote {len(ids)} scores to {args.scores} and labels to {args.key}")


if __name__ == "__main__":
    main()
