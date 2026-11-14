"""Slice a finished run's chain into a cascade, calibrated on validation faces.

  .venv/bin/python cascade.py --run-dir runs/NAME [--face-loss 0.01] [--cuts 1,2,4,8]

Writes RUN_DIR/cascade.json and prints, on the validation images: faces kept and
non-faces left after each cut, detection at fixed FP rates with and without the
cascade, and average rules (and ns) per image. These numbers are calibrated and
measured on the same validation images, so they are optimistic; the test set
gives the real numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common.data import load_labeled_folder
from common.metrics import auc, detection_at_fp
from ensemble import apply_cascade, calibrate, default_cuts, learner_outputs, mean_ns_per_image, running_scores

FP_RATES = (0.001, 0.01, 0.1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data", default=".", help="folder that holds validate/")
    ap.add_argument("--face-loss", type=float, default=0.01, help="total share of validation faces the cuts may reject")
    ap.add_argument("--cuts", help="comma-separated rule counts, e.g. 1,2,4,8 (default: powers of 2 below T)")
    args = ap.parse_args()

    images, y, _ = load_labeled_folder(Path(args.data) / "validate")
    entries, H = learner_outputs(args.run_dir, images)
    alphas = np.array([e["alpha"] for e in entries])
    ns = [e["ns"] for e in entries]
    F_cum = running_scores(H, alphas)
    T = len(entries)
    cuts = [int(c) for c in args.cuts.split(",")] if args.cuts else default_cuts(T)
    stages = calibrate(F_cum, y, cuts, args.face_loss)
    score, used = apply_cascade(F_cum, stages, alphas)

    faces = y > 0
    print(f"{T} rules; {int(faces.sum())} validation faces, {int((~faces).sum())} validation non-faces; "
          f"face-loss budget {args.face_loss:.3f} split over {len(cuts)} cuts")
    print("after rules | theta    | faces kept | non-faces left")
    for st in stages:
        exit_by = used > st["after"]
        th = "none" if st["theta"] is None else f"{st['theta']:8.4f}"
        print(f"{st['after']:11d} | {th:>8} | {exit_by[faces].mean():10.4f} | {exit_by[~faces].mean():14.4f}")
    full = F_cum[-1]
    print(f"\nAUC: full chain {auc(full, y):.4f}, cascade {auc(score, y):.4f}")
    for r in FP_RATES:
        print(f"detection at FP rate {r:g}: full chain {detection_at_fp(full, y, r):.4f}, "
              f"cascade {detection_at_fp(score, y, r):.4f}")
    print(f"rules per image: full chain {T}, cascade {used.mean():.2f} "
          f"(non-faces {used[~faces].mean():.2f}, faces {used[faces].mean():.2f})")
    print(f"ns per image (rules only): full chain {sum(ns):.1f}, cascade {mean_ns_per_image(used, ns):.1f}")

    out = Path(args.run_dir) / "cascade.json"
    out.write_text(json.dumps({"face_loss": args.face_loss, "calibrated_on": "validate", "stages": stages},
                              indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
