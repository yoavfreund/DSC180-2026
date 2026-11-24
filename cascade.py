"""Choose early exits for a finished rule chain using validation trajectories.

  .venv/bin/python cascade.py --run-dir runs/NAME [--rules 16] [--face-loss 0.01]

Writes the cascade configuration and prints validation quality and rule counts.
"""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from common.data import load_labeled_folder
from common.metrics import FP_RATES, auc, detection_at_fp
from ensemble import apply_cascade, calibrate_trajectories, learner_outputs, running_scores


def fit_cascade(run_dir, images, y, face_loss=0.01, rules=None, cuts=None):
    with TemporaryDirectory(prefix="dsc180-calibrate-") as tmp:
        entries, H = learner_outputs(run_dir, images, rules, build_dir=tmp)
    alphas = np.array([e["alpha"] for e in entries])
    confidence = running_scores(H, alphas)
    stages = calibrate_trajectories(confidence, y, face_loss, cuts)
    config = {
        "algorithm": "validation_trajectories",
        "calibrated_on": "validate",
        "face_loss": face_loss,
        "rules": len(entries),
        "source_sha256": [e["source_sha256"] for e in entries],
        "stages": stages,
    }
    return config, entries, confidence


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rules", type=int, help="use this many accepted rules from the start of the chain")
    ap.add_argument("--data", default=os.environ.get("FACE_DATA", "."), help="folder that holds validate/")
    ap.add_argument("--face-loss", type=float, default=0.01, help="total share of validation faces the cuts may reject")
    ap.add_argument("--cuts", help="restrict candidate positions, e.g. 1,2,4,8 (default: all before the end)")
    ap.add_argument("--output", type=Path, help="override the cascade JSON output path")
    args = ap.parse_args()

    images, y, _ = load_labeled_folder(Path(args.data) / "validate")
    cuts = [int(c) for c in args.cuts.split(",")] if args.cuts else None
    config, entries, F_cum = fit_cascade(args.run_dir, images, y, args.face_loss, args.rules, cuts)
    alphas = np.array([e["alpha"] for e in entries])
    T = len(entries)
    stages = config["stages"]
    score, used = apply_cascade(F_cum, stages, alphas, rank_exits=False)

    faces = y > 0
    print(f"{T} rules; {int(faces.sum())} validation faces, {int((~faces).sum())} validation non-faces; "
          f"face-loss budget {args.face_loss:.3f} across {len(stages)} selected checkpoints")
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
    default_name = "cascade.json" if args.rules is None else f"cascade-{args.rules}.json"
    out = args.output or Path(args.run_dir) / default_name
    out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
