"""Score a labeled image folder with a finished run, in the shared evaluator's format.

  .venv/bin/python predict.py --run-dir runs/NAME --images DIR --scores OUT.txt --key KEY.txt [--cascade]
  eval/eval OUT.txt KEY.txt --fp-rates 0.01,0.05,0.1

DIR holds face/ and non-face/ PGM folders (validate/, or the test set on test day).
SCORES gets one "<id> <score>" line per image, higher = more face-like: the full
chain sum(alpha * h), or with --cascade the cascade score from RUN_DIR/cascade.json.
KEY gets one "<id> <label>" line per image, 1 = face, 0 = non-face.
--source cbcl|cmu keeps only images from that source (cmu_* files came from the original CBCL
test download), both classes, so each source can be evaluated on its own.
"""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from common.data import SOURCES, load_labeled_folder, source_of
from ensemble import chain_prefix
from harness.build import write_images
from harness.detector import build_detector, score_detector


def write_predictions(scores_path, key_path, scores, y, ids):
    with open(scores_path, "x") as s, open(key_path, "x") as k:
        for i, sc, lab in zip(ids, scores, y):
            s.write(f"{i} {sc:.17g}\n")
            k.write(f"{i} {1 if lab > 0 else 0}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--cascade", action="store_true")
    ap.add_argument("--rules", type=int, help="use this many accepted rules from the start of the chain")
    ap.add_argument("--cascade-file", type=Path, help="use early exits from this cascade JSON")
    ap.add_argument("--source", choices=SOURCES, help="score only images from this source")
    args = ap.parse_args()

    use_cascade = args.cascade or args.cascade_file is not None
    config = {}
    if use_cascade:
        name = "cascade.json" if args.rules is None else f"cascade-{args.rules}.json"
        config = json.loads((args.cascade_file or Path(args.run_dir) / name).read_text())
    entries = chain_prefix(args.run_dir, args.rules if args.rules is not None else config.get("rules"))
    if "rules" in config and config["rules"] != len(entries):
        ap.error("cascade configuration was calibrated for a different chain length")
    if "source_sha256" in config and config["source_sha256"] != [e["source_sha256"] for e in entries]:
        ap.error("cascade configuration does not match the run's saved rules")
    images, y, ids = load_labeled_folder(args.images)
    if args.source:
        keep = np.array([source_of(i) == args.source for i in ids])
        if not (keep & (y > 0)).any() or not (keep & (y < 0)).any():
            ap.error(f"no faces or no non-faces from source {args.source} under {args.images}")
        images, y, ids = images[keep], y[keep], [i for i, k in zip(ids, keep) if k]
    with TemporaryDirectory(prefix="dsc180-predict-") as tmp:
        tmp = Path(tmp)
        legacy_scores = use_cascade and "algorithm" not in config
        binary = build_detector(args.run_dir, entries, config.get("stages", []), tmp / "build",
                                rank_exits=legacy_scores)
        images_file = tmp / "images.bin"
        write_images(images, images_file)
        score, used = score_detector(binary, images_file, len(images), tmp / "scores.bin", use_cascade)
    write_predictions(args.scores, args.key, score, y, ids)
    print(f"wrote {len(ids)} scores to {args.scores} and labels to {args.key}")
    if use_cascade:
        print(f"mean rules evaluated per patch: {used.mean():.2f}")


if __name__ == "__main__":
    main()
