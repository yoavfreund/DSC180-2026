"""Score a finished whiteboard run on held-out images. Nothing here goes back to the LLM.

    python -m whiteboard.score runs/whiteboard-XXXX --audit          the 20% audit share of train/
    python -m whiteboard.score runs/whiteboard-XXXX --images DIR     any face/ + non-face/ folder (test day)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common.data import load_dataset, load_labeled_folder
from whiteboard import evaluator, harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--images")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    run = Path(a.run_dir)
    rules = json.loads((run / "rules.json").read_text())
    model = {"alpha": np.array([r["alpha"] for r in rules]),
             "stages": json.loads((run / "stages.json").read_text())}
    if a.audit:
        d = load_dataset(".", audit_frac=0.2, seed=0)
        X, y = d.images[d.audit_idx], d.y[d.audit_idx]
    else:
        X, y, _ = load_labeled_folder(a.images)
    for r in rules:
        r["s"], _ = harness.scores(harness.load(r["code"]), X)
    out = evaluator.evaluate(rules, model, y, key="s")
    print(f"{len(y)} images ({(y == 1).sum()} faces)")
    print(out.pop("text"))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
