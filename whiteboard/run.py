"""The whiteboard pipeline, start to end.

    for each round:                                    big loop
        repeat until we have K new rules:              small loop
            LLM writes a candidate rule                  box 1
            Harness checks it and times it               box 2
            not good enough -> tell the LLM, try again
            good enough -> keep it, reweight training (boosting)
        Combiner builds classifier + cascade              box 3
        Evaluator scores it on validation                 box 4
        the report goes back to the LLM                   big arrow

Run from the repo root:
    python -m whiteboard.run --fake              no Claude, random Haar rules (quick test)
    python -m whiteboard.run --rounds 4          real run with Claude
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from common.data import load_dataset
from whiteboard import combiner, evaluator, harness, llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4, help="big-loop rounds")
    ap.add_argument("--rules-per-round", type=int, default=3)
    ap.add_argument("--tries", type=int, default=4, help="small-loop tries per rule")
    ap.add_argument("--bar", type=float, default=0.45, help="keep a rule only if weighted error is below this")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--fake", action="store_true", help="random Haar rules instead of Claude")
    ap.add_argument("--audit-frac", type=float, default=0.2,
                    help="share of train/ held out from everything (same split as the team's main.py)")
    a = ap.parse_args()

    d = load_dataset(".", audit_frac=a.audit_frac, seed=0)     # the CBCL test set is never touched here
    Xtr, ytr = d.images[d.train_idx], d.y[d.train_idx]
    Xva, yva = d.images[d.val_idx], d.y[d.val_idx]
    w = np.where(ytr == 1, 0.5 / (ytr == 1).sum(), 0.5 / (ytr == -1).sum())  # faces and non-faces half each
    out = Path("runs") / time.strftime("whiteboard-%m%d-%H%M%S")
    out.mkdir(parents=True)
    print(f"train {len(ytr)} images, validate {len(yva)}, writing to {out}")

    rules, report, history = [], "", []
    for rnd in range(1, a.rounds + 1):
        for k in range(a.rules_per_round):
            note = ""
            for t in range(1, a.tries + 1):
                code = llm.write_rule(rules, Xtr, ytr, w, note, report, a.model, a.fake)       # box 1
                try:
                    r = harness.check(code, Xtr, ytr, w)                                       # box 2
                except Exception as e:
                    note = f"Your code failed in the harness: {type(e).__name__}: {e}\n{code}"
                    print(f"  round {rnd} rule {k + 1} try {t}: crashed ({e})")
                    continue
                if r["err"] >= a.bar:
                    note = (f"Not good enough: weighted error {r['err']:.3f}, need below {a.bar}. "
                            f"Try a different region or idea.\n{code}")
                    print(f"  round {rnd} rule {k + 1} try {t}: error {r['err']:.3f}, not good enough")
                    continue
                # Good enough: keep it and reweight the training set (AdaBoost).
                r["alpha"] = 0.5 * np.log((1 - r["err"]) / max(r["err"], 1e-10))
                w = w * np.exp(-r["alpha"] * ytr * harness.predict(r, r["train_scores"]))
                w /= w.sum()
                r["val_scores"], _ = harness.scores(harness.load(code), Xva)
                rules.append(r)
                print(f"  round {rnd} rule {k + 1} try {t}: kept, error {r['err']:.3f}, {r['us']:.1f} us")
                break
        if not rules:
            print("no rule kept yet, nothing to combine")
            continue
        model = combiner.build(rules, ytr)                                                    # box 3
        result = evaluator.evaluate(rules, model, yva)                                        # box 4
        report = result.pop("text")                                                           # big arrow
        print(report)
        history.append({"round": rnd, **result})

    (out / "rules.json").write_text(json.dumps(
        [{k: r[k] for k in ("code", "thr", "pol", "err", "us", "alpha")} for r in rules], indent=1))
    (out / "stages.json").write_text(json.dumps(model["stages"]))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "report.txt").write_text(report)
    print(f"saved {len(rules)} rules to {out}")


if __name__ == "__main__":
    main()
