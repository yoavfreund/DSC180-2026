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
    ap.add_argument("--tries", type=int, default=10, help="most small-loop tries per rule before the run stops")
    ap.add_argument("--bar", type=float, default=0.35, help="keep a rule only if weighted error is below this")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--fake", action="store_true", help="random Haar rules instead of Claude")
    ap.add_argument("--audit-frac", type=float, default=0.2,
                    help="share of train/ held out from everything (same split as the team's main.py)")
    a = ap.parse_args()

    d = load_dataset(".", audit_frac=a.audit_frac, seed=0)     # the CBCL test set is never touched here
    raw_tr, ytr = d.images[d.train_idx], d.y[d.train_idx]          # raw pixels: only for showing Claude
    Xtr = harness.normalize(raw_tr)                                  # what every rule sees
    Xva, yva = harness.normalize(d.images[d.val_idx]), d.y[d.val_idx]
    w = np.where(ytr == 1, 0.5 / (ytr == 1).sum(), 0.5 / (ytr == -1).sum())  # faces and non-faces half each
    out = Path("runs") / time.strftime("whiteboard-%m%d-%H%M%S")
    out.mkdir(parents=True)
    print(f"train {len(ytr)} images, validate {len(yva)}, writing to {out}")

    def find_rule(label):
        """Small loop: keep Claude working until a rule beats the bar. None if it never does."""
        tries = []                                   # every try for this rule, shown back to Claude
        for t in range(1, a.tries + 1):
            code = llm.write_rule(rules, raw_tr, ytr, w, "\n\n".join(tries[-5:]), report, a.model, a.fake)  # box 1
            try:
                r = harness.check(code, Xtr, ytr, w)                                                     # box 2
            except Exception as e:
                tries.append(f"Try {t} failed in the harness: {type(e).__name__}: {e}\n{code}")
                print(f"  {label} try {t}: rejected ({e})")
                continue
            if r["err"] < a.bar:
                print(f"  {label} try {t}: kept, error {r['err']:.3f}, best of {r['tried']} settings {r['params']}")
                return r
            tries.append(f"Try {t}: weighted error {r['err']:.3f} at best knobs {r['params']}, "
                         f"not good enough (need below {a.bar}).\n{code}")
            print(f"  {label} try {t}: error {r['err']:.3f} (best of {r['tried']}), not good enough")
        return None

    rules, report, history = [], "", []
    for rnd in range(1, a.rounds + 1):
        for k in range(a.rules_per_round):
            r = find_rule(f"round {rnd} rule {k + 1}")
            if r is None:
                break
            # Good enough: keep it and reweight the training set (AdaBoost).
            r["alpha"] = 0.5 * np.log((1 - r["err"]) / max(r["err"], 1e-10))
            w = w * np.exp(-r["alpha"] * ytr * harness.predict(r, r["train_scores"]))
            w /= w.sum()
            r["val_scores"], _ = harness.scores(harness.load(r["code"])[0], Xva, r["params"])
            rules.append(r)
        if r is None:
            print(f"no rule reached the bar {a.bar} in {a.tries} tries: stopping with {len(rules)} rules")
        if not rules:
            return
        model = combiner.build(rules, ytr)                                                    # box 3
        result = evaluator.evaluate(rules, model, yva)                                        # box 4
        report = result.pop("text")                                                           # big arrow
        print(report)
        history.append({"round": rnd, **result})
        if r is None:
            break

    (out / "rules.json").write_text(json.dumps(
        [{k: r[k] for k in ("code", "params", "thr", "pol", "err", "us", "alpha")} for r in rules], indent=1))
    (out / "stages.json").write_text(json.dumps(model["stages"]))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    (out / "report.txt").write_text(report)
    print(f"saved {len(rules)} rules to {out}")


if __name__ == "__main__":
    main()
