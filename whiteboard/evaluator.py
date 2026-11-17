"""Box 4: Evaluator. Checks the classifier and the cascade on the validation set.

Faces are rare in real scans, so the main number is detection rate at a fixed
false-alarm rate, not accuracy. The text report it returns is the big arrow on
the whiteboard: it goes back to the LLM for the next round.
"""
import numpy as np

from .combiner import votes


def detection_at(F, y, fp):
    """Share of faces caught when the threshold lets through `fp` of non-faces."""
    thr = np.quantile(F[y == -1], 1 - fp)
    return float(np.mean(F[y == 1] > thr))


def evaluate(rules, model, y, key="val_scores"):
    running = np.cumsum(model["alpha"][:, None] * votes(rules, key), axis=0)
    F = running[-1]
    out = {"rules": len(rules)}
    for fp in (0.01, 0.05, 0.10):
        out[f"det@{fp:.0%}FP"] = detection_at(F, y, fp)

    # Run the cascade: an image leaves at the first checkpoint it fails.
    us = np.array([r["us"] for r in rules])
    alive = np.ones(len(y), bool)
    cost = np.zeros(len(y))                     # microseconds spent on each image
    used = np.zeros(len(y))                     # rules run on each image
    start, stage_rows = 0, []
    for k, st in enumerate(model["stages"], 1):
        cost[alive] += us[start:st["end"]].sum()
        used[alive] += st["end"] - start
        fail = alive & (running[st["end"] - 1] < st["theta"])
        stage_rows.append((k, st["end"] - start, int((fail & (y == -1)).sum()), int((fail & (y == 1)).sum())))
        alive &= ~fail
        start = st["end"]
    out["cascade_det"] = float(alive[y == 1].mean())
    out["cascade_fp"] = float(alive[y == -1].mean())
    out["cascade_us"] = float(cost.mean())
    out["all_rules_us"] = float(us.sum())
    out["mean_rules_run"] = float(used.mean())

    lines = [f"Validation after {len(rules)} rules:",
             "  classifier: detection " + ", ".join(f"{out[k]:.1%} at {k[4:]}" for k in out if k.startswith("det@")),
             f"  cascade: catches {out['cascade_det']:.1%} of faces, lets through {out['cascade_fp']:.1%} "
             f"of non-faces, {out['mean_rules_run']:.2f} of {len(rules)} rules run per image, "
             f"{out['cascade_us']:.1f} us per image (all rules: {out['all_rules_us']:.1f} us)",
             "  stage | rules | non-faces thrown out | faces lost"]
    lines += [f"  {k:5} | {n:5} | {nf:20} | {f}" for k, n, nf, f in stage_rows]
    lines.append(f"  non-faces that pass every stage: {int(alive[y == -1].sum())}, "
                 f"faces missed: {int((~alive[y == 1]).sum())}")
    out["text"] = "\n".join(lines)
    return out
