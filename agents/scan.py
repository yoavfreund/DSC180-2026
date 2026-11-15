"""scan: try every placement of a template rule on the weighted TRAINING images.

The agent writes its idea once, with P0..P3 where it is unsure of a position or
size, and gives a range for each. The harness builds the template once, scores
every combination in C, fits a threshold for each, and returns the best
concrete rules, ready to submit. The LLM chooses the kind of rule; the search
chooses the numbers.
"""
import numpy as np

from coordinator.brief import one_line
from harness.build import build_template, expand_ranges, run_scan, substitute
from harness.threshold import fit_stump


def scan_on_train(template, ranges, data, build_dir, cache_dir, top=5):
    """Returns (report text, best concrete source or None). Raises LearnerError on a bad template."""
    combos = expand_ranges(ranges)
    build = build_template(template, build_dir, cache_dir)
    scores = run_scan(build.binary, data.images_path, data.n, combos)
    errs = np.full(len(combos), np.inf)
    for k, row in enumerate(scores):
        if np.all(np.isfinite(row)):
            errs[k] = fit_stump(row, data.y, data.w).error
    finite = np.isfinite(errs)
    where = (f"a sample of {data.n} training images drawn in proportion to their weight" if data.sampled
             else "the weighted training images")
    lines = [f"Scanned {len(combos)} placements on {where}. The best of many "
             f"is optimistic: expect a higher error on the hidden validation images."]
    if not finite.any():
        return "\n".join(lines + ["Every placement returned NaN or infinity."]), None
    if not finite.all():
        lines.append(f"{int((~finite).sum())} placements returned NaN or infinity and were skipped.")
    order = np.argsort(errs, kind="stable")[:top]
    best = None
    for rank, k in enumerate(order, 1):
        if not np.isfinite(errs[k]):
            break
        concrete = substitute(template, combos[k])
        best = best or concrete
        lines.append(f"{rank}. P0..P3 = {list(combos[k])}: training error {errs[k]:.4f}\n   {one_line(concrete, 600)}")
    return "\n".join(lines), best
