"""The text each agent receives at the start of an attempt.

The agent has no file tools, so everything goes into one prompt: the task, the
C contract (vj_learner.h), a summary of the weighted TRAINING images, the rules
already in the ensemble, and this agent's own feedback. Every agent gets the
same facts; only the feedback section differs. Strategy belongs in each
agent's own .md (its system prompt), not here.
"""
import re

import numpy as np

from harness.build import HEADER, MAX_SOURCE_BYTES

RAMP = " .:-=+*#%@"


def _ramp_rows(image):
    levels = np.clip((image.astype(np.float64) / 256 * len(RAMP)).astype(int), 0, len(RAMP) - 1)
    return ["".join(RAMP[v] for v in row) for row in levels]


def _number_grid(values, title):
    lines = [title, "     x=" + "".join(f"{x:4d}" for x in range(19))]
    for y, row in enumerate(values):
        lines.append(f"y={y:2d}   " + "".join(f"{int(v):4d}" for v in row))
    return "\n".join(lines)


def data_summary(images, y, w, n_examples=3):
    """Weighted statistics of the TRAINING images only."""
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    faces = y > 0
    flat = images.reshape(len(images), -1).astype(np.float64)
    z = (flat - flat.mean(1, keepdims=True)) / np.maximum(flat.std(1, keepdims=True), 1.0)

    def wmean(mask):
        ww = w[mask] / w[mask].sum()
        return (ww[:, None] * z[mask]).sum(0).reshape(19, 19)

    diff = (wmean(faces) - wmean(~faces)) * 100
    face_share = w[faces].sum()
    parts = [
        f"Training images: {int(faces.sum())} faces, {int((~faces).sum())} non-faces. "
        f"Current weight on faces: {face_share:.3f}, on non-faces: {1 - face_share:.3f}.",
        f"Weight concentration: the heaviest 10% of images hold {np.sort(w)[::-1][:max(1, len(w) // 10)].sum():.3f} of the weight.",
        "",
        _number_grid(diff, "Weighted mean of normalized pixels, faces minus non-faces, x100 "
                           "(positive: faces brighter there than non-faces):"),
        "",
    ]
    for label, mask in (("faces", faces), ("non-faces", ~faces)):
        idx = np.flatnonzero(mask)
        top = idx[np.argsort(w[idx])[::-1][:n_examples]]
        rows = [_ramp_rows(images[i]) for i in top]
        parts.append(f"The {n_examples} heaviest (hardest) {label}, brightness ramp '{RAMP}' (dark to bright), "
                     f"weights {', '.join(f'{w[i]:.2e}' for i in top)}:")
        for r in range(19):
            parts.append("   ".join(rows[k][r] for k in range(len(rows))))
        parts.append("")
    return "\n".join(parts)


def describe(source):
    """The first C comment in the source, as the rule's one-line description."""
    m = re.search(r"/\*(.*?)\*/|//([^\n]*)", source, re.S)
    text = (m.group(1) or m.group(2) or "") if m else ""
    text = " ".join(text.split())
    return (text[:160] + "...") if len(text) > 160 else (text or "(no description comment)")


def history_text(ensemble):
    if not ensemble:
        return "No rules have been accepted yet. This is the first round."
    lines = ["round | agent | validation error | alpha | ns/image | description",
             "----- | ----- | ---------------- | ----- | -------- | -----------"]
    for m in ensemble:
        lines.append(f"{m['round']} | {m['agent']} | {m['val_err']:.4f} | {m['alpha']:.3f} | "
                     f"{m['ns']:.2f} | {m['description']}")
    return "\n".join(lines)


def brief(*, round_t, rounds, attempt, bar, deadline_s, summary, history, feedback):
    header = HEADER.read_text(encoding="utf-8")
    return f"""# Task: write ONE weak rule for a boosted face detector

We are building a Viola-Jones style face detector for 19x19 grayscale images by
AdaBoost. Each round, several agents each propose one weak rule. The coordinator
keeps one rule per round, then reweights the training images so the next round
focuses on the images the ensemble still gets wrong.

Round {round_t + 1} of {rounds}, attempt {attempt + 1}. Current error bar: {bar:.3f}.

## What to submit
C source that defines `double weak_score(const vj_image *im)`. The coordinator fits
the threshold and the polarity for you on the weighted training images, so return
a raw feature value. Start the source with a one-line comment that describes the
rule, for example `/* rule: eyes darker than the cheeks */`.

Limits: at most {MAX_SOURCE_BYTES} bytes; no #include, no #pragma, no asm, no
__attribute__; only <math.h> functions may be called (no I/O, no malloc). The
header below is already included for you.

## How the coordinator judges you
1. Your rule's weighted error on HIDDEN validation images must be below the bar ({bar:.3f}).
2. Among rules under the bar, the coordinator keeps the one with the most boosting
   progress per nanosecond: progress = -0.5 * ln(1 - 4 g^2), g = 0.5 - error.
   Time is the extra ns per weak_score call over an empty rule; anything under
   1 ns counts as 1 ns. Building the vj_image (integral images, mean, std) is
   free, so one Haar feature costs well under 1 ns, while a loop over all 361
   pixels costs tens to hundreds of ns. Among cheap rules, lower error wins;
   an expensive rule must buy a much lower error to win.
3. You have {deadline_s:.0f} seconds from the start of this session. Then you are
   stopped. Only a rule saved with submit_rule before then counts, so submit early
   and resubmit if you find something better. The last submission counts.

## Tools
- try_rule(source): build the rule; report weighted TRAINING error and time.
- submit_rule(source): the same, and save it as your submission.

## The contract: vj_learner.h
```c
{header}
```

## The weighted training images now
{summary}
## Rules already in the ensemble
{history}

## Your last result
{feedback or "No feedback yet."}
"""
