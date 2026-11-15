"""The text each agent receives at the start of an attempt.

The agent has no file tools, so everything goes into one prompt: the task, the
C contract (vj_learner.h), a preview of 8 weight-sampled TRAINING images, the rules
already in the ensemble, and this agent's own feedback. Every agent gets the
same facts; only the feedback section differs. Strategy belongs in each
agent's own .md (its system prompt), not here.
"""
import re

import numpy as np

import harness.build as hb
from harness.build import MAX_SOURCE_BYTES

RAMP = " .:-=+*#%@"


def _ramp_rows(image):
    levels = np.clip((image.astype(np.float64) / 256 * len(RAMP)).astype(int), 0, len(RAMP) - 1)
    return ["".join(RAMP[v] for v in row) for row in levels]


def preview(images, y, w, rng, correct=None, variant=None, per_class=4):
    """A small, fresh view of the TRAINING images: per_class faces and per_class non-faces,
    drawn without replacement in proportion to their current weight. Only these images are
    shown; nothing is averaged over the whole training set.

    correct: per-row bool (does the current ensemble get it right?) or None before any rule.
    Returns (text, row indices shown).
    """
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    faces = y > 0
    picks, blocks = [], []
    for label, mask in (("faces", faces), ("non-faces", ~faces)):
        idx = np.flatnonzero(mask & (w > 0))
        k = min(per_class, len(idx))
        if k == 0:
            continue
        chosen = rng.choice(idx, size=k, replace=False, p=w[idx] / w[idx].sum())
        picks.extend(int(i) for i in chosen)
        rows = [_ramp_rows(images[i]) for i in chosen]
        notes = []
        for n, i in enumerate(chosen, 1):
            verdict = ("no rules yet" if correct is None else
                       "the ensemble gets it RIGHT" if correct[i] else "the ensemble gets it WRONG")
            extra = f", copy made by {variant[i]}" if variant is not None and variant[i] else ""
            notes.append(f"  {label[:-1]} {n}: {verdict}, weight {w[i] * len(w):.1f} x average{extra}")
        blocks.append(f"{label.capitalize()} 1-{k}, left to right:")
        blocks += ["   ".join(r[row] for r in rows) for row in range(19)]
        blocks += notes + [""]
    head = (f"{int(faces.sum())} training faces and {int((~faces).sum())} training non-faces"
            f"{' (including transformed copies)' if variant is not None and any(variant) else ''}. "
            f"Below: {len(picks)} of them, drawn at random in proportion to their current weight, "
            f"so they show what the ensemble now struggles with. A new draw each attempt. "
            f"Brightness ramp '{RAMP}' (dark to bright), one character per pixel, x = column, y = row.")
    return head + "\n\n" + "\n".join(blocks), picks


def describe(source):
    """The first C comment in the source, as the rule's one-line description."""
    m = re.search(r"/\*(.*?)\*/|//([^\n]*)", source, re.S)
    text = (m.group(1) or m.group(2) or "") if m else ""
    text = " ".join(text.split())
    return (text[:160] + "...") if len(text) > 160 else (text or "(no description comment)")


def one_line(source, limit=400):
    """C source squeezed onto one line, for the prompt."""
    text = " ".join(source.split())
    return text if len(text) <= limit else text[:limit] + " ..."


def history_text(ensemble, codes=(), code_limit=20):
    """Accepted rules: a table, then the source of the newest code_limit rules."""
    if not ensemble:
        return "No rules have been accepted yet. This is the first round."
    lines = ["round | agent | validation error | alpha | ns/image | description",
             "----- | ----- | ---------------- | ----- | -------- | -----------"]
    for m in ensemble:
        lines.append(f"{m['round'] + 1} | {m['agent']} | {m['val_err']:.4f} | {m['alpha']:.3f} | "
                     f"{m['ns']:.2f} | {m['description']}")
    if codes:
        lines += ["", "Source of the accepted rules (an accepted rule has error near 0.5 now, "
                      "so do not submit it or a near copy again):"]
        start = max(len(codes) - code_limit, 0)
        for m, code in list(zip(ensemble, codes))[start:]:
            lines.append(f"- round {m['round'] + 1}: `{one_line(code)}`")
    return "\n".join(lines)


def memory_text(entries, limit=6):
    """This agent's own recent submissions and what happened to each."""
    if not entries:
        return "You have not submitted a rule yet."
    lines = [f"Your last {min(limit, len(entries))} submissions, oldest first. Each error was measured "
             "with the weights of its own round, so it is not current."]
    for e in entries[-limit:]:
        head = f"- round {e['round'] + 1}, attempt {e['attempt'] + 1}: {e['outcome']}"
        if e.get("val_err") is not None:
            head += f" (validation error {e['val_err']:.3f}, {e['ns']:.2f} ns)"
        lines.append(head)
        if e.get("source"):
            lines.append(f"  `{one_line(e['source'], 300)}`")
    return "\n".join(lines)


def brief(*, round_t, rounds, attempt, bar, deadline_s, preview, history, feedback, memory="", margin=0.0):
    header = hb.HEADER.read_text(encoding="utf-8")
    noise = (f"\n   The weights now sit on few images, so the measured error is noisy. The coordinator\n"
             f"   adds a noise margin (about {margin:.3f} at error 0.35) before it compares with the bar."
             if margin > 0 else "")
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
1. Your rule's weighted error on HIDDEN validation images must be below the bar ({bar:.3f}).{noise}
2. Among rules under the bar, the coordinator keeps the one with the most boosting
   progress per nanosecond: progress = -0.5 * ln(1 - 4 g^2), g = 0.5 - error.
   Time is the extra ns per weak_score call over an empty rule; anything under
   1 ns counts as 1 ns. Building the vj_image (integral images, mean, std) is
   free. One Haar feature costs under 1 ns (counted as 1 ns). A rule with two
   Haar features measured about 3 ns, so it needs about 3 times the progress
   of a one-feature rule to win. A loop over all 361 pixels costs tens of ns.
3. You have {deadline_s:.0f} seconds from the start of this session. Then you are
   stopped. Only a rule saved with submit_rule before then counts, so submit early
   and resubmit if you find something better. The last submission counts.

## Tools
- scan(source, ranges): the source is a TEMPLATE that uses P0, P1, P2, P3 for the integers
  you are unsure of (positions, sizes), and ranges gives [lo, hi] or [lo, hi, step] for each,
  at most 1000 combinations. The scan tries every combination on the weighted TRAINING
  images and returns the 5 best concrete rules, ready to submit. Saves nothing.
- try_rules(sources): build 1 to 4 concrete rules in ONE call; report each one's weighted
  TRAINING error and time. Saves nothing.
- submit_rule(source): build one concrete rule, report it, and save it as your submission.
  P0..P3 exist only in scan templates; a submitted rule uses numbers.
You may call several tools in the same turn.

## The contract: vj_learner.h
```c
{header}
```

## A preview of the training images
{preview}
## Rules already in the ensemble
{history}

## Your last result
{feedback or "No feedback yet."}

## Your memory
{memory or "You have not submitted a rule yet."}
"""
