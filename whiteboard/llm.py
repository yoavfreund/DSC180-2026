"""Box 1: LLM. Generates one candidate rule.

It is told: the task, the rules kept so far, the training images that are
hardest right now (highest weight), why its last try failed (small loop), and
the evaluator's validation report (big arrow).
"""
import os
import random
import re
import subprocess

import numpy as np

TASK = """You are writing one weak rule for a face detector (Viola-Jones style boosting).

Write Python with:
  rule(X, *knobs)  X is a batch of images, numpy float array of shape (n, 19, 19), X[:, row, col],
                   row 0 is the top. Every image is already adjusted to mean 0 and std 1, so
                   brightness and contrast carry no information. Return one number per image,
                   shape (n,). Work on the whole batch with numpy (no Python loop over images).
  GRID             list of knob tuples to try, e.g. every position and size of your boxes.
                   The harness tries each one (at most 4000; more get sampled) and keeps the best.

You don't pick a threshold: the harness tries every threshold and both directions.
Goal: lowest weighted error on the training images (the hard examples below show what the
weights focus on now), and cheap, because a detector runs this on millions of windows.
Region means and their differences (eyes darker than cheeks, nose bridge brighter than eyes),
AND-style combinations (np.minimum / np.maximum of two checks) and symmetry are good ideas.
Only numpy is available, as np. Don't import anything.

Example:
def rule(X, r, c, h, w):
    top = X[:, r:r + h, c:c + w].mean(axis=(1, 2))
    below = X[:, r + h:r + 2 * h, c:c + w].mean(axis=(1, 2))
    return below - top
GRID = [(r, c, h, w) for r in range(19) for c in range(19) for h in range(1, 8) for w in range(2, 20)
        if r + 2 * h <= 19 and c + w <= 19]

Reply with exactly one ```python code block and nothing else."""


def picture(img):
    """19x19 image as digits 0 (dark) to 9 (bright)."""
    return "\n".join("".join(str(min(9, int(v) // 26)) for v in row) for row in img)


def hard_examples(X, y, w, per_class=2):
    out = []
    for label, name in ((1, "face"), (-1, "non-face")):
        idx = np.where(y == label)[0]
        for i in idx[np.argsort(-w[idx])[:per_class]]:
            out.append(f"{name}, weight {w[i] * len(w):.1f}x average:\n{picture(X[i])}")
    return "\n\n".join(out)


def prompt(rules, hard, note, report):
    kept = "\n\n".join(f"# rule {i + 1}: weighted error {r['err']:.3f}, best knobs {r['params']}\n{r['code']}"
                       for i, r in enumerate(rules[-8:])) or "None yet. This is the first rule."
    parts = [TASK, "## Rules kept so far (newest 8)\n" + kept,
             "## Hardest training images right now (the next rule should get these right)\n" + hard]
    if report:
        parts.append("## How the whole detector does on validation (from the evaluator)\n" + report)
    if note:
        parts.append("## Your tries so far for this rule (none passed yet; build on the best one)\n" + note)
    return "\n\n".join(parts)


def ask_claude(text, model):
    claude = os.environ.get("CLAUDE_BIN", "claude")
    res = subprocess.run([claude, "-p", text, "--model", model, "--tools", ""],
                         capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(f"claude failed: {res.stderr.strip()[:300]}")
    m = re.search(r"```(?:python)?\n(.*?)```", res.stdout, re.S)
    return (m.group(1) if m else res.stdout).strip()


def fake_rule(rng):
    """A two-box Haar shape (top/bottom or left/right) searched over every position and size,
    so the pipeline runs without Claude. Like brute force, but one shape per try."""
    if rng.random() < 0.5:
        return ("def rule(X, r, c, h, w):\n"
                "    return X[:, r:r+h, c:c+w].mean(axis=(1, 2)) - X[:, r+h:r+2*h, c:c+w].mean(axis=(1, 2))\n"
                "GRID = [(r, c, h, w) for r in range(19) for c in range(19) for h in range(1, 10)"
                " for w in range(1, 20) if r + 2*h <= 19 and c + w <= 19]")
    return ("def rule(X, r, c, h, w):\n"
            "    return X[:, r:r+h, c:c+w].mean(axis=(1, 2)) - X[:, r:r+h, c+w:c+2*w].mean(axis=(1, 2))\n"
            "GRID = [(r, c, h, w) for r in range(19) for c in range(19) for h in range(1, 20)"
            " for w in range(1, 10) if r + h <= 19 and c + 2*w <= 19]")


def write_rule(rules, X, y, w, note, report, model="sonnet", fake=False, rng=random.Random(0)):
    if fake:
        return fake_rule(rng)
    return ask_claude(prompt(rules, hard_examples(X, y, w), note, report), model)
