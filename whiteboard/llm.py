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
Images are 19x19 grayscale, numpy uint8 array img[row, col], row 0 is the top.
Write a Python function rule(img) that returns ONE number. You don't pick a threshold:
the harness tries every threshold and both directions and keeps the best.
Goal: low weighted error on the training images (weights are shown through the hard
examples below), and fast, because a detector runs this on millions of windows.
Rectangle sums, differences of region means, and simple comparisons are good.
Only numpy is available, as np. Don't import anything.
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
    kept = "\n\n".join(f"# rule {i + 1}: weighted error {r['err']:.3f}, {r['us']:.1f} us/image\n{r['code']}"
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
    """Random two-rectangle Haar rule, so the pipeline runs without Claude."""
    r0, c0 = rng.randrange(0, 15), rng.randrange(0, 15)
    h, wd = rng.randrange(2, 20 - r0), rng.randrange(2, 20 - c0)
    if rng.random() < 0.5:
        a, b = f"img[{r0}:{r0 + h // 2}, {c0}:{c0 + wd}]", f"img[{r0 + h // 2}:{r0 + h}, {c0}:{c0 + wd}]"
    else:
        a, b = f"img[{r0}:{r0 + h}, {c0}:{c0 + wd // 2}]", f"img[{r0}:{r0 + h}, {c0 + wd // 2}:{c0 + wd}]"
    return f"def rule(img):\n    return {a}.mean() - {b}.mean()"


def write_rule(rules, X, y, w, note, report, model="sonnet", fake=False, rng=random.Random(0)):
    if fake:
        return fake_rule(rng)
    return ask_claude(prompt(rules, hard_examples(X, y, w), note, report), model)
