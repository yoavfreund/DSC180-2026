"""Ranking metrics for real-valued face scores (higher = more face-like).

Ties follow the shared evaluator's rule: tied scores are all in or all out.
"""
import numpy as np

FP_RATES = (0.01, 0.05, 0.1)   # the team's reporting points: detection at 1%, 5% and 10% false positives


def auc(scores, y):
    """Probability that a random face outscores a random non-face; ties count one half."""
    scores = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(y) > 0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):                      # average ranks over tied groups
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def detection_at_fp(scores, y, fp_rate):
    """Highest detection rate whose false positives stay <= floor(fp_rate * n_neg). Same rule as eval/eval."""
    scores = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(y) > 0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    allowed = int(np.floor(fp_rate * n_neg + 1e-9))
    order = np.argsort(-scores, kind="stable")
    s, p = scores[order], pos[order]
    tp = fp = 0
    best = 0
    i = 0
    while i < len(s):                      # admit one tied group at a time
        j = i
        while j < len(s) and s[j] == s[i]:
            tp += int(p[j])
            fp += int(not p[j])
            j += 1
        if fp > allowed:
            break
        best = tp
        i = j
    return best / n_pos if n_pos else float("nan")
