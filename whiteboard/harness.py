"""Box 2: Harness. Checks a candidate rule and measures its runtime.

A rule is Python code that defines rule(img): img is a 19x19 uint8 array, the
return value is one number. The harness runs it on every training image, picks
the threshold and direction that give the lowest weighted error, and times it.
"""
import time

import numpy as np


def load(code):
    """Turn the rule's source code into a function. Only numpy is available."""
    env = {"np": np}
    exec(code, env)  # local research code: the rule comes from our own Claude call
    if "rule" not in env:
        raise ValueError("the code must define rule(img)")
    return env["rule"]


def scores(rule, X):
    """The rule's number for every image, plus microseconds per image."""
    t = time.perf_counter()
    s = np.array([float(rule(img)) for img in X])
    us = (time.perf_counter() - t) / len(X) * 1e6
    if not np.all(np.isfinite(s)):
        raise ValueError("the rule returned nan or inf on some image")
    return s, us


def best_threshold(s, y, w):
    """Threshold and direction with the lowest weighted error.

    Prediction: face (+1) if pol * (s - thr) > 0, else non-face (-1).
    """
    order = np.argsort(s, kind="stable")
    s, y, w = s[order], y[order], w[order]
    n = len(s)
    face_below = np.concatenate([[0], np.cumsum(np.where(y == 1, w, 0))])   # face weight left of each cut
    non_below = np.concatenate([[0], np.cumsum(np.where(y == -1, w, 0))])
    err_pos = face_below + (non_below[-1] - non_below)   # pol +1: left of the cut called non-face
    err_neg = non_below + (face_below[-1] - face_below)  # pol -1: left of the cut called face
    cuts = np.arange(n + 1)
    ok = (cuts == 0) | (cuts == n) | np.r_[True, s[1:] > s[:-1], True]  # never cut between equal scores
    err_pos[~ok], err_neg[~ok] = np.inf, np.inf
    i_pos, i_neg = int(np.argmin(err_pos)), int(np.argmin(err_neg))
    i, pol, err = (i_pos, 1, err_pos[i_pos]) if err_pos[i_pos] <= err_neg[i_neg] else (i_neg, -1, err_neg[i_neg])
    thr = s[0] - 1 if i == 0 else s[-1] + 1 if i == n else (s[i - 1] + s[i]) / 2
    return float(thr), pol, float(err / w.sum())


def predict(rule_info, s):
    """+1 / -1 votes of one checked rule, given its scores."""
    return np.where(rule_info["pol"] * (s - rule_info["thr"]) > 0, 1, -1)


def check(code, X, y, w):
    """Run the candidate on the weighted training set. Raises if the code is broken."""
    s, us = scores(load(code), X)
    thr, pol, err = best_threshold(s, y, w)
    r = {"code": code, "thr": thr, "pol": pol, "err": err, "us": us, "train_scores": s}
    if len(np.unique(predict(r, s))) == 1:
        raise ValueError("the rule gives every image the same answer")
    return r
