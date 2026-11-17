"""Box 2: Harness. Checks a candidate rule, searches its knobs, and measures its runtime.

A rule is Python code that defines
  rule(X, *p)  X is a batch of images, shape (n, 19, 19), each adjusted to brightness 0 and
               contrast 1 (so a rule can't cheat on lighting). Returns one number per image.
  GRID         optional list of knob settings p to try, e.g. every position and size of a box.
The harness tries every setting on the weighted training set, picks the threshold and
direction with the lowest weighted error for each, and keeps the best setting.
"""
import time

import numpy as np

MAX_GRID = 4000   # settings tried per candidate; a bigger GRID is sampled down to this


def normalize(X):
    """Each image to mean 0 and std 1 (Viola-Jones variance normalization)."""
    X = X.astype(np.float32).reshape(len(X), -1)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)
    return X.reshape(-1, 19, 19)


def load(code):
    """The rule function and its knob settings. Only numpy is available."""
    env = {"np": np}
    exec(code, env)  # local research code: the rule comes from our own Claude call
    if "rule" not in env:
        raise ValueError("the code must define rule(X, ...)")
    grid = [tuple(p) if isinstance(p, (tuple, list)) else (p,) for p in env.get("GRID", [()])]
    if not grid:
        raise ValueError("GRID is empty")
    if len(grid) > MAX_GRID:
        keep = np.sort(np.random.default_rng(0).choice(len(grid), MAX_GRID, replace=False))
        grid = [grid[i] for i in keep]
    return env["rule"], grid


def scores(rule, X, p=()):
    """The rule's number for every (normalized) image, plus microseconds per image."""
    t = time.perf_counter()
    with np.errstate(all="ignore"):
        s = np.asarray(rule(X, *p), dtype=float).reshape(-1)
    us = (time.perf_counter() - t) / len(X) * 1e6
    if s.shape != (len(X),):
        raise ValueError(f"rule returned shape {s.shape}, expected ({len(X)},)")
    if not np.all(np.isfinite(s)):
        raise ValueError("the rule returned nan or inf on some image")
    return s, us


def best_threshold(s, y, w, graded=False):
    """Threshold and direction with the lowest weighted error, plus that cut's Z.

    Prediction: face (+1) if pol * (s - thr) > 0, else non-face (-1).
    graded: choose the cut with the lowest Z = sum over both sides of sqrt(face weight * non-face
    weight) instead (Real AdaBoost: the cut whose sides are most one-sided, so votes can be sure).
    """
    order = np.argsort(s, kind="stable")
    s, y, w = s[order], y[order], w[order]
    n = len(s)
    face_below = np.concatenate([[0], np.cumsum(np.where(y == 1, w, 0))])   # face weight left of each cut
    non_below = np.concatenate([[0], np.cumsum(np.where(y == -1, w, 0))])
    err_pos = face_below + (non_below[-1] - non_below)   # pol +1: left of the cut called non-face
    err_neg = non_below + (face_below[-1] - face_below)  # pol -1: left of the cut called face
    cuts = np.arange(n + 1)
    ok = (cuts > 0) & (cuts < n) & np.r_[True, s[1:] > s[:-1], True]   # a real cut: both answers used
    if not ok.any():
        return 0.0, 1, 1.0, 1.0                                            # constant rule: useless
    err_pos[~ok], err_neg[~ok] = np.inf, np.inf
    z = (np.sqrt(face_below * non_below)
         + np.sqrt((face_below[-1] - face_below) * (non_below[-1] - non_below))) / w.sum()
    z[~ok] = np.inf
    if graded:
        i = int(np.argmin(z))
        pol, err = (1, err_pos[i]) if err_pos[i] <= err_neg[i] else (-1, err_neg[i])
    else:
        i_pos, i_neg = int(np.argmin(err_pos)), int(np.argmin(err_neg))
        i, pol, err = (i_pos, 1, err_pos[i_pos]) if err_pos[i_pos] <= err_neg[i_neg] else (i_neg, -1, err_neg[i_neg])
    return float((s[i - 1] + s[i]) / 2), pol, float(err / w.sum()), float(z[i])


def predict(rule_info, s):
    """The vote of one checked rule, given its scores: `hi` above the threshold, `lo` below.
    Plain rules vote +1 / -1; graded rules vote how sure they are on each side."""
    return np.where(s > rule_info["thr"], rule_info["hi"], rule_info["lo"])


def grade(r, y, w):
    """Graded votes: on each side of the threshold, 0.5 * ln(face weight / non-face weight)."""
    eps = 1 / len(w)
    side = r["train_scores"] > r["thr"]
    for key, mask in (("hi", side), ("lo", ~side)):
        r[key] = 0.5 * float(np.log((w[mask & (y == 1)].sum() + eps) / (w[mask & (y == -1)].sum() + eps)))


def check(code, X, y, w, graded=False):
    """Try every knob setting on the weighted training set and keep the best one."""
    rule, grid = load(code)
    best, problem = None, None
    for p in grid:
        try:
            s, us = scores(rule, X, p)
        except Exception as e:
            problem = e
            continue
        thr, pol, err, z = best_threshold(s, y, w, graded)
        if best is None or (z < best["z"] if graded else err < best["err"]):
            best = {"code": code, "params": list(p), "thr": thr, "pol": pol, "err": err, "z": z, "us": us,
                    "train_scores": s, "tried": len(grid)}
    if best is None:
        raise ValueError(f"no setting in GRID ran: {type(problem).__name__}: {problem}")
    if best["err"] >= 0.5:
        raise ValueError("the rule gives every image the same answer")
    best["hi"], best["lo"] = best["pol"], -best["pol"]
    if graded:
        grade(best, y, w)
    return best
