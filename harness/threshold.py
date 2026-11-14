"""Fit the threshold and polarity of a weak rule on weighted training images.

The rule predicts h(x) = polarity if score(x) > theta, else -polarity.
Thresholds fall between distinct sorted scores, so tied scores are always on
the same side. The fit is exact: it checks every split in O(n log n).
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stump:
    theta: float
    polarity: int   # +1: a high score means face. -1: a low score means face.
    error: float    # weighted error on the data it was fitted on, in [0, 1]

    def predict(self, scores):
        return np.where(np.asarray(scores) > self.theta, self.polarity, -self.polarity).astype(np.int8)


def fit_stump(scores, y, w):
    """Lowest weighted-error threshold and polarity. y in {-1, +1}; w >= 0, any scale."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y)
    w = np.asarray(w, dtype=np.float64)
    n = len(scores)
    if n == 0 or y.shape != (n,) or w.shape != (n,):
        raise ValueError("scores, y and w must be non-empty and the same length")
    total = w.sum()
    if not total > 0:
        raise ValueError("weights must have a positive sum")

    order = np.argsort(scores, kind="stable")
    s, ys, ws = scores[order], y[order], w[order]
    wpos = np.where(ys > 0, ws, 0.0)
    cpos = np.concatenate([[0.0], np.cumsum(wpos)])        # face weight among the k lowest scores
    cneg = np.concatenate([[0.0], np.cumsum(ws - wpos)])   # non-face weight among the k lowest scores
    P, N = cpos[-1], cneg[-1]

    # Split k: the k lowest scores are predicted "low", the rest "high".
    valid = np.ones(n + 1, dtype=bool)
    valid[1:n] = s[:-1] < s[1:]
    err_high_is_face = cpos + (N - cneg)
    err_low_is_face = cneg + (P - cpos)
    e1 = np.where(valid, err_high_is_face, np.inf)
    e2 = np.where(valid, err_low_is_face, np.inf)
    k1, k2 = int(np.argmin(e1)), int(np.argmin(e2))
    if e1[k1] <= e2[k2]:
        k, polarity, err = k1, 1, e1[k1]
    else:
        k, polarity, err = k2, -1, e2[k2]

    if k == 0:
        theta = np.nextafter(s[0], -np.inf)   # every score is above it
    elif k == n:
        theta = s[-1]                         # no score is above it
    else:
        theta = s[k - 1] + (s[k] - s[k - 1]) / 2.0
        if not s[k - 1] <= theta < s[k]:   # adjacent doubles: the midpoint rounds onto s[k]
            theta = s[k - 1]
    return Stump(float(theta), polarity, float(err / total))
