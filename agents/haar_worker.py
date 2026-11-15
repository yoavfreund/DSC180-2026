"""Brute-force Haar search: no LLM. Used three ways:
  - as the fallback when no agent passes the top of the ladder,
  - as a baseline agent to compare LLM agents against,
  - to test the whole pipeline without spending LLM usage.

It evaluates Viola-Jones Haar features on the weighted TRAINING images, picks
the one with the lowest weighted error, and writes it as C source using the
same helpers an LLM agent would use. The coordinator then re-scores it in C
like any other submission.

Usage: python -m agents.haar_worker --private DIR --out DIR
"""
import argparse
import time
from pathlib import Path

import numpy as np

from agents.common import TrainData, write_result, write_submission

# Rectangle widths and heights searched (a coarse grid keeps the search to a few seconds).
SIZES = (1, 2, 3, 4, 6, 8)
BATCH = 1024


def integral(images):
    ii = np.zeros((len(images), 20, 20), dtype=np.float64)
    ii[:, 1:, 1:] = images.astype(np.float64).cumsum(1).cumsum(2)
    return ii


def window_std(images):
    v = images.reshape(len(images), -1).astype(np.float64).var(axis=1)
    return np.where(v > 1.0, np.sqrt(np.maximum(v, 1.0)), 1.0)


def rect_sums(ii, w, h):
    """(n, 20-h, 20-w): sum of the w x h rectangle whose top-left corner is (x, y), at [:, y, x]."""
    return ii[:, h:, w:] - ii[:, :-h, w:] - ii[:, h:, :-w] + ii[:, :-h, :-w]


def features(ii, std):
    """Yield (kind, w, h, values (n, k), positions [(x, y)])."""
    for w in SIZES:
        for h in SIZES:
            R = rect_sums(ii, w, h)
            ny, nx = R.shape[1], R.shape[2]   # valid top-left corners: y < ny, x < nx
            shapes = {
                "haar2_lr": (lambda X, Y: R[:, :Y, :X] - R[:, :Y, w:w + X], nx - w, ny),
                "haar2_tb": (lambda X, Y: R[:, :Y, :X] - R[:, h:h + Y, :X], nx, ny - h),
                "haar3_lr": (lambda X, Y: 2 * R[:, :Y, w:w + X] - R[:, :Y, :X] - R[:, :Y, 2 * w:2 * w + X], nx - 2 * w, ny),
                "haar3_tb": (lambda X, Y: 2 * R[:, h:h + Y, :X] - R[:, :Y, :X] - R[:, 2 * h:2 * h + Y, :X], nx, ny - 2 * h),
                "haar4": (lambda X, Y: R[:, :Y, :X] + R[:, h:h + Y, w:w + X] - R[:, :Y, w:w + X] - R[:, h:h + Y, :X], nx - w, ny - h),
            }
            for kind, (make, X, Y) in shapes.items():
                if X <= 0 or Y <= 0:
                    continue
                vals = (make(X, Y) / std[:, None, None]).reshape(len(std), -1)
                pos = [(x, y) for y in range(Y) for x in range(X)]
                yield kind, w, h, vals, pos


def best_errors(F, y, w):
    """Lowest weighted stump error for each column of F (n, k)."""
    order = np.argsort(F, axis=0, kind="stable")
    Fs = np.take_along_axis(F, order, axis=0)
    ws = w[order]
    wp = np.where(y[order] > 0, ws, 0.0)
    cp, cn = np.cumsum(wp, axis=0), np.cumsum(ws - wp, axis=0)
    P, N = cp[-1], cn[-1]
    e = np.minimum(cp[:-1] + (N - cn[:-1]), cn[:-1] + (P - cp[:-1]))
    e = np.where(Fs[:-1] < Fs[1:], e, np.inf)
    return np.minimum(e.min(axis=0), np.minimum(P, N))


def search(images, y, w):
    w = w / w.sum()
    ii, std = integral(images), window_std(images)
    best = (np.inf, None)
    pending_vals, pending_meta = [], []

    def flush():
        nonlocal best
        if not pending_vals:
            return
        errs = best_errors(np.concatenate(pending_vals, axis=1), y, w)
        i = int(np.argmin(errs))
        if errs[i] < best[0]:
            best = (float(errs[i]), pending_meta[i])
        pending_vals.clear()
        pending_meta.clear()

    for kind, fw, fh, vals, pos in features(ii, std):
        pending_vals.append(vals.astype(np.float32))
        pending_meta.extend((kind, x, yy, fw, fh) for x, yy in pos)
        if len(pending_meta) >= BATCH:
            flush()
    flush()
    return best


def to_source(meta, err):
    kind, x, y, w, h = meta
    return (f"/* rule: Haar {kind} at x={x}, y={y}, each rectangle {w}x{h} "
            f"(brute-force search, training error {err:.3f}) */\n"
            f"double weak_score(const vj_image *im) {{ return vj_{kind}(im, {x}, {y}, {w}, {h}); }}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.monotonic()
    data = TrainData.load(args.private, with_images=True, search=True)   # weighted sample if large
    err, meta = search(data.images, data.y, data.w)
    if meta is None:
        write_result(args.out, status="error", error="no feature found")
        return
    write_submission(Path(args.out), to_source(meta, err))
    write_result(args.out, status="finished", train_error=err, feature=list(meta),
                 seconds=round(time.monotonic() - t0, 2))


if __name__ == "__main__":
    main()
