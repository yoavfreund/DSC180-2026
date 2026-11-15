"""A finished run's ensemble: reload it, score images, and slice it into a cascade.

The run trains ONE chain of weak rules, F_T(x) = sum over t <= T of alpha_t * h_t(x).
A cascade is the same chain with rejection points: after the first c rules, an
image whose running score F_c(x) is below theta is rejected as a non-face and
no more rules are computed for it.

theta at each cut comes from validation FACES: it is set so that only a small
share of the faces still alive at that cut fall below it. Face loss adds up over
cuts, so the total budget is split evenly: per_cut = face_loss / number of cuts.

One sortable score per image (for the shared FP-point evaluator): an image
rejected at an earlier cut always ranks below one rejected later, and survivors
rank above all rejected images. Within the same exit point, the running score
decides.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

import harness.build as hb
from harness.build import build_learner, run_scores, write_images
from harness.threshold import Stump


def load_manifest(run_dir):
    path = Path(run_dir) / "manifest.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"{path} has no accepted rules")
    return entries


def learner_outputs(run_dir, images):
    """h_t(x) in {-1, +1} for every accepted rule t and image x: shape (T, n)."""
    run_dir = Path(run_dir)
    if (run_dir / "harness" / "runner.c").exists():
        hb.use_harness(run_dir / "harness")       # rebuild exactly as the run built
    entries = load_manifest(run_dir)
    images_path = run_dir / "build" / f"images-{len(images)}-{hashlib.sha256(images.tobytes()).hexdigest()[:12]}.bin"
    images_path.parent.mkdir(parents=True, exist_ok=True)
    write_images(images, images_path)
    H = np.empty((len(entries), len(images)), dtype=np.int8)
    for t, e in enumerate(entries):
        source = (run_dir / e["source"]).read_text(encoding="utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != e["source_sha256"]:
            raise ValueError(f"{e['source']} does not match its hash in the manifest")
        build = build_learner(source, run_dir / "build" / f"round_{e['round']:03d}", run_dir / "cache")
        scores = run_scores(build.binary, images_path, len(images))
        H[t] = Stump(e["theta"], e["polarity"], e["val_err"]).predict(scores)
    return entries, H


def running_scores(H, alphas):
    """F_cum[c-1] = score after the first c rules. Shape (T, n)."""
    return np.cumsum(np.asarray(alphas)[:, None] * H, axis=0)


def default_cuts(T):
    cuts, c = [], 1
    while c < T:
        cuts.append(c)
        c *= 2
    return cuts


def calibrate(F_cum, y, cuts, face_loss):
    """theta for each cut from the faces in (F_cum, y). theta None means the cut rejects nothing."""
    T = F_cum.shape[0]
    if any(not 1 <= c < T for c in cuts) or sorted(set(cuts)) != list(cuts):
        raise ValueError(f"cuts must be increasing and between 1 and {T - 1}")
    per_cut = face_loss / max(len(cuts), 1)
    alive = np.asarray(y) > 0
    stages = []
    for c in cuts:
        vals = np.sort(F_cum[c - 1, alive])
        k = int(np.floor(per_cut * len(vals) + 1e-9))
        theta = float(vals[k]) if 0 < k < len(vals) else None
        stages.append({"after": c, "theta": theta})
        if theta is not None:
            alive &= F_cum[c - 1] >= theta
    return stages


def apply_cascade(F_cum, stages, alphas):
    """Returns (score, learners_evaluated) for each image."""
    T, n = F_cum.shape
    offset = 2.0 * float(np.sum(np.abs(alphas))) + 2.0     # |F| < offset / 2 always
    score = np.empty(n)
    used = np.full(n, T)
    alive = np.ones(n, dtype=bool)
    for j, st in enumerate(stages):
        if st["theta"] is None:
            continue
        out = alive & (F_cum[st["after"] - 1] < st["theta"])
        score[out] = j * offset + F_cum[st["after"] - 1, out]
        used[out] = st["after"]
        alive &= ~out
    score[alive] = len(stages) * offset + F_cum[-1, alive]
    return score, used


def mean_ns_per_image(used, ns):
    """Average cost when rule t only runs on images that reached it."""
    ns = np.asarray(ns, dtype=np.float64)
    reached = np.array([(used >= t + 1).mean() for t in range(len(ns))])
    return float(np.dot(reached, ns))
