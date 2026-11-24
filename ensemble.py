"""A finished run's ensemble: reload it, score images, and slice it into a cascade.

The run trains ONE chain of weak rules, F_T(x) = sum over t <= T of alpha_t * h_t(x).
A cascade is the same chain with rejection points: after the first c rules, an
image whose running score F_c(x) is below theta is rejected as a non-face and
no more rules are computed for it.

Trajectory calibration chooses checkpoints and thresholds from validation
scores under a single face-loss budget for the entire cascade.

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


def learner_outputs(run_dir, images, rules=None, build_dir=None):
    """h_t(x) in {-1, +1} for every accepted rule t and image x: shape (T, n)."""
    run_dir = Path(run_dir)
    snapshot = run_dir / "harness"
    hb.use_harness(snapshot if (snapshot / "runner.c").exists() else Path(__file__).parent / "harness")
    entries = chain_prefix(run_dir, rules)
    build_dir = Path(build_dir) if build_dir is not None else run_dir / "build"
    images_path = build_dir / f"images-{len(images)}-{hashlib.sha256(images.tobytes()).hexdigest()[:12]}.bin"
    images_path.parent.mkdir(parents=True, exist_ok=True)
    write_images(images, images_path)
    H = np.empty((len(entries), len(images)), dtype=np.int8)
    for t, e in enumerate(entries):
        source = read_rule(run_dir, e)
        build = build_learner(source, build_dir / f"round_{e['round']:03d}", build_dir / "cache")
        scores = run_scores(build.binary, images_path, len(images))
        H[t] = Stump(e["theta"], e["polarity"], e["val_err"]).predict(scores)
    return entries, H


def chain_prefix(run_dir, rules=None):
    entries = load_manifest(run_dir)
    if rules is None:
        return entries
    if not 1 <= rules <= len(entries):
        raise ValueError(f"requested {rules} rules, available range is 1..{len(entries)}")
    return entries[:rules]


def read_rule(run_dir, entry):
    source = (Path(run_dir) / entry["source"]).read_text(encoding="utf-8")
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != entry["source_sha256"]:
        raise ValueError(f"{entry['source']} does not match its hash in the manifest")
    return source


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


def calibrate_trajectories(F_cum, y, face_loss, cuts=None):
    """Choose soft-cascade stages from complete validation score trajectories.

    A stage rejects only below zero and preserves any validation trajectory
    that reaches zero or higher at or after that checkpoint. Among the
    surviving patches, rank checkpoints by non-face rule evaluations saved
    minus one comparison per patch reaching the stage. Spend the face budget
    across all chosen stages, keeping tied scores together.
    """
    T, n = F_cum.shape
    cuts = list(range(1, T)) if cuts is None else list(cuts)
    if any(not 1 <= c < T for c in cuts) or sorted(set(cuts)) != cuts:
        raise ValueError(f"cuts must be increasing and between 1 and {T - 1}")
    if not 0 <= face_loss <= 1:
        raise ValueError("face loss must be between 0 and 1")
    faces = np.asarray(y) > 0
    remaining = int(np.floor(face_loss * faces.sum() + 1e-9))
    recovers = np.maximum.accumulate(F_cum[::-1], axis=0)[::-1] >= 0
    alive = np.ones(n, dtype=bool)
    stages = []
    while cuts:
        best = None
        best_gain = 0
        checks = int(alive.sum())
        for cut in cuts:
            scores = F_cum[cut - 1]
            recovered_scores = scores[alive & recovers[cut - 1]]
            face_scores = np.sort(scores[alive & faces])
            theta = min(0.0, float(recovered_scores.min(initial=0.0)))
            if remaining < len(face_scores):
                theta = min(theta, float(face_scores[remaining]))
            exits = alive & (scores < theta)
            gain = int(np.count_nonzero(exits & ~faces)) * (T - cut) - checks
            if gain > best_gain:
                best, best_gain = (cut, theta, exits), gain
        if best is None:
            break
        cut, theta, exits = best
        stages.append({"after": cut, "theta": theta})
        alive &= ~exits
        remaining -= int(np.count_nonzero(exits & faces))
        cuts = [c for c in cuts if c > cut]
    return stages


def apply_cascade(F_cum, stages, alphas, *, rank_exits=True):
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
        score[out] = j * offset + F_cum[st["after"] - 1, out] if rank_exits else -offset
        used[out] = st["after"]
        alive &= ~out
    score[alive] = len(stages) * offset + F_cum[-1, alive] if rank_exits else F_cum[-1, alive]
    return score, used


def mean_ns_per_image(used, ns):
    """Average cost when rule t only runs on images that reached it."""
    ns = np.asarray(ns, dtype=np.float64)
    reached = np.array([(used >= t + 1).mean() for t in range(len(ns))])
    return float(np.dot(reached, ns))
