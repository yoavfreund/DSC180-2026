"""Shared by every agent worker process.

A worker gets a PRIVATE folder with this round's training images, labels and
weights. Validation images are never written there. The worker's only output
is OUT/submission.c (the last one written before the deadline counts) and
OUT/result.json.
"""
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coordinator.selection import progress_per_ns
from harness.build import baseline_binary, build_learner, marginal_ns, run_scores
from harness.threshold import fit_stump


TIMING_IMAGES = 256   # small enough to stay in cache; see harness.build.marginal_ns


@dataclass
class TrainData:
    images_path: Path
    timing_path: Path     # a fixed mixed sample of faces and non-faces, for timing only
    n: int
    n_timing: int
    y: np.ndarray
    w: np.ndarray
    images: np.ndarray | None = None

    @classmethod
    def load(cls, private_dir, with_images=False):
        d = Path(private_dir)
        y = np.load(d / "train_labels.npy")
        images = None
        if with_images:
            images = np.fromfile(d / "train_images.bin", dtype=np.uint8).reshape(-1, 19, 19)
        n_timing = (d / "timing_images.bin").stat().st_size // 361
        return cls(d / "train_images.bin", d / "timing_images.bin", len(y), n_timing, y,
                   np.load(d / "train_weights.npy"), images)


def write_private(private_dir, images, y, w, seed=0):
    d = Path(private_dir)
    d.mkdir(parents=True, exist_ok=True)
    images = np.ascontiguousarray(images, dtype=np.uint8)
    images.tofile(d / "train_images.bin")
    pick = np.random.default_rng(seed).permutation(len(images))[:TIMING_IMAGES]
    images[np.sort(pick)].tofile(d / "timing_images.bin")
    np.save(d / "train_labels.npy", np.asarray(y, dtype=np.int8))
    np.save(d / "train_weights.npy", np.asarray(w, dtype=np.float64) / np.sum(w))


def write_submission(out_dir, source):
    """Atomic: a kill during the write leaves the previous submission intact."""
    out_dir = Path(out_dir)
    fd, tmp = tempfile.mkstemp(suffix=".c", dir=out_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(source)
    os.replace(tmp, out_dir / "submission.c")
    with open(out_dir / "submissions.log", "a", encoding="utf-8") as f:
        f.write(json.dumps({"bytes": len(source)}) + "\n")


def write_result(out_dir, **fields):
    out_dir = Path(out_dir)
    tmp = out_dir / "result.json.tmp"
    tmp.write_text(json.dumps(fields, default=str, indent=1), encoding="utf-8")
    os.replace(tmp, out_dir / "result.json")


def check_on_train(source, data, build_dir, cache_dir, bar, ns_floor=1.0):
    """Build the rule and measure it on the weighted TRAINING images. Returns agent-readable text and ok flag."""
    build = build_learner(source, build_dir, cache_dir)
    scores = run_scores(build.binary, data.images_path, data.n)
    stump = fit_stump(scores, data.y, data.w)
    ns, _ = marginal_ns([build.binary], data.timing_path, data.n_timing, baseline_binary(cache_dir), rounds=2)[0]
    faces = data.y > 0
    lines = [
        "compiled: yes" + (f" (warnings: {build.warnings})" if build.warnings else ""),
        f"weighted training error: {stump.error:.4f}  (current bar: {bar:.3f})",
        "  This is measured on the same images the threshold was fitted on, so it is optimistic.",
        "  The coordinator judges the rule on hidden validation images.",
        f"polarity: {'high score = face' if stump.polarity > 0 else 'low score = face'}, threshold {stump.theta:.6g}",
        f"score range: faces [{scores[faces].min():.4g}, {scores[faces].max():.4g}], "
        f"non-faces [{scores[~faces].min():.4g}, {scores[~faces].max():.4g}]",
        f"time: {ns:.2f} ns per image over an empty rule (approximate: other agents may be running now; "
        f"times under {ns_floor:g} ns count as {ns_floor:g})",
        f"progress per ns at this error: {progress_per_ns(min(stump.error, 0.4999), ns, ns_floor):.5f}",
    ]
    return "\n".join(lines), stump.error
