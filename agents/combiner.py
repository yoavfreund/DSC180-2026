# agents/combiner.py
import numpy as np
import hashlib
import json
from pathlib import Path

from agents.learner_agent import ModelHandle

MODES = ("majority", "soft", "adaboost")


def load_ensemble(run_dir):
    """Load a saved ensemble, verifying model paths and hashes."""
    root = Path(run_dir).resolve()
    ensemble = []
    with (root / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            path = (root / item["model_ref"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("model path is outside run directory")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item["model_sha256"]:
                raise ValueError("saved model checksum mismatch")
            handle = ModelHandle(item["learner_id"], item["spec"], str(path), digest)
            ensemble.append({"handle": handle, "spec": item["spec"],
                             "learner_id": item["learner_id"], "alpha": float(item["alpha"])})
    if not ensemble:
        raise ValueError("no accepted learners")
    return ensemble


def _stack(ensemble, learners, X_new):
    if not ensemble:
        raise ValueError("ensemble is empty")
    labels, probs = [], []
    for m in ensemble:
        r = learners.predict(m["handle"], X_new)   # deterministic C inference
        labels.append(r.labels)
        probs.append(r.scores)
    alphas = np.array([m["alpha"] for m in ensemble], dtype=np.float64)
    return np.array(labels), np.array(probs), alphas


def combine(ensemble, learners, X_new, mode):
    """Returns labels in {-1, +1}. Ties go to +1."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    P, Q, A = _stack(ensemble, learners, X_new)

    if mode == "majority":                 # unweighted vote over hard labels
        s = P.sum(axis=0)
    elif mode == "soft":                   # mean P(y=+1|x) vs 0.5
        s = Q.mean(axis=0) - 0.5
    else:                                  # adaboost: alpha-weighted hard vote
        s = (A[:, None] * P).sum(axis=0)

    return np.where(s >= 0, 1, -1).astype(np.int32)
