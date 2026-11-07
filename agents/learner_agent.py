# agents/learner_agent.py
import ctypes as C
import hashlib
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------- ctypes mirrors of weak.h ----------

FAMILIES = {"stump": 0, "tree": 1, "logreg": 2}

class _Spec(C.Structure):
    _fields_ = [("family", C.c_int), ("max_depth", C.c_int), ("min_leaf", C.c_int),
                ("iters", C.c_int), ("lr", C.c_double), ("l2", C.c_double)]

class _EvalResult(C.Structure):
    _fields_ = [("err", C.c_double), ("fpr", C.c_double),
                ("fnr", C.c_double), ("w_pos_val", C.c_double)]


# ---------- messages (match the coordinator's usage) ----------

@dataclass(frozen=True)
class TrainRequest:
    request_id: str
    learner_id: str
    spec: dict                  # already validated by the coordinator
    train_idx: np.ndarray       # learner never receives validation indices
    w: Optional[np.ndarray]     # None means uniform weights
    round_dir: str              # where the model artifact is written

@dataclass(frozen=True)
class ModelHandle:
    model_id: str
    spec: dict
    artifact_path: str
    sha256: str                 # hash of the serialized model file

@dataclass(frozen=True)
class TrainStatus:
    request_id: str
    learner_id: str
    ok: bool
    handle: Optional[ModelHandle]
    train_err: Optional[float]  # retained for compatibility; never reported
    error: Optional[str]
    elapsed_s: float

@dataclass(frozen=True)
class PredictResult:
    labels: np.ndarray          # {-1, +1}
    scores: np.ndarray          # P(y=+1|x)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _ptr(a, t):
    return a.ctypes.data_as(C.POINTER(t))


def training_rows(X, variants, y, w, train_idx):
    groups = [X[train_idx]]
    if variants is not None:
        groups.extend(v[train_idx] for v in variants)
    count = len(groups)
    return (np.concatenate(groups), np.ascontiguousarray(np.tile(y[train_idx], count), dtype=np.int32),
            np.ascontiguousarray(np.tile(w[train_idx] / count, count)))


class LearnerAgent:
    """Trains and predicts one concrete model at a time. Stateless between calls."""

    def __init__(self, lib_path, X, y, variants=None):
        self._lib = C.CDLL(lib_path)   # ctypes releases the GIL during calls
        self._set_signatures()
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.y = np.ascontiguousarray(y, dtype=np.int32)
        self.n, self.f = self.X.shape
        self.variants = variants

    def _set_signatures(self):
        lib = self._lib
        lib.train_predict.restype = C.c_int
        lib.train_predict.argtypes = [
            C.POINTER(_Spec), C.c_char_p, C.c_int, C.c_int,
            C.POINTER(C.c_double), C.POINTER(C.c_int), C.POINTER(C.c_double),
            C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_double)]
        lib.predict_model.restype = C.c_int
        lib.predict_model.argtypes = [
            C.c_char_p, C.c_int, C.c_int, C.POINTER(C.c_double),
            C.POINTER(C.c_int), C.POINTER(C.c_double)]
        lib.evaluate.restype = _EvalResult
        lib.evaluate.argtypes = [
            C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_double),
            C.c_int, C.POINTER(C.c_int)]

    # ---------- Accept a train request ----------

    def train(self, req: TrainRequest) -> TrainStatus:
        t0 = time.perf_counter()
        try:
            c_spec = self._to_c_spec(req.spec)
            train_idx = np.ascontiguousarray(req.train_idx, dtype=np.int32)
            if train_idx.size == 0 or train_idx.min() < 0 or train_idx.max() >= self.n:
                raise ValueError("train_idx empty or out of range")
            w = (np.full(self.n, 1.0 / self.n) if req.w is None
                 else np.ascontiguousarray(req.w, dtype=np.float64))
            if w.shape != (self.n,) or not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError("invalid sample weights")

            os.makedirs(req.round_dir, exist_ok=True)
            model_path = os.path.join(req.round_dir, f"{req.request_id}.bin")
            # Construct only training rows, preserving one original's total mass.
            rows, labels, weights = training_rows(self.X, self.variants, self.y, w, train_idx)
            X = self._select_features(rows, req.spec)
            indices = np.arange(len(labels), dtype=np.int32)
            pred = np.empty(len(labels), dtype=np.int32)
            prob = np.empty(len(labels), dtype=np.float64)
            rc = self._lib.train_predict(
                C.byref(c_spec), model_path.encode(), len(labels), X.shape[1],
                _ptr(X, C.c_double), _ptr(labels, C.c_int), _ptr(weights, C.c_double),
                len(indices), _ptr(indices, C.c_int),
                _ptr(pred, C.c_int), _ptr(prob, C.c_double))
            if rc != 0:
                raise RuntimeError(f"train_predict returned {rc}")

            handle = ModelHandle(model_id=req.request_id, spec=req.spec,
                                 artifact_path=model_path,
                                 sha256=_sha256_file(model_path))
            return TrainStatus(req.request_id, req.learner_id, True, handle,
                               None, None, time.perf_counter() - t0)
        except Exception as e:
            # Failures are reported, not raised, so one bad candidate doesn't stop the round.
            return TrainStatus(req.request_id, req.learner_id, False, None,
                               None, str(e), time.perf_counter() - t0)

    # ---------- Accept a predict request ----------

    def predict(self, handle: ModelHandle, X_new) -> PredictResult:
        X_new = self._select_features(X_new, handle.spec)
        n, f = X_new.shape
        labels = np.empty(n, dtype=np.int32)
        scores = np.empty(n, dtype=np.float64)
        rc = self._lib.predict_model(
            handle.artifact_path.encode(), n, f, _ptr(X_new, C.c_double),
            _ptr(labels, C.c_int), _ptr(scores, C.c_double))
        if rc != 0:
            raise RuntimeError(f"predict_model returned {rc}")
        return PredictResult(labels, scores)

    # ---------- helpers ----------

    @staticmethod
    def _select_features(X, spec):
        # The image loader provides 361 pixels, 289 interior LBP codes,
        # and 64 integral-image rectangle means, in that order.
        slices = {"pixels": slice(0, 361), "lbp": slice(361, 650),
                  "rectangles": slice(650, 714)}
        feature = spec.get("features", "pixels")
        if X.shape[1] != 714 or feature not in slices:
            raise ValueError("expected image features from common.images")
        return np.ascontiguousarray(X[:, slices[feature]], dtype=np.float64)

    def _evaluate(self, pred, w, idx):
        return self._lib.evaluate(self.n, _ptr(pred, C.c_int), _ptr(self.y, C.c_int),
                                  _ptr(w, C.c_double), idx.size, _ptr(idx, C.c_int))

    @staticmethod
    def _to_c_spec(d):
        fam = d.get("family")
        if fam not in FAMILIES:
            raise ValueError(f"unknown family {fam!r}")
        return _Spec(FAMILIES[fam],
                     int(d.get("max_depth", 0)), int(d.get("min_leaf", 0)),
                     int(d.get("iters", 0)),
                     float(d.get("lr", 0.0)), float(d.get("l2", 0.0)))
