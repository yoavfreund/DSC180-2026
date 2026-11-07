# agents/evaluator.py
import ctypes as C
from dataclasses import dataclass

import numpy as np


class _EvalStruct(C.Structure):          # mirrors EvalResult in weak.h
    _fields_ = [("err", C.c_double), ("fpr", C.c_double),
                ("fnr", C.c_double), ("w_pos_val", C.c_double)]


@dataclass(frozen=True)
class Score:
    err: float          # weighted validation error; the only error used for selection and alpha
    fpr: float
    fnr: float
    w_pos_val: float


def _ptr(a, t):
    return a.ctypes.data_as(C.POINTER(t))


class Evaluator:
    """Scores predictions with the C evaluate() function. Stateless between calls."""

    def __init__(self, lib):
        # lib: a ctypes.CDLL handle to libweak.so
        self._lib = lib
        lib.evaluate.restype = _EvalStruct
        lib.evaluate.argtypes = [
            C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_double),
            C.c_int, C.POINTER(C.c_int)]

    def score(self, pred, y, w, idx) -> Score:
        pred = np.ascontiguousarray(pred, dtype=np.int32)
        y = np.ascontiguousarray(y, dtype=np.int32)
        w = np.ascontiguousarray(w, dtype=np.float64)
        idx = np.ascontiguousarray(idx, dtype=np.int32)

        n = len(y)
        if pred.shape != (n,) or w.shape != (n,):
            raise ValueError("pred, y, and w must have the same length")
        if idx.size == 0 or idx.min() < 0 or idx.max() >= n:
            raise ValueError("idx empty or out of range")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("invalid weights")

        r = self._lib.evaluate(n, _ptr(pred, C.c_int), _ptr(y, C.c_int),
                               _ptr(w, C.c_double), idx.size, _ptr(idx, C.c_int))
        return Score(r.err, r.fpr, r.fnr, r.w_pos_val)

    def describe(self, spec, s: Score) -> str:
        # Aggregates only: never labels, weights, or alphas.
        return f"Spec: {spec}\nWeighted validation error: {s.err:.3f}"
