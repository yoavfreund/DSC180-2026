# coordinator/specs.py
import json

FAMILY_PARAMS = {
    "stump":  {},
    "tree":   {"max_depth": (1, 6, 2), "min_leaf": (1, 500, 5)},
    "logreg": {"iters": (10, 5000, 200), "lr": (1e-4, 1.0, 0.1), "l2": (0.0, 10.0, 0.0)},
}

DEFAULT_SPECS = [
    {"family": "stump", "features": "pixels"},
    {"family": "stump", "features": "lbp"},
    {"family": "stump", "features": "rectangles"},
    {"family": "tree", "features": "rectangles", "max_depth": 2, "min_leaf": 5},
]

def strip_fences(raw):
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0]
    return s.strip()

def validate_spec(d):
    if not isinstance(d, dict):
        raise ValueError("each spec must be an object")
    fam = d.get("family")
    if fam not in FAMILY_PARAMS:
        raise ValueError(f"unknown family {fam!r}")
    features = d.get("features", "pixels")
    if features not in ("pixels", "lbp", "rectangles"):
        raise ValueError(f"unknown feature set {features!r}")
    out = {"family": fam, "features": features}
    for key, (lo, hi, default) in FAMILY_PARAMS[fam].items():
        v = d.get(key, default)
        out[key] = min(max(type(default)(v), lo), hi)   # clamp, never trust raw values
    return out

def parse_specs(raw, k):
    items = json.loads(strip_fences(raw))
    if not isinstance(items, list) or len(items) != k:
        raise ValueError(f"expected a JSON list of {k} specs")
    return [validate_spec(d) for d in items]
