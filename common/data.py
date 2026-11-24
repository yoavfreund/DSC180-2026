"""Load 19x19 CBCL PGM images.

The split is the pooled split made by make_split.py (CBCL training and test sets mixed, then
drawn at random by class and source; see split_manifest.tsv):
  <root>/train/{face,non-face}     agent-visible training images
  <root>/validate/{face,non-face}  hidden validation images (selection and alpha)
Optionally, an AUDIT share of the training folder is held out from everything:
agents never see it, it has no weight, and no decision uses it. It only
measures the ensemble honestly while the run goes. The held-out test split lives
outside the repo and is never loaded here.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def read_pgm(path):
    with Path(path).open("rb") as f:
        tokens = []
        while len(tokens) < 4:
            line = f.readline()
            if not line:
                raise ValueError(f"incomplete PGM header: {path}")
            tokens.extend(line.split(b"#", 1)[0].split())
        if tokens != [b"P5", b"19", b"19", b"255"]:
            raise ValueError(f"expected a 19x19 8-bit binary PGM: {path}")
        data = f.read()
    if len(data) != 361:
        raise ValueError(f"invalid PGM data length: {path}")
    return np.frombuffer(data, dtype=np.uint8).reshape(19, 19)


SOURCES = ("cbcl", "cmu")


def source_of(name):
    """'cmu' for an image from the original CBCL test download (files cmu_*), else 'cbcl'."""
    return "cmu" if Path(name).name.startswith("cmu_") else "cbcl"


def load_labeled_folder(folder, limit_per_class=None):
    """Images from folder/face (+1) and folder/non-face (-1), in sorted file order."""
    folder = Path(folder)
    images, labels, ids = [], [], []
    for label, sub in ((1, "face"), (-1, "non-face")):
        paths = sorted((folder / sub).glob("*.pgm"))
        if limit_per_class is not None:
            paths = paths[:limit_per_class]
        for p in paths:
            images.append(read_pgm(p))
            labels.append(label)
            ids.append(f"{sub}/{p.name}")
    if not images:
        raise ValueError(f"no PGM images under {folder}/face or {folder}/non-face")
    return np.stack(images), np.array(labels, dtype=np.int8), ids


AUGMENT_KINDS = ("flip", "rotate", "zoom", "shift")


def _warp(images, angle_deg=0.0, scale=1.0, dx=0.0, dy=0.0):
    """Rotate, zoom and shift a batch of 19x19 images about the center (bilinear, edges clamped)."""
    n, h, w = images.shape
    cy, cx = (h - 1) / 2, (w - 1) / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = np.deg2rad(angle_deg)
    x, y = xx - cx - dx, yy - cy - dy
    xs = np.clip((np.cos(a) * x + np.sin(a) * y) / scale + cx, 0, w - 1)
    ys = np.clip((-np.sin(a) * x + np.cos(a) * y) / scale + cy, 0, h - 1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = xs - x0, ys - y0
    f = images.astype(np.float64)
    out = (f[:, y0, x0] * (1 - fx) * (1 - fy) + f[:, y0, x1] * fx * (1 - fy)
           + f[:, y1, x0] * (1 - fx) * fy + f[:, y1, x1] * fx * fy)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def augment(images, y, kinds):
    """Label-preserving copies of the given images.

    flip    left-right mirror (faces are symmetric)
    rotate  +7 and -7 degrees
    zoom    0.9 and 1.1 about the center
    shift   1 pixel left, right, up, down
    (An "invert" option, the photographic negative of non-faces, was removed: in a 30-round
    Haar test the inverted non-faces took 13% of the weight, the search fitted them, and the
    run stalled after 13 rules.)
    Returns (images, labels, source index into the input, variant name).
    """
    unknown = set(kinds) - set(AUGMENT_KINDS)
    if unknown:
        raise ValueError(f"unknown augmentation {sorted(unknown)}; choose from {AUGMENT_KINDS}")
    n = len(images)
    parts = []
    if "flip" in kinds:
        parts.append(("flip", images[:, :, ::-1]))
    if "rotate" in kinds:
        parts += [("rot+7", _warp(images, angle_deg=7)), ("rot-7", _warp(images, angle_deg=-7))]
    if "zoom" in kinds:
        parts += [("zoom0.9", _warp(images, scale=0.9)), ("zoom1.1", _warp(images, scale=1.1))]
    if "shift" in kinds:
        parts += [(name, _warp(images, dx=dx, dy=dy)) for name, dx, dy in
                  (("shift+x", 1, 0), ("shift-x", -1, 0), ("shift+y", 0, 1), ("shift-y", 0, -1))]
    out_x = [p[1] for p in parts]
    out_y = [y] * len(parts)
    src = [np.arange(n)] * len(parts)
    names = [np.full(n, p[0], dtype=object) for p in parts]
    if not out_x:
        empty = np.zeros((0, 19, 19), dtype=np.uint8)
        return empty, np.zeros(0, dtype=np.int8), np.zeros(0, dtype=int), np.zeros(0, dtype=object)
    return (np.ascontiguousarray(np.concatenate(out_x)), np.concatenate(out_y).astype(np.int8),
            np.concatenate(src), np.concatenate(names))


@dataclass
class Dataset:
    images: np.ndarray    # (n, 19, 19) uint8: training originals, validation, then training copies
    y: np.ndarray         # (n,) int8 in {-1, +1}
    ids: list             # "train/face/x.pgm", "train/face/x.pgm#flip", ...
    train_idx: np.ndarray
    val_idx: np.ndarray
    audit_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    variant: np.ndarray | None = None   # "" for an original image, else the transform name

    def __post_init__(self):
        if self.variant is None:
            self.variant = np.full(len(self.y), "", dtype=object)

    @property
    def n(self):
        return len(self.y)


def load_dataset(root, limit_per_class=None, audit_frac=0.0, seed=0, augment_kinds=()):
    """Training copies are made only from training images outside the audit share, so no copy
    of a validation or audit image exists, and every copy stays on the training side."""
    root = Path(root)
    tr_x, tr_y, tr_ids = load_labeled_folder(root / "train", limit_per_class)
    va_x, va_y, va_ids = load_labeled_folder(root / "validate", limit_per_class)
    n_tr = len(tr_y)
    train_idx, audit_idx = np.arange(n_tr), np.array([], dtype=int)
    if audit_frac > 0:
        if not audit_frac < 0.5:
            raise ValueError("audit_frac must be below 0.5")
        rng = np.random.default_rng(seed)
        held = []
        for label in (1, -1):                        # stratified: same face share as the training folder
            idx = np.flatnonzero(tr_y == label)
            held.append(rng.permutation(idx)[:int(round(audit_frac * len(idx)))])
        audit_idx = np.sort(np.concatenate(held))
        train_idx = np.setdiff1d(train_idx, audit_idx)
    ids = ["train/" + i for i in tr_ids] + ["validate/" + i for i in va_ids]
    images, y = np.concatenate([tr_x, va_x]), np.concatenate([tr_y, va_y])
    variant = np.full(len(y), "", dtype=object)
    if augment_kinds:
        ax, ay, src, names = augment(tr_x[train_idx], tr_y[train_idx], augment_kinds)
        start = len(y)
        images, y = np.concatenate([images, ax]), np.concatenate([y, ay])
        ids += [f"{ids[train_idx[s]]}#{nm}" for s, nm in zip(src, names)]
        variant = np.concatenate([variant, names])
        train_idx = np.concatenate([train_idx, np.arange(start, start + len(ay))])
    return Dataset(images, y, ids, train_idx, np.arange(n_tr, n_tr + len(va_y)), audit_idx, variant)
