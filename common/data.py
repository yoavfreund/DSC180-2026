"""Load 19x19 CBCL PGM images.

The fixed split is the folder split the team already staged:
  <root>/train/{face,non-face}     agent-visible training images
  <root>/validate/{face,non-face}  hidden validation images (selection and alpha)
The CBCL test set is never loaded here.
"""
from dataclasses import dataclass
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


@dataclass
class Dataset:
    images: np.ndarray    # (n, 19, 19) uint8; training images first, then validation
    y: np.ndarray         # (n,) int8 in {-1, +1}
    ids: list             # "train/face/x.pgm", ...
    train_idx: np.ndarray
    val_idx: np.ndarray

    @property
    def n(self):
        return len(self.y)


def load_dataset(root, limit_per_class=None):
    root = Path(root)
    tr_x, tr_y, tr_ids = load_labeled_folder(root / "train", limit_per_class)
    va_x, va_y, va_ids = load_labeled_folder(root / "validate", limit_per_class)
    n_tr = len(tr_y)
    return Dataset(np.concatenate([tr_x, va_x]), np.concatenate([tr_y, va_y]),
                   ["train/" + i for i in tr_ids] + ["validate/" + i for i in va_ids],
                   np.arange(n_tr), np.arange(n_tr, n_tr + len(va_y)))
