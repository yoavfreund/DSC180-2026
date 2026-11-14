import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def synthetic_images(n_face, n_non, seed=0):
    """Faces have a dark band over rows 5-7 (eyes) above a bright band; non-faces are noise."""
    rng = np.random.default_rng(seed)
    faces = rng.normal(135, 45, (n_face, 19, 19))
    faces[:, 5:8, 2:17] -= 12
    faces[:, 8:11, 2:17] += 6
    non = rng.normal(130, 45, (n_non, 19, 19))
    images = np.clip(np.concatenate([faces, non]), 0, 255).astype(np.uint8)
    y = np.concatenate([np.ones(n_face), -np.ones(n_non)]).astype(np.int8)
    return images, y


@pytest.fixture
def synthetic_dataset():
    from common.data import Dataset
    tr_x, tr_y = synthetic_images(150, 250, seed=1)
    va_x, va_y = synthetic_images(100, 150, seed=2)
    n_tr = len(tr_y)
    return Dataset(np.concatenate([tr_x, va_x]), np.concatenate([tr_y, va_y]),
                   [f"train/{i}" for i in range(n_tr)] + [f"validate/{i}" for i in range(len(va_y))],
                   np.arange(n_tr), np.arange(n_tr, n_tr + len(va_y)))
