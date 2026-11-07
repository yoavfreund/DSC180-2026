import numpy as np


def seeded_split(n, seed, ratio):
    """Disjoint random train and validation indices; ratio is validation share."""
    if n < 2 or not 0 < ratio < 1:
        raise ValueError("split requires at least two rows and 0 < ratio < 1")
    count = min(n - 1, max(1, round(n * ratio)))
    order = np.random.default_rng(seed).permutation(n)
    return order[count:].astype(np.int32), order[:count].astype(np.int32)
