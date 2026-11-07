"""Load the original 19x19 CBCL PGM files and extract C image features."""
import ctypes as C
from pathlib import Path

import numpy as np


def _read_pgm(path):
    with path.open("rb") as f:
        tokens = []
        while len(tokens) < 4:
            line = f.readline()
            if not line:
                raise ValueError(f"incomplete PGM header: {path}")
            tokens.extend(line.split(b"#", 1)[0].split())
        if tokens != [b"P5", b"19", b"19", b"255"]:
            raise ValueError(f"expected 19x19 8-bit PGM: {path}")
        data = f.read()
    if len(data) != 361:
        raise ValueError(f"invalid PGM data length: {path}")
    return np.frombuffer(data, dtype=np.uint8).copy().reshape(19, 19)


def transform_image(image, angle=0.0, dx=0, dy=0):
    """Nearest-neighbor rotation/translation, padded with the image edge."""
    yy, xx = np.indices(image.shape, dtype=np.float64)
    cy, cx = (np.array(image.shape) - 1) / 2
    a = np.deg2rad(angle)
    x, y = xx - cx - dx, yy - cy - dy
    src_x = np.rint(np.cos(a) * x + np.sin(a) * y + cx).astype(int)
    src_y = np.rint(-np.sin(a) * x + np.cos(a) * y + cy).astype(int)
    return image[np.clip(src_y, 0, image.shape[0]-1), np.clip(src_x, 0, image.shape[1]-1)]


def extract_features(images, library):
    lib = C.CDLL(str(Path(library).resolve()))
    u8p = C.POINTER(C.c_uint8)
    u64p = C.POINTER(C.c_uint64)
    lib.lbp_u8.argtypes = [u8p, C.c_size_t, C.c_size_t, u8p]
    lib.integral_image_u8.argtypes = [u8p, C.c_size_t, C.c_size_t, u64p]
    X = np.empty((len(images), 714), dtype=np.float64)
    edges = np.linspace(0, 19, 9, dtype=int)
    for i, image in enumerate(images):
        image = np.ascontiguousarray(image, dtype=np.uint8)
        lbp = np.empty_like(image)
        integral = np.empty((20, 20), dtype=np.uint64)
        if lib.lbp_u8(image.ctypes.data_as(u8p), 19, 19, lbp.ctypes.data_as(u8p)) or \
           lib.integral_image_u8(image.ctypes.data_as(u8p), 19, 19, integral.ctypes.data_as(u64p)):
            raise RuntimeError("feature extraction failed")
        X[i, :361] = image.ravel() / 255.0
        X[i, 361:650] = lbp[1:-1, 1:-1].ravel() / 255.0
        k = 650
        for y0, y1 in zip(edges[:-1], edges[1:]):
            for x0, x1 in zip(edges[:-1], edges[1:]):
                total = (int(integral[y1, x1]) - int(integral[y0, x1])
                         - int(integral[y1, x0]) + int(integral[y0, x0]))
                X[i, k] = total / ((y1-y0)*(x1-x0)*255)
                k += 1
    return X


def load_images(root, library, limit_per_class=None, augment=False):
    paths = []
    for split in ("train", "validate"):
        for label, folder in ((1, "face"), (-1, "non-face")):
            items = sorted((Path(root) / split / folder).glob("*.pgm"))
            if limit_per_class is not None:
                items = items[:limit_per_class]
            paths.extend((path, label) for path in items)
    if not paths:
        raise ValueError("no images found in train/ and validate/")
    images = np.stack([_read_pgm(path) for path, _ in paths])
    y = np.array([label for _, label in paths], dtype=np.int32)
    X = extract_features(images, library)
    variants = None
    if augment:
        variants = np.stack([extract_features([transform_image(im, angle=angle, dx=dx, dy=dy)
                                               for im in images], library)
                             for angle, dx, dy in ((5, 0, 0), (-5, 1, 0))])
    return X, y, [str(p.relative_to(root)) for p, _ in paths], variants
