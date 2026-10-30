# Face Detection Data: Train / Validate Split

This repo holds the **MIT CBCL Face Database** (training set), split into a
training set and a held-out validation set for the boosted-cascade face
detector.

## Layout

```
DSC180-2026/
├── face.train.tar.gz       # original download, full UNSPLIT set (raw backup)
├── train/
│   ├── face/               # 2,065 face images
│   └── non-face/           # 3,866 non-face images
├── validate/
│   ├── face/               #   364 face images
│   └── non-face/           #   682 non-face images
├── validate_manifest.txt   # exact list of files placed in validate/
└── DATA_SPLIT.md           # this file
```

| Class      | Original | Train | Validate | Validate % |
|------------|---------:|------:|---------:|-----------:|
| `face`     |    2,429 | 2,065 |      364 |      15.0% |
| `non-face` |    4,548 | 3,866 |      682 |      15.0% |
| **Total**  |    6,977 | 5,931 |    1,046 |      15.0% |

No file appears in both `train/` and `validate/`.

## The data

- **Source:** MIT Center for Biological and Computational Learning (CBCL)
  Face Database #1, training portion (`face.train.tar.gz`).
- **Format:** binary PGM (`P5`), **19×19 pixels**, 8-bit grayscale
  (max value 255). Each file is about 374 bytes: a short text header followed
  by 361 pixel bytes.
- **Faces** (`face00001.pgm` … `face02429.pgm`): cropped, roughly aligned
  frontal faces.
- **Non-faces:** background patches. The filename prefix names the source
  image they were cut from:

  | Prefix        | Original | Train | Validate |
  |---------------|---------:|------:|---------:|
  | `B20`         |    2,799 | 2,376 |      423 |
  | `B1`          |      559 |   491 |       68 |
  | `B5`          |      341 |   284 |       57 |
  | `graves111`   |      228 |   191 |       37 |
  | `goldwater67` |      228 |   204 |       24 |
  | `geyser27`    |      228 |   190 |       38 |
  | `GULF`        |      165 |   130 |       35 |

### Loading an image

```python
from PIL import Image
import numpy as np

img = np.asarray(Image.open("train/face/face00001.pgm"))  # shape (19, 19), dtype uint8
```

## How the split was made

1. For each class (`face`, `non-face`) on its own, list every `*.pgm` in
   `train/<class>/` and **sort the names** so the order doesn't depend on the
   filesystem.
2. Draw `floor(0.15 * n)` files at random with Python's `random.Random(42)`,
   using one RNG shared across both classes: `face` first, then `non-face`.
3. Move the chosen files with `git mv` into `validate/<class>/`, so git
   records them as renames.
4. Write the chosen paths to `validate_manifest.txt`, one `class/filename`
   per line.

The selection code:

```python
import random
from pathlib import Path

SEED, FRAC = 42, 0.15
rng = random.Random(SEED)
for cls in ["face", "non-face"]:
    files = sorted(p.name for p in Path("train", cls).glob("*.pgm"))
    k = int(len(files) * FRAC)
    for name in sorted(rng.sample(files, k)):
        print(f"{cls}/{name}")
```

Run against the full, unsplit `train/` folder, this prints exactly the
contents of `validate_manifest.txt`.

## Design decisions

- **15% validation, per class (stratified).** Each class is sampled
  separately, so validate keeps the train set's roughly 1 : 1.9
  face-to-non-face ratio. If we sampled 15% of the combined pool instead,
  the ratio in validate could drift from train's.
- **Moved, not copied.** Keeping each image in only one place makes leakage
  between train and validate impossible.
- **Fixed seed (42) and saved manifest.** Anyone can check or rebuild the
  exact split. If you change the split, update the manifest and this
  document too.
- **Round down.** `floor(0.15 × 2429) = 364` and `floor(0.15 × 4548) = 682`.
- **Non-faces sampled from the whole pool, not per source image.** We drew
  15% of all non-faces at random, without balancing each prefix (`B20`,
  `B1`, …). The per-prefix share in validate therefore ranges from about
  10.5% (`goldwater67`) to about 21% (`GULF`). That seemed fine for a first
  pass. If source-image bias becomes a concern (for example, the detector
  overfitting to one background), switch to per-prefix stratification, or
  hold out whole source images.
- **Validate is not a test set.** Use it for model selection and tuning
  (cascade thresholds, number of boosting rounds, and so on). For a final,
  untouched estimate of performance, use the separate CBCL **test** set,
  which this repo doesn't include yet.

## Caveats

- `face.train.tar.gz` still contains the **full, unsplit** set. Extracting it
  over the repo restores the original `train/` folder and breaks the split.
  Treat it as a raw backup only.
- macOS `.DS_Store` files have been committed in a few places. They can be
  ignored, and adding them to a `.gitignore` is a good idea.
