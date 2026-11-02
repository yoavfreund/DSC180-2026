# Face Detection Data: Windows / WSL2 companion

The original files remain unchanged. This companion uses the same data split,
staged IDs, learner interface, FP targets, and C evaluator, with a Linux backend
for local Windows development. No changes to `agents/` are required.

## Local setup

In Ubuntu WSL2, install the build and sandbox tools once:

```bash
sudo apt-get update
sudo apt-get install -y build-essential bubblewrap
```

From PowerShell in the repository root:

```powershell
$repoLinux = (wsl -d Ubuntu -- wslpath -u (Get-Location).Path).Trim()
wsl -d Ubuntu -- python3 "$repoLinux/pipeline/stage_WIN.py" doctor
wsl -d Ubuntu -- python3 "$repoLinux/pipeline/stage_WIN.py" prepare
wsl -d Ubuntu -- python3 "$repoLinux/pipeline/stage_WIN.py" workspace sirio run-01
```

The workspace command prints its Windows path. Open that folder as a separate
Codex project/chat; its `learner/` is empty so you can write your own solution.
The generated `AGENTS.md` contains the cascade instructions and local commands,
and `.codex/config.toml` exposes the `validation` MCP tool through WSL. Trust
this new project when Codex asks, so it can load that project configuration.
Do not start your training-only learner session in this labeled repository.

Alternatively, start the installed Codex CLI from PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '\\wsl.localhost\Ubuntu\home\sirio\dsc180-work-win\sirio\run-01\start-codex.ps1'
```

This launcher passes the validation settings for this invocation, so it also
works before project-local configuration is trusted. It does not edit your
global Codex settings. Add `-Check` to verify configuration without starting a
model session. The execution-policy override is limited to this PowerShell
process, allowing the generated unsigned script on the WSL network path;
it does not change system policy. Substitute the actual workspace path printed
by staging.

Compile and test C code through WSL, using the workspace's Linux path:

```powershell
wsl -d Ubuntu --cd /home/sirio/dsc180-work-win/sirio/run-01 -- make -C learner
```

For manual validation from this repository's PowerShell terminal:

```powershell
wsl -d Ubuntu -- python3 "$repoLinux/harness/run_validation_WIN.py" /home/sirio/dsc180-work-win/sirio/run-01 --run-name sirio/run-01 --require-cascade
```

Start with this prompt in the separate workspace:

> Read AGENTS.md and build my own boosted cascade face learner in learner/.
> Train only on the supplied training data. Test compilation and use an
> internal training split before using the validation tool.

Run the companion's integration tests from PowerShell:

```powershell
wsl -d Ubuntu -- bash "$repoLinux/harness/tests/test_harness_WIN.sh"
```

Defaults inside Ubuntu are `~/dsc180-private-win` for trusted data/results and
`~/dsc180-work-win` for workspaces. Override with `DSC180_PRIVATE_WIN` and
`DSC180_WORK_WIN` **inside WSL** before staging; the workspace generator records
the private root explicitly for its validation server. Neither the Claude SDK
nor Windows Python is required. Codex uses your existing sign-in.

`prepare` stages the committed `HEAD` image bytes through `git archive`.
This matters on Windows: `core.autocrlf=true` can alter PGM files that Git
mistakes for text, changing their pixel bytes. The companion bypasses checkout
conversion without changing Git settings or the original images. Uncommitted
image edits are intentionally excluded. The staged hashes must match the spec.

Learner subprocesses run under Bubblewrap with no network and only the build
and current input mounted. The Windows Codex editing session does not inherit
Andrew's agent-level restrictions or audit hooks. This is a personal development
workflow; benchmark comparisons still need equivalent agent controls and a
single machine for final timing. See [SPEC_WIN.md](pipeline/SPEC_WIN.md).

## Reference: existing split

This repo holds the **MIT CBCL Face Database** (training set), split into a
training set and a held-out validation set for the boosted-cascade face
detector.

## Layout

```
DSC180-2026/
├── train/
│   ├── face/               # 2,065 face images
│   └── non-face/           # 3,866 non-face images
├── validate/
│   ├── face/               #   364 face images
│   └── non-face/           #   682 non-face images
├── pipeline/               # stage_WIN.py + SPEC_WIN.md (Windows companions)
├── eval/                   # shared C evaluator + its tests
├── harness/                # run_validation_WIN.py (WSL/Bubblewrap validation)
├── agents/<name>/          # each student's agent and its saved runs
└── DATA_SPLIT.md           # this file
```

The committed `face/` and `non-face/` folders are the team's source of truth.
Learner workspaces receive only staged training data. See
[How agents see the data](#how-agents-see-the-data).

| Class      | Original | Train | Validate | Validate % |
|------------|---------:|------:|---------:|-----------:|
| `face`     |    2,429 | 2,065 |      364 |      15.0% |
| `non-face` |    4,548 | 3,866 |      682 |      15.0% |
| **Total**  |    6,977 | 5,931 |    1,046 |      15.0% |

No file appears in both `train/` and `validate/`.

## The data

- **Source:** MIT Center for Biological and Computational Learning (CBCL)
  Face Database #1, training portion (`face.train.tar.gz`, no longer kept
  in the repo; the unsplit tree is in commit `25d972e`).
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

Run against the unsplit `train/` folder from commit `25d972e`, this
reproduces exactly the 1,046 files now in `validate/` (checked).

## Design decisions

- **15% validation, per class (stratified).** Each class is sampled
  separately, so validate keeps the train set's roughly 1 : 1.9
  face-to-non-face ratio. If we sampled 15% of the combined pool instead,
  the ratio in validate could drift from train's.
- **Moved, not copied.** Each source file belongs to one split. This prevents
  file-level overlap, but does not rule out duplicate or near-duplicate pixels.
- **Fixed seed (42).** Anyone can check or rebuild the exact split with the
  code above. If you change the split, update this document too.
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

## How agents see the data

An audit of the readable folders found several ways an agent could learn
labels without learning faces:

| Leak | Evidence | Fix |
|---|---|---|
| Folder and file names | `face/face00001.pgm`, `non-face/B1_00001.pgm` | neutral random ids |
| **File timestamps** | every face has mtime July 1997, every non-face April 1999 | one fixed mtime for all files |
| Git history | renaming in the repo would leave the labeled paths in history | agents run outside the repo |
| Validate labels | an agent able to read `validate/face/` could memorize those images by their pixels | agents never get validate images or labels |
| Public dataset | CBCL, test set included, can be downloaded | agents have no network |

So `pipeline/stage_WIN.py` builds a **staged** copy outside the repo
(`~/dsc180-private-win/staged/<split>/`). It has neutral ids from a seeded
permutation, rewritten files with one canonical header and one mtime,
`labels.txt`, and a `ledger.tsv` mapping each id back to its original. Train,
validate and (on test day) the CBCL test set all go through the same script.

- A personal workspace (`~/dsc180-work-win/<name>/<run>/`) holds only staged
  **train** images and labels plus an empty `learner/`.
- Validation goes through `harness/run_validation_WIN.py`, which returns
  rounded aggregate metrics.
- See `pipeline/SPEC_WIN.md` for the contract and the distinction between
  learner-process isolation and the personal Codex editing session.

```bash
python3 pipeline/stage_WIN.py prepare                 # run inside Ubuntu WSL
python3 pipeline/stage_WIN.py workspace sirio run-01    # empty personal learner
```

### Other audit findings

- **18 non-face patches are the same flat gray image** (σ = 0), and 5 of them
  are in validate. Learners must handle zero variance (see SPEC).
  There are also 3 duplicate face pairs, all within train.
- **Near-duplicates across splits are rare.** 0.3% of validate faces
  correlate above 0.99 with some train face, about the same rate as within
  train. The split stays as is.
- **Validate is close to saturated.** It comes from the same pool as train,
  and the median validate face correlates 0.945 with its nearest train face.
  A mean-face baseline already scores AUC 0.973, and the first agent run
  (`agents/andrew/runs/run-01`, 600-round AdaBoost) scored 0.99999. A
  shuffled-label control of that same code scores 0.42. That is useful evidence,
  but not proof that every leakage path is absent. Validate appears easy,
  making it a weak signal for
  telling strong solutions apart; the CBCL test set will be the real
  comparison.

## Caveats

- Some macOS `.DS_Store` files were committed before `.gitignore` existed.
  Git still tracks them until they are removed with `git rm --cached`.
