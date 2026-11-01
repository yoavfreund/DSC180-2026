# Andrew's agent: benchmarks

Reference points for comparing runs. Everything was measured on the same
machine (Andrew's Mac, 10 threads available), through the same sandboxed
harness, on the staged validate set (364 faces, 682 non-faces).
Full-precision values come from each run's `runs.jsonl`.

## Validation accuracy

| Run | What it is | AUC | Detection @ 1% FP | @ 5% FP | @ 10% FP | Precision @ 1% FP |
|---|---|---:|---:|---:|---:|---:|
| baseline | mean-face template (`harness/tests/baseline_learner`), not agent-written | 0.973418 | 0.676 (6 FP) | 0.852 (34 FP) | 0.931 (67 FP) | 0.976 |
| **run-01** | agent, **no algorithm constraint**: monolithic discrete AdaBoost, 600 Haar stumps, no cascade | 0.999992 | 1.000 (2 FP) | 1.000 (2 FP) | 1.000 (2 FP) | 0.995 |
| **run-02** | agent, **cascade required**: Viola-Jones-style cascade, 5 stages of Real AdaBoost stumps (5, 11, 23, 37, 250) | 0.997645 | 0.995 (0 FP) | 0.995 (0 FP) | 0.995 (0 FP) | 1.000 |

run-02 loses 2 of 364 validate faces to early rejection (one at stage 1, one
at stage 3). That is the whole gap to run-01. In exchange, no non-face scores
above any surviving face.

## Speed

| Run | Train, wall | Train, CPU | Weak classifiers evaluated per non-face | per face | Predict CPU per image* |
|---|---:|---:|---:|---:|---:|
| baseline | 1.1 s | 0.2 s | — | — | — |
| **run-01** | 34.9 s | 257.6 s | 600 | 600 | 21.2 µs |
| **run-02** | **24.4 s** | **120.5 s** | **45.8** | 324.3 | **16.0 µs** |

\* Predict CPU per image is the slope between timing 1 image and 5,931
images, minimum of 7 repeats, with start-up and model loading (about 8 ms)
removed. The harness now measures it the same way, with all validate
images, the minimum of 3 runs, and one image subtracted. Re-running run-02's
learner through it gave 15.8 µs and the same weak-classifier counts.

**Where non-faces leave run-02's cascade** (validate, 682 non-faces):
stage 1: 317 (46%) · stage 2: 154 · stage 3: 104 · stage 4: 47 · reach the final stage: 60 (9%).

**What the speed numbers mean:**
- run-02 evaluates **13× fewer weak classifiers** per non-face than run-01,
  but its per-image CPU time is only **25% lower**.
- On pre-cropped 19×19 patches, most of the per-image cost is reading the
  file and building the variance-normalized integral image. The classifier
  itself is a small part.
- In real Viola-Jones detection, one integral image serves thousands of
  scanned windows, so the cascade's savings would dominate.
- **Weak classifiers evaluated per window** is the paper's own speed metric
  ("an average of 10 features ... per sub-window"). It's independent of the
  machine, so it's the fairer number for comparing teammates' solutions.

## Notes

- **Validate is close to saturated.** A shuffled-label control of run-01's
  code scores AUC 0.42, so the high scores are genuine, not a leak. But the
  baseline already reaches 0.973. The CBCL test set (FP targets 0.1%, 1%,
  10%) is the real comparison. Both learners are saved under `runs/*/learner/`.
- **run-01's timings were measured afterwards** with its saved build. The
  harness had no timing yet when run-01 ran.
- **Run settings:**
  - run-01: 15 turns, 1 of 5 validation calls, no algorithm constraint.
  - run-02: 17 turns, 1 of 8 validation calls, cascade required. It checked
    its design on its own 80/20 split of train before its single validation
    call.
