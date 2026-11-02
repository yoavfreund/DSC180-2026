# Learner and Evaluation Spec — Windows / WSL2 companion

This companion preserves the learner interface and evaluator used on macOS.
Run Python, `make`, and the C executables inside Ubuntu WSL2. The Windows
Codex app can edit your separate learner workspace. Original files and the
`agents/` directory are not modified by this workflow.

See [DATA_SPLIT_WIN.md](../DATA_SPLIT_WIN.md) for setup and commands.
`harness/run_validation_WIN.py` uses Bubblewrap, not `learner.sb`, to give each
learner step an empty root with read-only system tools, a writable build mount,
and only that step's read-only data mount. Network and PID namespaces are isolated.
It fails closed if Bubblewrap is unavailable. The validation key, ledger,
repository, Windows drives, host home, and private logs are not mounted.

The Codex editing session itself is a personal development session, not the
Claude SDK agent. These companions do not reproduce Andrew's agent tool hooks,
tool audit logs, turn budget, or agent-level filesystem restrictions. Keep your
own learner training-only; these runs are not automatically equivalent to the
original isolated benchmark.

Every student's solution follows this contract, so one evaluator can score all
of them the same way. If you change it, tell the team, because everyone's
learner has to match.

## Data the learner sees

Staged by `pipeline/stage_WIN.py`. Nothing about a file except its pixels says
what it is:

```
images/000001.pgm ...   19×19 8-bit binary PGM, header exactly "P5\n19 19\n255\n"
labels.txt              "<id> <label>" per line, 1 = face, 0 = non-face (train only)
```

- Ids are a seeded random permutation (team seed 42), the same on every
  machine. Tested identical under Python 3.12 and 3.14. After
  `stage_WIN.py prepare`, check that yours matches:

  | Staged file | sha256 |
  |---|---|
  | `train/labels.txt` | `ada12695704433989298955081028bbde0aaeb457ea820a27449a00e44d37001` |
  | `train/images/*.pgm` (concatenated in name order) | `3d14b1331f3fdd397cd70fbe820b0763544da90a71cd2fd6cbe99c07b23bf665` |
  | `validate/labels.txt` | `74d414880a2db3b13f66aead27ee2584e8dce4b829b12e6012d0168e7c8be40a` |
  | `validate/images/*.pgm` | `152233842af3700f8ebb2270d73d3bd936319d53ab11ac984e579fd03c530243` |

  ```bash
  cd ~/dsc180-private-win/staged
  sha256sum train/labels.txt validate/labels.txt
  cat train/images/*.pgm | sha256sum
  cat validate/images/*.pgm | sha256sum
  ```
- Every file has the same mtime. File order, names and timestamps carry no label.
- **Constant images exist.** 18 patches in the original pool are a single flat gray
  (σ = 0): 13 in train and 5 in validate. Variance normalization must not
  divide by zero, and the score must stay finite.

## Learner contract

The learner is a directory with a `Makefile`. `make` must produce two
executables:

```
./train   <images_dir> <labels.txt> <model_out>
./predict <model> <images_dir>          # stdout: "<id> <score>" per image
```

- `score` is any finite double, and **higher means more face-like**.

### Cascades (proposed by Andrew; needs team agreement)

- **Score convention:** an image rejected at stage *k* must score lower than
  every image that reaches stage *k*+1. Among images that stop at the same
  stage, that stage's margin decides the order. Example:
  with one-based stage numbers, use
  `score = (k - 1) + sigmoid(margin)`, with sigmoid clamped away from 0 and 1
  to avoid floating-point rounding across interval boundaries. Scores for stage
  k must lie in `[k-1, k)`. For WaldBoost, each weak classifier counts as a stage.
  Sweeping this scalar score also orders early-rejected examples; ROC points
  below the last-stage interval are not obtainable by changing only the final
  stage threshold while holding all earlier thresholds fixed.
- **Layout report:** `./predict --describe <model>` prints
  `stage=<k> weak=<n>` for each stage, reading only the model file. With
  `--require-cascade` (Andrew's agent sets it), the harness rejects learners
  that report fewer than 2 stages. Stage numbers must be consecutive starting
  at 1, and every weak count must be positive. This checks the declared layout,
  not the implementation: an independent code review is still needed to verify
  that the learner actually executes the reported cascade.
- `predict` scores every `*.pgm` in the directory exactly once, and the id is
  the filename without `.pgm`.
- Learners read only the paths passed as arguments and use no network. The
  harness enforces this: `make`, `train` and `predict` run under
  Bubblewrap in `harness/run_validation_WIN.py`.

## Evaluator

```
eval/eval <scores.txt> <labels.txt> --fp-rates 0.01,0.05,0.1
eval/eval <scores.txt> <labels.txt> --fp-counts 10,50,100
```

- If the ids in the scores file don't match the key exactly (missing,
  duplicate or extra ids, or NaN/inf scores), it exits 1 and scores nothing.
- An image counts as a face prediction when score ≥ threshold. Tied scores
  are always all in or all out.
- For each FP target, eval reports the operating point with the most true
  positives whose false positives stay within the target (a rate target r
  allows floor(r · n_neg) false positives). Among points with equal true
  positives, it takes the one with the fewest false positives.
- Output, as fixed `key=value` lines:
  ```
  n_pos=364 n_neg=682 auc=0.973210
  fp_rate=0.01 fp_allowed=6 threshold=... tp=246 fp=6 tpr=0.675824 fpr=0.008798 precision=0.976190
  ```
- Tests in Ubuntu WSL: `bash harness/tests/test_harness_WIN.sh`. The suite
  also exercises the original evaluator's hand-checked mock classifiers
  (perfect, inverted, constant, a 6-image case with ties), malformed-input
  rejection, and a 500-trial random crosscheck against a brute-force
  reference. It catches the deliberately broken versions we tried: splitting
  ties, and picking a worse point at the same true-positive count.

**Small-sample caution:** validate has only 682 non-faces, so FP rate moves in
steps of about 0.15%, and a 0.1% target allows zero false positives. Low-FP
targets mean more on the CBCL test set (23,573 non-faces).

## Validation during agent runs

`harness/run_validation_WIN.py` is the validation feedback entry point.
One call takes a snapshot of `learner/` and runs `make`, then `./train` on
staged train, then `./predict` on staged validate images (without the key),
then `eval`. It returns:

```
validation: 364 faces, 682 non-faces, AUC 0.973
at FP rate <= 0.01: detection rate 0.676, precision 0.976, false positives 6
...
timing: train 24.1 s wall (123.5 s CPU); predict 15.8 us CPU per image (start-up and model loading, 7.6 ms, excluded)
predict batch: 1046 images, 0.025 s wall (0.024 s CPU), including start-up, model loading and image I/O
cascade: 5 stages, weak classifiers per stage: 5,11,23,37,250                    (only with --require-cascade)
weak classifiers evaluated per image: non-faces 45.8, faces 324.3 (of 326)   (only with --require-cascade)
```

- **Train timing** is wall-clock and CPU seconds. CPU time counts every
  thread, so it varies less from machine to machine.
- **Predict batch timing** reports wall-clock and CPU seconds for all 1,046
  validation images, including startup, model loading, image I/O and scoring.
- **Model size and memory** are also reported: model bytes and train/predict
  peak RSS. A trusted GNU time wrapper measures each command's maximum resident
  set in KiB outside the learner's writable mount; it is not the AI agent's
  memory, nor a cumulative maximum carried over from earlier commands.
- The WSL companion uses Bubblewrap `--as-pid-1` so the child CPU accounting
  includes the learner workload rather than only Bubblewrap's parent process.
- **Predict timing** is CPU time per image with start-up and model loading
  removed: the difference between predicting all validate images and
  predicting one image, each the minimum of 3 runs. At this scale, start-up
  (about 8 ms) would otherwise dominate.
- **Weak classifiers evaluated** is the Viola-Jones paper's own speed metric,
  and it doesn't depend on the machine. The harness reads each image's exit
  stage from its score, which requires the cascade score convention above.
- For the final table, time every learner on the same machine.

- Metrics are rounded to 3 decimals, and results are aggregates only, never
  per image.
- Each run gets a limited number of calls (default 20). The Windows companion
  persists the call count in the run log and locks each run across processes,
  so reconnecting MCP or repeating the CLI command does not reset the budget.
- `predict`'s stderr is never shown to the agent, because predict reads
  validation images.
- Full-precision results go to `~/dsc180-private-win/runs/<agent>/<run>/runs.jsonl`
  inside Ubuntu, or the root selected by `DSC180_PRIVATE_WIN`.
- Timing reruns must exit successfully. Scores and trusted evaluator outputs
  stay outside the learner's writable mount. Symlinks created during a learner
  step are rejected, as are symlinks present in the input learner directory.

## FP targets (fixed 2026-10-31; don't change after results are seen)

| Where | FP rates | False positives allowed |
|---|---|---|
| **Final test-set report** | **0.1%, 1%, 10%** | 23, 235, 2,357 of 23,573 non-faces |
| Agent validation feedback | 1%, 5%, 10% (+ AUC) | 6, 34, 68 of 682 non-faces |

- These are rates, not counts, so they mean the same thing on any set size.
- The 0.1% point is left out of validation because 0.1% of 682 allows zero
  false positives, which would make it mostly noise.
- The two sets share 1% and 10%, which shows each solution's drop from
  validate to test.
- The Viola-Jones paper (Table 2) reports counts of false detections from a
  full-image scan (10 to 167 out of about 75M windows). Those don't transfer
  to pre-cropped patches.
- Every run reports detection rate (recall) and precision. Precision isn't
  comparable between validate (35% faces) and test (2% faces).

## Test day

1. Stage the re-downloaded CBCL test set with `stage_WIN.py split test <src> ~/dsc180-private-win/staged/test`.
2. Run each frozen learner's `predict` once, under the same sandbox.
3. Run `eval ... --fp-rates 0.001,0.01,0.1` once per learner, on the same
   machine, recording train and predict times.
4. Collect one row per student.
