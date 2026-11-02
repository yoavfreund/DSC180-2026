# Run-01: shared evaluator results and corrected timing

The hidden-validation results use the repository's `eval/eval.c`, not the
learner's local `evaluate.py`. The latter was used only for the internal
training holdout. The shared evaluator was recompiled and run with
`--fp-rates 0.01,0.05,0.1`; its output exactly matched the original validation.

Validation contains 364 faces and 682 non-faces. AUC: **0.997637**.

| FP-rate limit | Detection (recall) | Precision | True positives | False positives |
|---|---:|---:|---:|---:|
| 1% | 98.3516% | 99.4444% | 358 / 364 | 2 / 682 |
| 5% | 99.4505% | 92.1120% | 362 / 364 | 31 / 682 |
| 10% | 99.4505% | 92.1120% | 362 / 364 | 31 / 682 |

Targets are upper limits, not required FP counts. The evaluator selects the
most true positives within each limit, breaking equal-TP ties toward fewer
false positives. That is why 5% and 10% report the same point.

## Corrected WSL measurements

| Operation | Images | Wall-clock time | CPU time (user + system) |
|---|---:|---:|---:|
| Train on all training data | 5,931 | 26.3209 s | 22.5856 s |
| Predict on validation, full batch | 1,046 | 0.372051 s | 0.032478 s |

Full-batch prediction includes process startup, model loading, image I/O,
classification and score output. A separate warm timing, taking the minimum
of three full-batch and three one-image predictions with output discarded,
estimates startup-subtracted cost at **3.71 microseconds wall** and
**3.84 microseconds CPU per image**. These are different measurements from
the complete batch and should not be substituted for its total runtime.

The frozen `build_001` learner was used. Retraining reproduced the validated
model byte-for-byte; re-evaluation reproduced all original accuracy metrics.
No algorithm changes, agent session, hyperparameter tuning, or additional
agent-validation call was used for this report. The budget remains 1/20.

The earlier harness CPU results (0.001 s training CPU and 0.2 microseconds
prediction CPU per image) are invalid. Bubblewrap's intermediate PID-1 reaper
prevented the parent from accounting for the learner's CPU use. The Windows
companion now runs the learner as namespace PID 1, preserving isolation and
correct child accounting. A synthetic CPU-workload regression and all existing
harness tests passed after this correction.

Raw corrected measurements and source hashes: [benchmark.json](benchmark.json).
Original validation log, retained unchanged: [runs.jsonl](runs.jsonl).
