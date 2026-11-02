# Surya's agent: benchmarks

Reference results for two C learners generated and run locally through the
Windows/WSL sandboxed harness. Successful runs use the same staged copy of
the repository's committed `validate/` set: **364 faces and 682 non-faces**.
Accuracy is measured with the shared `eval/eval.c`. Internal training-split
results are separate from hidden validation.

## Validation accuracy

| Run | Design | AUC | Detection @ 1% FP | @ 5% FP | @ 10% FP | Precision @ 1% FP |
|---|---|---:|---:|---:|---:|---:|
| **run-01** | Discrete AdaBoost Haar cascade; fixed stages 8 / 24 / 48 / 80 | 0.997637 | 98.3516% (2 FP) | 99.4505% (31 FP) | 99.4505% (31 FP) | 99.4444% |
| **run-02** | Same feature family; adaptive construction and independent calibration; selected 14 / 37 / 64 / 45 | 0.996898 | 98.6264% (6 FP) | 98.9011% (18 FP) | 98.9011% (18 FP) | 98.3562% |

FP targets are upper limits. With 682 negatives, the integer budgets are
6, 34, and 68 false positives. The evaluator maximizes true positives within
each budget and breaks equal-TP ties toward fewer false positives. Tied
scores can make the 5% and 10% targets select the same operating point.

Run-02 detects one more face than run-01 at the 1% FP ceiling (359 versus
358), with four more false positives. At the 5% and 10% ceilings it detects
two fewer faces (360 versus 362), with 13 fewer false positives. These small
differences on one validation set do not establish a general improvement.

## Speed

| Run | Train wall | Train CPU | Predict batch wall | Predict batch CPU | Predict CPU per image* |
|---|---:|---:|---:|---:|---:|
| **run-01** | 26.3209 s | 22.5856 s | 0.372051 s | 0.032478 s | 3.84 µs |
| **run-02** | 29.668 s | 29.664 s | 0.173 s | 0.022 s | 3.6 µs |

Batch prediction covers all 1,046 validation images and includes startup,
model loading, image I/O, classification, and score output. CPU means user
plus system CPU. Training times cover final fitting and calibration, not
the AI agent session or the complete architecture search.

\* Per-image CPU is a separate startup-subtracted estimate: minimum of three
full-batch measurements minus minimum of three one-image measurements,
divided by 1,045. It must not replace full-batch runtime. WSL filesystem,
caching, and startup costs affect wall time; the small per-image differences
are insufficient to claim a robust speed improvement.

Run-01 timings were corrected afterwards using its frozen build; retraining
reproduced the validated model byte-for-byte and the shared evaluator
reproduced its accuracy. The original run-01 harness CPU figures were invalid
because Bubblewrap's intermediate reaper lost learner CPU accounting. This
table uses the corrected measurements with `--as-pid-1`, not those original
CPU figures. Run-02 used the corrected harness.

## Classifier work, model size, and memory

| Run | Stages | Total weak classifiers | Mean weak evaluations per non-face | Per face | Model bytes | Train peak RSS | Predict peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| **run-01** | 4 | 160 | 21.5 | 158.3 | 51,272 | Not recorded | Not recorded |
| **run-02** | 4 | 160 | 27.5 | 158.4 | 4,922 | 156.23 MiB | 2.23 MiB |

Weak-evaluation means are validation aggregates from the trusted harness,
rounded to one decimal. Run-02 evaluates about 28% more weak classifiers per
non-face than run-01, despite having the same total classifier count.
This work measure is less dependent on machine and file I/O than elapsed time.

Model size is the serialized learner artifact, excluding executable code.
Run-01's binary model is 51,272 bytes; run-02's compact text model is 4,922
bytes. Serialization and stored metadata differ, so this reduction does not
mean run-02 has fewer weak classifiers. Peak RSS is process resident memory,
not model size or the AI agent's memory. Run-01 RSS was not recorded and has
not been inferred from another run.

## Design and search limits

Both runs use discrete AdaBoost with Haar decision stumps, 6,500 fixed
candidate feature geometries, 128 response bins, coefficient cap 2, and
training-only augmentation. No external training data was supplied.

| Run | Stage construction and calibration | Search / validation usage |
|---|---|---|
| **run-01** | Fixed 8 / 24 / 48 / 80 architecture; thresholds target 99.85% positive training retention per stage, with subtractive offset 0.25 | One successful hidden-validation call out of 20 allowed |
| **run-02** | Three constructions with conditional negative-pass targets 0.50 / 0.35 / 0.20; ceilings of 64 rules per stage, eight stages, 160 total; independent calibration | Three constructions, 27 threshold settings; one hidden-validation call out of two allowed |

Run-02 chose its targets and selection rules before hidden
validation. Its threshold grid crosses face retention {0.995, 0.998, 0.999}
with margin offset {0, 0.25, 0.50}. Threshold sweeps reuse fitted stumps;
27 settings do not mean 27 architecture constructions. Selection prefers
at least 99% internal detection at <=1% FP, then lowest mean non-face weak
evaluations; otherwise it falls back to highest detection, then lowest cost.

Run-02 reserves 1,186 original training images for independent calibration and
1,186 for selection during search; construction weights use the remaining
3,559 originals. Final fitting combines the fitting and selection partitions
(4,745 originals) while retaining the independent calibration partition.
Splitting precedes augmentation. Run-01's final fit instead uses all 5,931
training originals, with thresholds calibrated on training positives.

Run-02 selected candidate C with retention 0.995 and offset 0.25. All three
constructions reached 160 classifiers; none met the cumulative 0.5%
calibration-negative target or the 99% internal selection preference.

## Sources

- **Run-01:** [results](runs/run-01/RESULTS.md), [corrected benchmark](runs/run-01/benchmark.json), [original validation log](runs/run-01/runs.jsonl), [learner](runs/run-01/learner/).
- **Run-02:** [results](runs/run-02/RESULTS.md), [machine-readable measurements](runs/run-02/results.json), [validation log](runs/run-02/runs.jsonl), [protocol audit](runs/run-02/protocol_audit.json).

These are validation benchmarks for pre-cropped patches, not an evaluation
of full-image face localization or an independent final test set. No baseline
was rerun for this report. Andrew's benchmark measurements come from a
different machine and are not mixed into these timing tables.
