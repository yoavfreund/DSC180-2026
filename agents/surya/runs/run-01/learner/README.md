> Updated results: [shared eval.c metrics and corrected wall/CPU timing](../RESULTS.md). This report supersedes the earlier invalid harness CPU measurements.

# Haar AdaBoost cascade

Build with `make`. Train with `./train ../train/images ../train/labels.txt model.bin`.
Predict with `./predict model.bin <images_dir>`; inspect with
`./predict --describe model.bin`.

The model is a Viola–Jones-style cascade with independent discrete AdaBoost
stages of 8, 24, 48, and 80 decision stumps. Features are two-rectangle,
three-rectangle, and checkerboard Haar contrasts, evaluated with an integral
image and divided by patch standard deviation (floored at 1 intensity unit).
There are 6,500 reproducibly sampled candidate features. Feature responses
are quantized into 128 bins with a scale of 24 bins per normalized contrast
unit. Weighted threshold search is exhaustive over the binned responses;
Each AdaBoost weak-classifier coefficient is capped at 2.

Each stage threshold is the 0.15th percentile of positive training scores
minus 0.25, targeting at least 99.85% training face retention per stage.
All positive examples are used at every stage, including horizontal flips.
Only negatives that pass the earlier stages train later stages. To reduce
negative exhaustion, each supplied negative is supplemented by a horizontal
flip and four one-pixel translations (edge replicated). No external data is
used. If no negatives remain after at least two stages, training stops.

Inference stops at the first rejecting stage. The returned score is the
zero-based stage index plus sigmoid(stage score minus stage threshold),
clamped to [1e-9, 1-1e-9]. Thus all scores stay within their stage intervals.
The final stage also ranks accepted faces by its margin.

The binary model embeds each selected feature and all stage thresholds;
`--describe` needs only this file. Training is deterministic. PGM inputs
must be binary 19 by 19 patches with maximum value 255.

## Local evaluation

A fixed-seed (7321) shuffled 80/20 split of the supplied training patches
held out 405 faces and 782 non-faces. Splitting preceded augmentation, so
transformed versions of held-out patches did not enter training. This is an
internal development check, not the private validation or final test set.

| False-positive target | Detection | Precision |
|---|---:|---:|
| 0.1% | 96.79% | 100.00% |
| 1% | 99.75% | 98.30% |
| 5% | 100.00% | 91.22% |
| 10% | 100.00% | 83.85% |

Internal AUC: 0.999886. At the 0.1% target, the 782-negative holdout permits
zero false positives, so this estimate has limited resolution. Held-out
negatives averaged 23.41 weak evaluations; faces averaged 159.80. Prediction
on all 5,931 supplied images with the split model took 39.00 seconds wall
and 0.783 seconds CPU on this WSL filesystem, including file I/O.

The final `model.bin` is trained on every supplied training patch. The
original non-faces averaged 21.07 weak evaluations; the final augmented
negative survivor counts were 23,196 -> 6,991 -> 1,633 -> 218 -> 0.

## Trusted hidden validation

The trusted server completed validation successfully (call 1 of 20 in this
session), using 364 faces and 682 non-faces. No hyperparameters were changed
in response to validation.

| False-positive target | Detection | Precision | False positives |
|---|---:|---:|---:|
| 1% | 98.4% | 99.4% | 2 |
| 5% | 99.5% | 92.1% | 31 |
| 10% | 99.5% | 92.1% | 31 |

AUC: 0.998. The server reported training at 25.1 seconds wall (0.0 seconds
CPU), prediction at 0.2 microseconds CPU per image, and 1.1 milliseconds
startup/model loading excluded from that prediction figure. These are the
server's reported timings; the independent local timings below include file
I/O and are not directly comparable. The server did not report prediction
wall time. Non-faces averaged 21.5 weak evaluations and faces 158.3, out of
160 total. The confirmed layout was 8/24/48/80 weak classifiers.

Earlier attempts recorded in this directory were blocked by an approval-policy
error; this session's successful call supersedes that validation limitation.

The timed final full-data fit took 49.43 seconds wall and 22.19 seconds CPU.
Both timings include data reads; wall times are sensitive to WSL filesystem
latency. The final build completed with `-Wall -Wextra` and no warnings.
All 5,931 full-data predictions were checked for unique IDs, finite scores,
and valid stage intervals. `--describe` reported 8/24/48/80 weak classifiers.

Timing verification note: the WSL/Bubblewrap harness recorded only 0.001 seconds of training CPU for 25.11 seconds of training wall time. Its CPU accounting does not capture the learner workload correctly, so the reported 0.2 microseconds CPU per image must not be used for performance comparisons. Validation accuracy and the score-derived weak-classifier counts are unaffected. Raw reported measurements are preserved in ../runs.jsonl.

