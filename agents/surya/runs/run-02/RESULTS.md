# Run-02 results

Run-02 keeps discrete AdaBoost stumps, the same 6,500 candidate Haar feature
geometries and generation seed, normalization, 128 response bins, coefficient
cap, and augmentation. It changes stage construction and independent threshold
calibration. The agent had access only to this run's learner, training data,
and instructions, plus the trusted aggregate validation tool.

The original prompt is archived at
[run-01/system_prompt_run01.md](../run-01/system_prompt_run01.md).
This experiment's frozen prompt is [system_prompt_run02.md](system_prompt_run02.md).

## Frozen design and search limit

Exactly **three complete candidate constructions** were performed. Per-stage
conditional non-face pass targets were fixed before training at A=0.50,
B=0.35, and C=0.20, with calibration-face retention 0.998. Each stage grew from
at least two stumps until its targets were met or a fixed budget was reached:
64 stumps/stage, eight stages, and 160 stumps overall. The cumulative
calibration-negative stopping target was 0.5%.

| Candidate | Learned stage sizes | Total classifiers |
|---|---|---:|
| A | 6 / 15 / 24 / 28 / 63 / 24 | 160 |
| B | 10 / 24 / 44 / 51 / 31 | 160 |
| C — selected | **14 / 37 / 64 / 45** | **160** |

Each candidate's fixed stumps were tested with nine threshold settings:
retention {0.995, 0.998, 0.999} crossed with subtractive margin offset
{0, 0.25, 0.50}. That is **27 settings**, not 27 architecture constructions.
Some threshold evaluations were repeated only to complete metadata; stumps
were not refitted for those repetitions.

The fixed selection rule preferred at least 99% detection at <=1% FP, then
lowest mean non-face weak evaluations. No candidate met 99%, so the declared
fallback chose highest detection, then lowest mean non-face evaluations.
It selected **C, retention 0.995, offset 0.25**: 412/417 selection faces
detected, with 7/769 false positives, and 26.03 mean non-face evaluations.

Final fitting retrained the selected stage sizes and calibration rule, without
performing another architecture search. The final stage thresholds are
-2.07372755579, -2.90678696595, -2.95590503832, and -1.73088829145.

## Hidden-validation accuracy

These results come from **the repository's shared `eval/eval.c`**, using the
staged copy of the committed `validate/` folder: **364 faces and 682 non-faces**.
They are separate from the internal selection results. Full-precision values
below are from the trusted host log; the agent received rounded aggregates.

**AUC: 0.996898.**

| FP-rate ceiling | Detection / recall | Precision | True positives | False positives |
|---|---:|---:|---:|---:|
| 1% | **98.6264%** | **98.3562%** | 359 / 364 | 6 / 682 |
| 5% | **98.9011%** | 95.2381% | 360 / 364 | 18 / 682 |
| 10% | **98.9011%** | 95.2381% | 360 / 364 | 18 / 682 |

The evaluator maximizes true positives within each integer FP budget, breaking
equal-TP ties toward fewer FPs. Hence the 5% and 10% ceilings select the same
operating point. Exactly **one hidden-validation call** was used. No code,
architecture, model, plan, or threshold hyperparameter was changed afterward.

## Classifier work and size

| Measurement | Result |
|---|---:|
| Mean weak evaluations per face | **158.4** |
| Mean weak evaluations per non-face | **27.5** |
| Total weak classifiers | **160** |
| Zero-coefficient placeholder classifiers | **0** |
| Stages | **4** |
| Serialized trained model | **4,922 bytes (4.81 KiB)** |

The weak-evaluation means are the trusted harness's class-specific aggregates
on validation. Classifier count was checked against `--describe` and the model.
The model is a compact text representation; file-size comparisons also depend
on serialization format and unused capacity, not only classifier count.

## Corrected timing and peak memory

| Operation | Wall-clock seconds | CPU seconds, user + system | Peak RSS |
|---|---:|---:|---:|
| Final training and calibration | **29.668** | **29.664** | **159,984 KiB (156.23 MiB)** |
| Predict all 1,046 validation images | **0.173** | **0.022** | **2,288 KiB (2.23 MiB)** |

Full-batch prediction includes startup, model loading, image I/O, classification,
and score output. The separate startup-subtracted, minimum-of-three estimate
was **3.6 microseconds CPU per image**, with 3.29 ms startup/model loading.
Do not substitute this per-image estimate for the measured full-batch totals.

The WSL harness uses Bubblewrap `--as-pid-1` for correct descendant CPU
accounting. GNU time measures per-command peak resident set size outside the
learner's mount. Peak RSS is resident process memory, not model-file size,
virtual address space, or the memory of the AI agent generating the code.
The CPU-workload and 50-MiB-memory regression tests and all eleven harness
tests passed before this run.

## Data use and limitations

| Partition | Faces | Non-faces | Original images |
|---|---:|---:|---:|
| Construction weight fitting | 1,249 | 2,310 | 3,559 |
| Independent calibration | 399 | 787 | 1,186 |
| Selection holdout | 417 | 769 | 1,186 |
| Final weight fitting: fitting + selection | 1,666 | 3,079 | 4,745 |

Final training reads the 5,931 supplied originals but reserves 1,186 solely
for calibration. Augmentation produces 21,806 fitting examples. Calibration
examples never fit stump weights. Original IDs were split before augmentation.
The final fit is deliberately different from fitting weights on all originals.

All candidates hit the 160-classifier budget; some stage pass targets were
unmet. None met the cumulative 0.5% calibration-negative stopping target.
The final fit passed 12/787 calibration negatives. The 99% selection-detection
preference was also unmet. The budgets and targets were not relaxed in response
to these outcomes or hidden-validation feedback. These are results of the
precommitted experiment, not a claim of improvement on every run-01 metric.

## Artifacts and verification

- [learner/](learner/): source, Makefile, final.model, configuration, and manifests.
- [learner/search_log.json](learner/search_log.json): three constructions and 27 settings.
- [protocol_audit.json](protocol_audit.json): exact feature-pool and freeze checks.
- [results.json](results.json): machine-readable full-precision measurements.
- [runs.jsonl](runs.jsonl): original trusted validation/evaluator output.
- [learner/RESULTS.md](learner/RESULTS.md): the agent's own report, based on rounded feedback.

Independent host checks verified that all 6,500 feature definitions match the
fixed baseline and that the plan/configuration hashes, source, and delivered
model match the frozen validated build. The three-construction limit was met.
