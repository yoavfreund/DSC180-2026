# Final results: CBCL test set

> **Old data split.** These Nov 2 results use the original CBCL train/test download, before the
> pooled re-split of Nov 24 (`make_split.py`), so they can't be compared with newer numbers. To
> reproduce them, check out the `archive/harness-validation` tag. Current results:
> [`results/RESULTS.md`](results/RESULTS.md).

Scored 2026-11-02 on one Mac. Three solutions were entered, each person's run-02.
Andrew's learner detected the most faces at every false-positive (FP) target, but
it also did the most work per image. Surya's and Sohan's learners weren't clearly
separated on any accuracy metric.

## Results

Test set: 472 faces, 23,573 non-faces. Evaluator: `eval/eval.c` with
`--fp-rates 0.01,0.05,0.1`.

| | Andrew | Surya | Sohan |
|---|---:|---:|---:|
| **Detection at ≤1% FP** (235 allowed) | **0.610** | 0.396 | 0.364 |
| **Detection at ≤5% FP** (1,178 allowed) | **0.714** | 0.583 | 0.572 |
| **Detection at ≤10% FP** (2,357 allowed) | **0.750** | 0.693 | 0.710 |
| Precision at ≤1% / 5% / 10% FP | 0.551 / 0.224 / 0.131 | 0.447 / 0.195 / 0.122 | 0.425 / 0.187 / 0.145 |
| AUC | **0.897** | 0.866 | 0.874 |
| Weak classifiers per non-face | 51.3 | 26.4 | **9.8** |
| Weak classifiers per face | 252.2 | 101.8 | **48.5** |
| Weak classifiers in total | 326 | 160 | **83** |
| Model size | 20,734 bytes | **4,926 bytes** | 5,187 bytes |

Precision is low because only 2% of the test set is faces. It can't be compared
with validate, which is 35% faces.

## Operating points

The evaluator reports the point with the most faces whose false positives stay
within the allowance, so a learner can land below the allowance.

| FP target | Andrew: faces, FP | Surya: faces, FP | Sohan: faces, FP |
|---|---:|---:|---:|
| ≤1% | 288, 235 | 187, 231 | 172, 233 |
| ≤5% | 337, 1,169 | 275, 1,137 | 270, 1,176 |
| ≤10% | 354, 2,353 | 327, 2,351 | 335, 1,978 |

## How sure these differences are

95% intervals from 2,000 paired bootstrap resamples of the test images. They
cover sampling of the test set only, and aren't adjusted for multiple comparisons.

| Metric | Andrew | Surya | Sohan |
|---|---:|---:|---:|
| Detection at ≤1% FP | 0.564–0.650 | 0.347–0.445 | 0.324–0.415 |
| Detection at ≤5% FP | 0.672–0.758 | 0.540–0.627 | 0.528–0.614 |
| Detection at ≤10% FP | 0.710–0.790 | 0.648–0.731 | 0.669–0.750 |
| AUC | 0.877–0.915 | 0.847–0.884 | 0.856–0.891 |

| Difference | AUC | ≤1% FP | ≤5% FP | ≤10% FP |
|---|---:|---:|---:|---:|
| Andrew − Surya | **+0.031** | **+0.214** | **+0.131** | **+0.057** |
| Andrew − Sohan | **+0.023** | **+0.246** | **+0.142** | +0.040 |
| Surya − Sohan | −0.008 | +0.032 | +0.011 | −0.017 |

Bold means the interval excludes zero. Andrew − Sohan at ≤10% FP is borderline:
its interval starts at +0.000.

## Validate overstated every learner

| | Andrew | Surya | Sohan |
|---|---:|---:|---:|
| AUC, validate to test | 0.998 to 0.897 | 0.997 to 0.866 | 0.990 to 0.874 |
| Detection at ≤1% FP, validate to test | 0.995 to 0.610 | 0.986 to 0.396 | 0.951 to 0.364 |
| Detection at ≤5% FP, validate to test | 0.995 to 0.714 | 0.989 to 0.583 | 0.970 to 0.572 |
| Detection at ≤10% FP, validate to test | 0.995 to 0.750 | 0.989 to 0.693 | 0.975 to 0.710 |
| **Faces reaching the final stage, validate to test** | **99.5% to 74.6%** | **98.6% to 49.4%** | **95.6% to 38.8%** |
| Non-faces reaching the final stage, test | 9.9% | 2.8% | 1.1% |

The early stages cause much of the drop. A face rejected early scores below every
image that gets further, so it can only be recovered by accepting those images too.
Andrew's learner loses 120 of 472 test faces before its final stage, and its
detection barely rises past that limit even at ≤10% FP (354 faces, against 352
reaching the final stage).

The most likely reason is a change of source: every test file is named `cmu_*`,
unlike train and validate. We didn't test this directly.

## Speed

Measured on the same Mac in the same sandbox, with learners run one at a time.

| | Andrew | Surya | Sohan |
|---|---:|---:|---:|
| Train, wall | 25.3 s | 34.9 s | 38.5 s |
| Train, CPU | 111.1 s | 34.2 s | 37.7 s |
| Predict CPU per image, excluding start-up | 15.5 µs | 14.3 µs | 12.0 µs |

- **Weak classifiers evaluated is the fairer speed metric.** It doesn't depend on
  the machine. On test, Andrew's learner uses about 1.9× Surya's and 5.2× Sohan's
  per non-face.
- **Predict CPU differences are within noise.** An earlier pass on the same
  machine gave 14.8, 15.3 and 13.7 µs, a different order. On pre-cropped patches,
  reading files likely outweighs classifier work.
- **Train times come from one run each.** An earlier pass gave 28.2, 35.3 and 40.1 s
  wall. Andrew's learner uses every CPU core, which is why its CPU time is higher.

## What this comparison does and doesn't show

It compares three frozen learners. It doesn't show which agent is better:

| | Andrew | Surya | Sohan |
|---|---|---|---|
| Agent | Claude Agent SDK | Codex CLI | Claude Agent SDK, same code as Andrew's |
| Model | Not recorded | Not recorded | `claude-sonnet-5-5` |
| System prompt | 53 lines | 156 lines, prescribing much of the design | Same as Andrew's |
| Algorithm | Real AdaBoost, 5 stages | Discrete AdaBoost, 4 stages | Discrete AdaBoost, 7 stages |
| Size limit | Chosen by the agent | 160 weak classifiers, set by the prompt | Chosen by the agent |

Each solution is one run. Andrew's and Sohan's agents share code and a prompt but
produced different learners, which shows that one run can't characterize an agent.

## How it was scored

1. **Freeze:** each learner's source hash was recorded at commit `4b95a6a`.
2. **Reproduce:** each learner was rebuilt and retrained from that source, and had
   to match its logged validate results exactly. All three did. The evaluator's own
   test suite also passed.
3. **Stage:** the test set was staged after the freeze with a new random seed,
   `715485928`. Counts, pixels and labels were checked against the originals. The
   sandbox could read the images but not the answer key.
4. **Predict:** each learner ran `predict` once on the test images, under
   `harness/learner.sb`. The script refuses a second run.
5. **Evaluate:** `eval/eval.c` produced every detection, FP, precision and AUC figure
   above. Weak-classifier counts use the harness's logic, and the intervals and
   final-stage rates come from `pipeline/test_day_analysis.py`. That script's point
   estimates match the evaluator's.

## Disclosures

- **The FP targets changed after the first scoring.** The first scoring used 0.1%,
  1% and 10%, from `pipeline/SPEC.md`. The team saw those results, then changed the
  targets to 1%, 5% and 10% to match the assignment instructions. `SPEC.md` had said
  not to change targets after seeing results, so this is recorded here. `predict` wasn't
  rerun: the saved scores reproduced the first evaluator output exactly, then were
  evaluated at the new targets. The 1% and 10% results didn't change, and the first
  output is kept in `~/dsc180-private/test-day/scored/`.
- **Two agents were told the old targets.** Line 45 of Andrew's and Sohan's system
  prompts said the test targets were 0.1%, 1% and 10%. Andrew's agent included 0.1%
  in its own checks on training data, and its stated reasons for its thresholds cite
  AUC and early-rejected faces. Sohan's agent tried a change that it said might help
  at 0.1%, but the edit failed, so the frozen learner doesn't include it. Influence
  can't be ruled out. Surya's prompt didn't mention 0.1%. All three agents' validation
  feedback used 1%, 5% and 10%.
- **The models scored are the ones retrained in step 2.** Surya's differs from the
  delivered `final.model` by 4 bytes, but gave identical validate results.
- **The first reproduction check stopped on a false alarm.** It flagged Sohan's
  scores because images passing all 7 stages score in `[7, 8)`. Those scores still
  order correctly and the weak-classifier count is right, so the check was widened
  to allow that band and everything was retrained. This happened before staging, so
  no learner had seen the test set. `harness/run_validation_WIN.py` would reject this
  score range, so the two harnesses should be reconciled.
- **The test data came from the team's copy of CBCL.** Its counts match the official
  set and it shares no images with train or validate, but it wasn't compared with a
  fresh download.
- **No fourth solution was entered.** `agents/` holds only these three.

**This test set is now spent.** Any learner changed after these results needs a
new held-out set for an honest score.

## Reproducing

```bash
python3 pipeline/test_day.py gate       # refused once scoring has started
python3 pipeline/test_day.py stage      # refused if the test set is already staged
python3 pipeline/test_day.py score      # refused for any learner already scored
python3 pipeline/test_day.py rescore    # applies the 1%, 5%, 10% targets to saved scores
python3 pipeline/test_day_analysis.py   # recomputes the intervals and final-stage rates
```

Per-image scores, the answer key and the logs are in `~/dsc180-private/test-day/`
and `~/dsc180-private/staged/test/`, outside the repo.
