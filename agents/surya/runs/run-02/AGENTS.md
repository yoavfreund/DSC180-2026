You are building run-02: a lightweight 19x19 grayscale face-patch classifier in C.
Use DISCRETE AdaBoost decision stumps on the existing Haar feature definition.
Change cascade stage construction and threshold calibration only. Do not use
Real AdaBoost, WaldBoost, neural networks, external data, or another run's code.

## Data, access and interface

Your current directory is already learner/. Write Makefile and source here.
../train/images contains 5,931 canonical binary 19x19 PGM patches;
../train/labels.txt has '<id> <label>', with face=1 and non-face=0.
There are 2,065 original faces and 3,866 original non-faces. Some images have
zero variance. All scores must be finite. Use only this run's supplied data.
Other runs, private labels, the repository and host credentials are excluded
by the native Linux file policy. Tool network, browsing, plugins, apps and
subagents are disabled. Never attempt to circumvent these boundaries.

make must produce:
    ./train <images_dir> <labels.txt> <model_out>
    ./predict <model> <images_dir>
    ./predict --describe <model>

Predict exactly one '<id> <score>' per .pgm. For a patch stopping at one-based
stage k, score in [k-1,k), ordered by stage margin. Use (k-1)+sigmoid(margin),
clamped away from interval boundaries. --describe reads only the model and
prints stage=<k> weak=<n>, consecutive k from 1, positive weak counts, at least
two stages. Stop negative predictions early; compute real evaluated counts.

## Fixed feature and boosting baseline: do not change

Use 6,500 reproducibly sampled Haar features: horizontal and vertical
two-rectangle, horizontal and vertical three-rectangle, and checkerboard.
Use integral images. Normalize contrast by rectangle area and
sqrt(max(1, E[pixel^2]-E[pixel]^2)). Quantize as int(64 + contrast*24),
clipped to [0,127], using 128 bins and C-style integer truncation.

Exact data-independent candidate-generation recipe:
32-bit unsigned xorshift RNG initialized to 1234567; each draw performs
x ^= x<<13; x ^= x>>17; x ^= x<<5 (unsigned 32-bit overflow at each step).
For each of 6,500 features draw, in this order:
    t = draw()%5
    dx = 2 if t is 0 or 4, else 3 if t is 2, else 1
    dy = 2 if t is 1 or 4, else 3 if t is 3, else 1
    w = dx*(1 + draw()%(19/dx))    # integer division
    h = dy*(1 + draw()%(19/dy))
    x = draw()%(20-w)
    y = draw()%(20-h)
Feature responses before area/variance normalization:
    t=0: whole - 2*left_half
    t=1: whole - 2*top_half
    t=2: whole - 3*middle_horizontal_third
    t=3: whole - 3*middle_vertical_third
    t=4: whole - 2*(top_left_quarter + bottom_right_quarter)
Save feature_pool.tsv as 't x y w h' rows for verification, but train generates
the same features itself and does not depend on an external feature file.

Each stump uses one feature, threshold bin and polarity selected by minimum
weighted classification error. Initialize each stage with half the weight on
each class. Use alpha=min(2,0.5*log((1-error)/error)), with finite error clamps,
and the ordinary exponential AdaBoost weight update. Keep augmentation fixed:
faces = original plus horizontal flip; non-faces = original, horizontal flip,
and four one-pixel shifts with edge replication. Split ORIGINAL images before
augmentation; a transformed image must stay with its originating image.

## Precommitted experiment: maximum THREE complete candidate constructions

Write experiment_plan.json BEFORE training or calling hidden validation.
The following settings and selection rules are fixed now, before any hidden
validation feedback. Do not add candidates, increase budgets, or change targets
after seeing results. Do not spend tokens on a broad architecture search.

Make a deterministic three-way split of original supplied training images:
shuffle labels in their input order with Python random.Random(7321); every
fifth row is the 20% selection holdout. From the remaining rows, shuffled with
random.Random(2026), every fourth row is the 20% calibration subset. The rest
is the 60% fitting subset. Save the original-ID split manifests.
Fit stump weights on fitting examples only. Use calibration data for thresholds
and stopping decisions. Use selection data only to rank completed candidates.

Construct AT MOST three candidate cascades from scratch. The only candidate
difference is the conditional per-stage non-face pass target:
    candidate A: 0.50
    candidate B: 0.35
    candidate C: 0.20
For each candidate, train on fitting faces and fitting negatives surviving
earlier stages. Use calibration faces reaching that stage to set its threshold
for at least 0.998 conditional face retention (zero margin offset initially).
Grow a stage until that retention and its candidate non-face pass target are
met on arriving calibration examples. The learned number of weak classifiers
is therefore decided by training/calibration data, not a fixed layer list.
Minimum 2 weak classifiers per stage; maximum 64 per stage; maximum 8 stages;
maximum 160 classifiers total. Require at least 2 stages. Stop constructing
when cumulative calibration non-face pass fraction <=0.005, or a fixed budget
is reached. Report unmet targets honestly; never silently relax a target or
add another construction. Handle zero surviving negatives explicitly.

With each candidate's trained stumps FIXED, sweep only these nine threshold
hyperparameter pairs; do not refit stumps or rebuild architecture for a sweep:
    conditional calibration-face retention: 0.995, 0.998, 0.999
    subtractive margin offset: 0.0, 0.25, 0.50
Calibrate thresholds sequentially from arriving calibration faces, using the
largest threshold satisfying the chosen empirical retention, minus the offset.
Explain discrete-sample resolution and empty-arrival cases. Recompute exits,
recall and evaluated weak counts after each threshold setting.

Rank candidates/settings on the selection holdout using a correct tied-score
ROC and integer FP budgets. Prefer settings with detection >=0.99 at <=1% FP;
among those minimize average weak evaluations per non-face, then prefer higher
detection, then fewer total classifiers. If none meets 0.99, choose highest
detection at <=1% FP, then lowest average non-face weak evaluations, then fewer
classifiers. Break remaining ties in A/B/C order and grid order above.

Record every construction and threshold sweep in search_log.json, including
chosen stage sizes, targets met, stage exits, selection metrics, and the final
choice. No more than three complete candidate constructions or revisions.
A final fit and the harness's mandatory fit may retrain the FROZEN stage sizes
and chosen calibration rule; these are not new architecture searches and must
not change the architecture or selected hyperparameters.

Freeze selected_config.json before hidden validation. ./train must read this
small shipped configuration, never the search models or prior trained weights.
For its final fit, train the fixed stage sizes on all non-calibration examples
(fitting plus selection; about 80%), retaining the independent calibration 20%
for thresholds. Do not train stump weights on calibration examples. State the
exact fitting/calibration counts: the final fit intentionally reserves data
for honest calibration instead of fitting weights on all 5,931 images.
Training must be deterministic and complete within 30 minutes.

## Validation and deliverables

Only AFTER selection is frozen, call the validation MCP run_validation tool
ONCE. It uses the repository's shared eval/eval.c on the hidden staged validate
folder. Do not tune architecture, thresholds or any other hyperparameters after
seeing this feedback. A second call is allowed only to fix an interface/runtime
failure, never to optimize measured accuracy. Keep logs of any failed call.

The trusted harness reports detection and precision at <=1%, <=5%, <=10% FP,
AUC, mean weak evaluations per face/non-face, total classifiers, model bytes,
train/predict wall and CPU times, and train/predict peak resident memory.
Use actual measured values, not estimates. Record unavailable values explicitly.

Leave working source, Makefile, selected_config.json, experiment_plan.json,
search_log.json, split manifests, README.md, and your trained final model.
Keep learner/ below the harness's 5 MB source-snapshot limit: do not leave large
feature caches, temporary candidate executables or large prediction dumps.
Do not rerun complete searches after the three-construction limit.

Write learner/RESULTS.md covering:
* algorithm, learned stage layout and thresholds, and construction count <=3;
* precommitted targets, threshold hyperparameters tested, and selection rule;
* internal selection metrics, clearly separate from hidden validation;
* hidden detection at 1%, 5%, 10% FP and precision at 1% (all precisions welcome);
* mean weak evaluations per face and non-face, total classifiers, model bytes;
* peak RSS for training and prediction, and corrected wall/CPU times for each;
* actual data-partition sizes, validation-call count, and any unmet targets.
Do not read any other run, including run-01. Work only in this learner directory
and on its supplied training data. Summarize the final results concisely.

## Protected Linux commands

You run natively inside Ubuntu WSL, with learner/ as your current directory. Compile with `make`; training data is at `../train/`.
Use the validation MCP server's run_validation tool for hidden validation.
Your total validation budget for this run is 2 calls.
