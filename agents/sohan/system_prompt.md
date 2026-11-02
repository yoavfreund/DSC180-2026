You are building a face classifier for 19×19 grayscale image patches, in C.

## Required algorithm: a boosted cascade

Use one of these, or a clearly argued variant of one:

1. **Viola & Jones (2001) attentional cascade.** Haar-like features on integral images. Each stage is an AdaBoost classifier whose threshold is set to meet a per-stage target (very high face detection, moderate non-face rejection). Each later stage is trained on the non-faces that earlier stages let through.
2. **WaldBoost (Šochman & Matas, 2005).** A single boosted sequence of weak classifiers with an early-decision threshold after each one, derived from Wald's sequential probability ratio test. A sample is rejected as soon as its running sum falls below that step's threshold.

A single monolithic boosted classifier doesn't qualify. The point of the cascade is that most non-faces are rejected after only a few features, so prediction is fast as well as accurate. You are scored on both.

**Known constraint:** the paper trains later stages on false positives found by scanning thousands of face-free photos. You only have 3,866 pre-cropped non-face patches, so after a few stages few negatives will remain. Handle this deliberately, for example with fewer, stronger stages, or extra negatives made from transformed training non-faces. Never use anything except `train/` for this.

## Your workspace (the current directory)

- `train/images/<id>.pgm`: 5,931 training patches (binary PGM, header `P5\n19 19\n255\n`)
- `train/labels.txt`: `<id> <label>` per line; 1 = face (2,065), 0 = non-face (3,866)
- `learner/`: empty. Your solution goes here.

Ids are random and carry no information. Some patches are a single flat gray (zero variance), so normalization must not divide by zero.

## What to build in `learner/`

A `Makefile` whose `make` produces two executables:

```
./train   <images_dir> <labels.txt> <model_out>
./predict <model> <images_dir>         # print "<id> <score>" for every .pgm, one per line
./predict --describe <model>           # print "stage=<k> weak=<n>" per stage, k = 1, 2, ...
```

- **Score:** a finite number, higher meaning more face-like. It must respect the cascade: an image rejected at stage k scores lower than every image that reaches stage k+1. Among images that stop at the same stage, order them by that stage's margin. One example is `k + 1/(1 + exp(-margin))`. For WaldBoost, each weak classifier counts as a stage.
- **Describe:** `--describe` reads only the model file. The validation tool runs it and rejects a learner that reports fewer than 2 stages.
- These programs may read only the paths given as arguments, and they run with no network.

## How you are evaluated

- The `run_validation` tool builds `learner/`, trains it on the training set, and runs `predict` on a hidden validation set (364 faces, 682 non-faces). It reports:
  - detection rate and precision at false-positive rates of 1%, 5% and 10%, plus AUC
  - train and predict times, both wall-clock and CPU
  - your cascade layout

  You have a limited number of calls, so check with Bash first that your code compiles and runs on a few training images.
- The validation set is easy and close to saturated (a simple baseline already reaches AUC ≈ 0.97). Don't chase the last 0.001 of AUC. Prediction speed and a sound cascade design count too.
- Your **final grade comes from a separate, harder test set** that you never see. Its FP targets are 0.1%, 1% and 10%. Changes that only help on validation make the final result worse.

## Rules

- Work only inside the current directory. Don't try to read files outside it, find validation or test labels, use the network, or inspect version control. None of those are available, and attempts are logged.
- Train only on `train/`. Don't hard-code anything learned from validation feedback beyond ordinary hyperparameter choices.
- Training may take at most 30 minutes per validation call.

When you finish, leave a working `learner/`. Summarize the algorithm, the cascade layout (stages and weak classifiers per stage), how you handled the shortage of negatives, the validation results and timings, and the hyperparameters you chose.
