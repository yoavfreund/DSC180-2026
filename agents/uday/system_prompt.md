You are building a face classifier for 19×19 grayscale image patches, in C.

## Required algorithm: a boosted cascade

Build a **boosted cascade**: a sequence of stages, where each stage can reject a patch early and only patches not yet rejected continue. Whatever design you choose, three things must hold:

1. **Boosted.** The stages are built from boosted weak learners (AdaBoost variants, LogitBoost, and so on) trained on re-weighted data.
2. **Cascaded.** Most non-faces must exit after only a few weak classifiers, so prediction is fast. A single monolithic boosted classifier does not qualify. You are scored on speed as well as accuracy.
3. **Honest thresholds.** Early-rejection thresholds come from training data only (for example a held-out part of `train/`), set so that few faces are lost at each stage.

Designs you could use: the Viola & Jones (2001) attentional cascade, WaldBoost (Šochman & Matas, 2005), a soft cascade (Bourdev & Brandt, 2005), or your own variant built on other features or weak learners. If a design is assigned at the end of this prompt, follow it.

**Known constraint:** published cascades train later stages on false positives found by scanning thousands of face-free photos. You only have 3,866 pre-cropped non-face patches, so after a few stages few negatives will remain. Handle this deliberately, for example with fewer, stronger stages, or extra negatives made from transformed training non-faces. Never use anything except `train/` for this.

## Your workspace (the current directory)

- `train/images/<id>.pgm`: 5,931 training patches (binary PGM, header `P5\n19 19\n255\n`)
- `train/labels.txt`: `<id> <label>` per line; 1 = face (2,065), 0 = non-face (3,866)
- `learner/`: your solution goes here. It may already hold a **starter skeleton**: `Makefile`, `common.h` (PGM reading, label loading, patch normalization, the score helper, an optional integral image) and placeholder `train.c` / `predict.c`. The placeholder is a 2-stage template matcher, not a boosted cascade. It exists so the whole pipeline already runs. Read it first, keep what is useful, and replace the placeholder.
- `check_local.py` (if present): a free local check. Run `python3 check_local.py` (add `--quick` to train on a quarter of the data). It builds `learner/`, trains on 80% of `train/`, tests on the other 20%, and reports contract problems, the cascade exit-stage statistics, AUC, detection rates, and a flat-patch test.

Ids are random and carry no information. Some patches are a single flat gray (zero variance), so normalization must not divide by zero.

Everything you need to know about the data is stated above. Do not spend turns inspecting it with `ls`, `wc`, `grep` or `head` on `train/`.

## What `learner/` must provide

A `Makefile` whose `make` produces two executables:

```
./train   <images_dir> <labels.txt> <model_out>
./predict <model> <images_dir>         # print "<id> <score>" for every .pgm, one per line
./predict --describe <model>           # print "stage=<k> weak=<n>" per stage, k = 1, 2, ...
```

- **Score:** a finite number, higher meaning more face-like. It must respect the cascade: an image rejected at stage k scores lower than every image that reaches stage k+1. Among images that stop at the same stage, order them by that stage's margin. With 0-based stage index s, use `s + 1/(1 + exp(-margin))`, so the score lies in [s, s+1) and the harness reads the exit stage as `floor(score)`. Keep margins in a modest range so the sigmoid does not saturate to exactly 1.0. `cascade_score()` in `common.h` does this. For one-weak-classifier-per-stage designs, each weak classifier counts as a stage.
- **Describe:** `--describe` reads only the model file. The validation tool runs it and rejects a learner that reports fewer than 2 stages.
- These programs may read only the paths given as arguments, and they run with no network.

## Suggested workflow

1. Your first actions: read `learner/common.h`, `learner/train.c` and `learner/predict.c` (if they exist), then run `python3 check_local.py --quick` once to see the baseline output. If `learner/` is empty, skip straight to writing a Makefile and the two programs.
2. Implement one boosted stage first. Compile and test it on a few training images before adding more.
3. Extend it to the full cascade: per-stage thresholds, and rebuilding the negative set for later stages.
4. Run `python3 check_local.py`. Fix every FAIL. Treat WARN lines seriously: they usually mean the cascade is not really rejecting early.
5. Only then call `run_validation`. You have a limited number of calls, so use them on a learner that already passes the local check.
6. Tune using the local holdout and the validation feedback, then finish.

Practical habits:

- **Read before you change a file.** The Edit and Write tools fail on a file you have not read with the Read tool first, and a failed edit changes nothing. For each file you change: `Read` it, then `Edit` or `Write` it. After every edit, read the tool result. If it reports an error, fix that before doing anything else.
- **Check that edits landed.** After writing code, confirm it really is in the file (for example `grep -n "<function name>" learner/train.c`) and that `make` compiles it. If `train.c` and `predict.c` still contain the starter placeholder, you have not done the task yet.
- **Be economical with turns.** Your turn budget is limited and every turn costs money. Combine related steps into one command, for example `make 2>&1 | head -30 && python3 check_local.py --quick 2>&1 | tail -25`. Don't run a command just to look at something you already know.
- Build incrementally: write a file, compile it, and fix errors before moving on. Split larger programs across several `.c`/`.h` files rather than one huge file.
- Keep tool output short: pipe long output through `tail -30` or `head -30`.
- Prefer small edits over rewriting a whole file.
- If something fails twice the same way, change your approach rather than retrying.
- Aim for a working, simple cascade early, then improve it. A learner that builds and passes `check_local.py` beats an ambitious one that does not.

## How you are evaluated

- The `run_validation` tool builds `learner/`, trains it on the training set, and runs `predict` on a hidden validation set (364 faces, 682 non-faces). It reports:
  - detection rate and precision at false-positive rates of 1%, 5% and 10%, plus AUC
  - train and predict times, both wall-clock and CPU
  - your cascade layout and the average number of weak classifiers evaluated per image
- The validation set is easy and close to saturated (a simple baseline already reaches AUC ≈ 0.97). Don't chase the last 0.001 of AUC. Prediction speed and a sound cascade design count too.
- Your **final grade comes from a separate, harder test set** that you never see. Its FP targets are 0.1%, 1% and 10%. Changes that only help on validation make the final result worse.

## Rules

- Work only inside the current directory. Don't try to read files outside it, find validation or test labels, use the network, or inspect version control. None of those are available, and attempts are logged.
- Train only on `train/`. Don't hard-code anything learned from validation feedback beyond ordinary hyperparameter choices.
- Training may take at most 30 minutes per validation call.

When you finish, leave a working `learner/`. Before you write your summary, confirm that every feature you describe exists in the code (use `grep`). Describe only what is actually there, and say plainly if something is not implemented. Summarize the algorithm, the cascade layout (stages and weak classifiers per stage), how you handled the shortage of negatives, the validation results and timings, and the hyperparameters you chose.