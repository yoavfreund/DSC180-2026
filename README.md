# LLM-assisted image boosting

This prototype trains an ensemble of concrete C classifiers to distinguish
face and non-face image patches. A Python coordinator asks an interchangeable
LLM adapter for candidate model specifications, evaluates the trained models,
and updates one weight per original image after each accepted round.

For Docker setup, ChatGPT sign-in, commands, and configuration, see
[BOOSTING.md](BOOSTING.md). For a different LLM provider, see
[LLM_ADAPTER.md](LLM_ADAPTER.md). Outstanding work is listed in
[TODO.txt](TODO.txt).

## Pipeline, step by step

1. **Load the images.** `main.py` reads 19 × 19, 8-bit PGM files from
   `train/face`, `train/non-face`, `validate/face`, and `validate/non-face`.
   It combines the two folders into a development pool without moving files.
   Face labels are `+1` and non-face labels are `-1`. The original file order
   is recorded in the run's `dataset.json`.

2. **Prepare features and variants.** `common/images.py` calls the C functions
   in `learner/image_features.c` to make local binary pattern (LBP) maps and
   integral images. Each row has 361 normalized pixels, 289 interior LBP
   values, and 64 normalized rectangle means. By default, two additional views
   are prepared: a +5° rotation and a −5° rotation with a one-pixel horizontal
   shift. Transformations use nearest-neighbor sampling and edge padding. Pass
   `--no-augmentation` to train on originals only.

3. **Initialize weights.** The coordinator assigns every original image a
   uniform weight. This single array persists across boosting rounds and is
   normalized to sum to one after each accepted learner.

4. **Make a fresh split.** For round `t`, a seeded permutation with seed
   `42 + t` assigns approximately 33% of original images to training and 67%
   to validation by default. The split is not stratified. Each original and
   its variants remain on the same side. For training, the original's weight
   is divided equally among its original and transformed views.

5. **Propose candidate learners.** The coordinator sends the adapter recent
   accepted specifications and a bounded summary of aggregate audit records.
   The adapter returns a JSON list of model and feature choices. Supported
   models are decision stumps, bounded-depth trees, and logistic regression;
   feature choices are pixels, LBP, and rectangle means. The coordinator
   validates the response against `coordinator/specs.py`. This is a bounded
   specification search; the LLM does not generate new learner code.

   If the adapter fails, times out, or returns invalid output, the coordinator
   uses fixed fallback specifications. With the default three candidates,
   these are stumps using pixels, LBP, and rectangle features. If a later
   attempt in the same round would use fallback again, refinement stops
   without retraining those fallback candidates.

6. **Train and evaluate candidates.** Separate worker processes train the C
   learners on the training images and their variants. For each successful
   learner, the coordinator predicts all originals. `learner/evaluate.c`
   computes weighted error using the original validation images only:

   ```text
   validation error = sum(weight of misclassified validation images)
                      / sum(weight of all validation images)
   ```

   The log also records mean weighted error on the two transformed validation
   views as a robustness check. It does not affect selection or learner weight.

7. **Choose or refine.** Candidates with validation error below 0.5 are
   eligible. Within an attempt, candidates within 0.005 of the lowest error
   compete on median per-image time after a warm-up and three measurements of
   feature extraction and prediction on the same batch. If timing is incomplete
   for any contender, the lowest error wins. Across attempts, the coordinator
   keeps the selected candidate with the lowest error. By default it stops
   refining at error ≤ 0.49, or after three attempts. An eligible candidate
   below 0.5 may still be accepted if refinement ends without reaching 0.49.
   If no candidate is below 0.5 and attempts and time remain, the coordinator
   requests another proposal. It skips configurations already tried in that
   round; repeated proposals still consume an attempt. If no candidate is
   eligible when refinement ends, the round is skipped and weights are unchanged.

8. **Update the distribution.** For the selected learner, its validation error
   `e` is clamped to `[1e-6, 1 - 1e-6]` and its ensemble weight is
   `alpha = 0.5 * log((1 - e) / e)`. Using that learner's predictions on all
   original images, the coordinator multiplies each weight by
   `exp(-alpha * label * prediction)` and normalizes the array. It saves the
   model, alpha, and updated weights before starting the next round.

9. **Use the saved ensemble.** The run's `manifest.jsonl` contains accepted
   learner specifications, model paths, model hashes, validation errors, and
   alphas. `predict_ensemble.py` verifies the saved model hashes and combines
   their hard predictions using the alpha-weighted vote. It expects the same
   labeled image-folder layout and writes one path and hard label per line.

The default configuration runs 10 rounds, proposes three candidates per
attempt, and uses a 55-second round deadline. The deadline bounds proposal and
training workers; preprocessing, coordinator-side prediction, timing, and
scoring are outside that bound. See `config.py` and `main.py --help` for the
available settings.

## Audit and outputs

Each run needs a new output directory. It contains `audit.jsonl` with decisions,
requests, statuses, validation errors, robustness checks, selections, and
weight-update records. Each `round_XXX/` contains candidate model files and
the input and, for accepted rounds, output weight arrays. `manifest.jsonl`
contains only accepted learners. The `runs/` directory is ignored by Git.

The ChatGPT plan adapter stores credentials and its local 50-attempt-per-five-
hour limit in a Docker-managed volume outside the repository. Failed calls count
toward that cap. Plan eligibility and limits are determined by OpenAI; API-key
access is separately billed. Never commit credentials.

## Interpretation and limitations

The changing validation subsets select individual learners; they are not an
independent evaluation of the finished ensemble. The original `train/` and
`validate/` folders are combined by this prototype, so a separate untouched
test dataset is needed to estimate generalization. Using validation-subset
error for alpha while updating all images is an experimental variation and
does not establish the standard AdaBoost training-distribution guarantee.

The current output is an alpha-weighted ensemble, not a staged cascade. It
does not automatically evaluate the finished ensemble on an independent test
set. The LLM can choose only among the implemented models and features.
