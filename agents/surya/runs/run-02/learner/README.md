# Run 02: discrete AdaBoost Haar cascade

Build with `make`, then run:

```
./train ../train/images ../train/labels.txt final.model
./predict final.model ../train/images
./predict --describe final.model
```

Training reads `selected_config.json`, `calibration.txt`, and `selection.txt` from the working directory. Run from learner/. It regenerates all 6,500 features, initializes new stage weights, and never loads a search model. The shipped split manifests contain original IDs and labels. Final stump fitting uses fitting plus selection originals, with fixed augmentation; calibration originals remain independent and unaugmented. Selection evaluation also uses unaugmented originals.

The feature definitions, xorshift sampling, normalization, quantization, discrete stumps, alpha cap, and exponential weight update follow the supplied baseline. Prediction builds one integral image and computes features lazily only for evaluated stumps, reusing feature responses within each patch. Standard deviation has a floor of one. Scores are stage index plus sigmoid of the stage margin, clamped to [1e-9,1-1e-9] within that stage's interval. Stderr reports actual evaluated weak counts; stdout contains only ID/score pairs. Describe reads only the model.

`experiment_plan.json` was written before training. `prepare.py` records the split convention: one-based every fifth shuffled row, then one-based every fourth remaining shuffled row. `search.py` performed exactly three constructions, A/B/C, then the nine precommitted threshold settings on each fixed model. Do not rerun it: the construction budget has been consumed. `search_log.json` preserves all outcomes, including failed stage targets and the final choice. Candidate weights are not needed for final training.

A stage threshold is sorted arriving calibration-face score at zero-based index floor((1-retention)*n), minus the chosen offset. Passing uses score >= threshold, so ties retain all examples at the threshold. This is the largest empirical threshold with the requested retention. At 399 arriving calibration faces, 0.998 and 0.999 permit zero losses; 0.995 permits one. Resolution changes with arrival counts. Empty calibration-face arrivals use finite -1e30 to retain all arrivals. Empty calibration-negative arrivals satisfy the pass target. With no fitting negatives, required remaining stages use zero-alpha stumps explicitly, rather than undefined class weights. A new stage is never started with fewer than two classifiers left in the total budget.

Stages fit all fitting faces and surviving fitting negatives with half of initial weight on each class. Construction uses 0.998 retention and zero offset, at least two stumps per stage, at most 64 per stage, eight stages, and 160 total; at least two stages are required. A budget can leave a stage target or cumulative target unmet. Sweeps change only thresholds. Selection ROC groups exactly tied output scores and accepts whole groups within floor(rate*Nnegative) false positives. The ranking rule is recorded verbatim in the plan.

Model format is plain text: magic/version and stage count; for each stage, weak count and threshold followed by feature index, threshold bin, polarity, alpha for each stump. No feature cache is stored. See RESULTS.md for measured results and limitations.
