The existing implementation is working and validated. Results are saved in [README.md](README.md).

- **Algorithm:** Viola–Jones-style Haar/AdaBoost cascade with **4 stages: 8, 24, 48, 80 weak classifiers**.
- **Negative shortage:** horizontal flips and four one-pixel translations expand training negatives; later stages train on survivors.
- **Hyperparameters:** 6,500 candidate features, 128 response bins, standard-deviation floor of 1, AdaBoost coefficient cap of 2, and stage thresholds targeting 99.85% training face retention.
- **Hidden validation:** AUC **0.998**. Detection/precision: **98.4%/99.4%** at 1% FP, **99.5%/92.1%** at both 5% and 10% FP.
- **Speed:** server reported **25.1 s** training wall time and **0.2 µs CPU/image**, excluding startup/loading. Non-faces averaged **21.5 of 160** weak evaluations.
- **Checks:** build, model description, smoke predictions, and saved holdout metrics passed.

Preserved the code and trained model; used **1/20 validation calls** this session.