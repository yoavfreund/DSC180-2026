"""Design variants for the boosted-cascade task.

Each entry is a short instruction appended to the system prompt under
"Assigned design". The cascade requirements and score convention in the main
prompt still apply to every variant. Add, remove or reword entries freely;
agent.py and run_local.py read this dict.
"""

VARIANTS = {
    # No extra instruction: the agent picks any boosted cascade it can justify.
    "free": "",

    "viola_jones": (
        "Build a classic staged attentional cascade (Viola & Jones, 2001): Haar-like rectangle features "
        "computed on integral images, AdaBoost decision stumps inside each stage, and a per-stage threshold "
        "set to keep roughly 99% or more of training faces while rejecting a large share of non-faces. "
        "Train each later stage on the non-faces the earlier stages still accept."
    ),

    "waldboost": (
        "Build a WaldBoost-style cascade (Sochman & Matas, 2005): one sequence of boosted, confidence-rated weak "
        "classifiers (for example Real AdaBoost stumps on Haar-like features) with an early-rejection threshold "
        "after every weak classifier, so each weak classifier is a stage. Report one `stage=<k> weak=1` line per "
        "weak classifier from `--describe`, k = 1..T. After adding each weak classifier, set its threshold from "
        "training data (use a held-out part of train) so that only a small, fixed fraction of faces falls below "
        "it. Then drop the rejected samples before the next round, and refill the negative pool from augmented "
        "copies (flips, small shifts) of training non-faces, since real negatives run short. Keep T moderate (a "
        "few hundred at most): speed is scored by the average number of weak classifiers evaluated per image. "
        "Scoring: a patch rejected after weak classifier t gets `cascade_score(t - 1, margin)`, where margin is "
        "its running sum minus that threshold, scaled so the sigmoid does not saturate; a patch that is never "
        "rejected is scored with the last stage's margin."
    ),

    "soft_cascade": (
        "Build a soft cascade (Bourdev & Brandt, 2005): first train one boosted classifier, then compute a "
        "rejection threshold after each weak classifier from training data so that at most a small fraction "
        "of faces is lost. Each weak classifier counts as a stage. Order the weak classifiers so the most "
        "discriminative ones come first."
    ),

    "lbp_cascade": (
        "Use Local Binary Pattern features (3x3-neighbourhood codes over small blocks of the patch) with "
        "lookup-table weak classifiers, in a classic staged cascade: boosted stages, a per-stage threshold "
        "that keeps nearly all faces, and later stages trained on the surviving non-faces."
    ),

    "pixel_comparison": (
        "Use very cheap pixel-comparison features, such as the normalized pixel difference (a - b) / (a + b) "
        "(treat a + b = 0 safely) or simple a > b tests, with shallow decision trees (depth 2 to 4) as weak "
        "classifiers, arranged as a boosted cascade. Prioritize prediction speed."
    ),

    "gradient_cascade": (
        "Use gradient-based features (per-cell gradient orientation histograms or magnitude sums, in the "
        "spirit of HOG) with decision stumps as weak classifiers, in a staged boosted cascade with per-stage "
        "thresholds."
    ),
}


def variant_note(name):
    """Prompt text to append for a variant, or '' for none/unknown."""
    text = VARIANTS.get(name or "free", "")
    if not text:
        return ""
    return ("\n\n## Assigned design\n\n" + text +
            "\n\nThe cascade requirements, score convention and rules above still apply. "
            "Within this design, the choices are yours.\n")