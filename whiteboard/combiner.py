"""Box 3: Combiner. Takes all rules, builds the classifier and the cascade.

Classifier: every rule votes +1 or -1, a vote counts alpha times
(alpha is bigger for rules with lower error). Score F = sum(alpha * vote).

Cascade: the same rules in order, with checkpoints after 1, 3, 7, 15, ... rules
(stage sizes 1, 2, 4, 8, ...). At each checkpoint an image is thrown out as a
non-face if its running score is too low. Each threshold keeps 99% of the
training faces that reached it. Most non-faces leave after a few rules, so the
average image costs far less than running every rule.
"""
import numpy as np

from .harness import predict


def votes(rules, key):
    """Matrix of +1/-1 votes, one row per rule, from scores cached under key."""
    return np.array([predict(r, r[key]) for r in rules])


def stage_ends(n_rules):
    ends, size, end = [], 1, 0
    while end < n_rules:
        end = min(end + size, n_rules)
        ends.append(end)
        size *= 2
    return ends


def build(rules, y_train, keep_faces=0.99):
    alpha = np.array([r["alpha"] for r in rules])
    running = np.cumsum(alpha[:, None] * votes(rules, "train_scores"), axis=0)  # score after each rule
    alive = y_train == 1                       # faces still in the cascade
    stages = []
    for end in stage_ends(len(rules)):
        F = running[end - 1]
        theta = float(np.quantile(F[alive], 1 - keep_faces))
        stages.append({"end": end, "theta": theta})
        alive &= F >= theta
    return {"alpha": alpha, "stages": stages}
