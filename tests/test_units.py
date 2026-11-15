import math

import numpy as np
import pytest

from common.metrics import auc, detection_at_fp
from coordinator.brief import brief, describe, history_text, preview
from coordinator.ladder import Ladder, flat_ladder, parse_ladder
from coordinator.selection import alpha, choose, progress, progress_per_ns, required_error
from ensemble import apply_cascade, calibrate, default_cuts, mean_ns_per_image, running_scores
from harness.threshold import fit_stump


# ---------------------------------------------------------------- threshold fit

def brute_force_stump(scores, y, w):
    best = math.inf
    cands = np.concatenate([[scores.min() - 1], np.unique(scores)])
    for theta in cands:
        for pol in (1, -1):
            pred = np.where(scores > theta, pol, -pol)
            best = min(best, w[pred != y].sum() / w.sum())
    return best


@pytest.mark.parametrize("seed", range(25))
def test_fit_stump_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    scores = rng.integers(0, 6, n).astype(float)          # many ties on purpose
    y = rng.choice([-1, 1], n)
    w = rng.random(n) + 0.01
    s = fit_stump(scores, y, w)
    assert s.error == pytest.approx(brute_force_stump(scores, y, w), abs=1e-12)
    pred = s.predict(scores)
    assert w[pred != y].sum() / w.sum() == pytest.approx(s.error, abs=1e-12)


def test_fit_stump_ties_never_split_and_extremes():
    s = fit_stump(np.array([1.0, 1.0, 1.0]), np.array([1, -1, 1]), np.ones(3))
    assert s.error == pytest.approx(1 / 3)
    big = np.array([1e300, 1e300, -1e300])
    s = fit_stump(big, np.array([1, 1, -1]), np.ones(3))
    assert s.error == 0 and list(s.predict(big)) == [1, 1, -1]
    adjacent = np.array([1.0, np.nextafter(1.0, 2.0)])
    s = fit_stump(adjacent, np.array([-1, 1]), np.ones(2))
    assert s.error == 0 and list(s.predict(adjacent)) == [-1, 1]


# ---------------------------------------------------------------- ladder

def test_flat_ladder_matches_the_plan():
    assert flat_ladder() == [0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.49]
    assert parse_ladder("0.3,0.36,0.4,0.49") == [0.3, 0.36, 0.4, 0.49]
    with pytest.raises(ValueError):
        parse_ladder("0.3,0.3")
    with pytest.raises(ValueError):
        parse_ladder("0.3,0.5")


def test_ladder_raises_after_fails_and_carries_over():
    lad = Ladder([0.30, 0.35, 0.40], fails_per_level=2)
    lad.start_round()
    assert lad.bar == 0.30
    assert lad.record_failure() == "failed"
    assert lad.record_failure() == "raised" and lad.bar == 0.35
    assert lad.record_failure() == "failed"
    assert lad.record_failure() == "raised" and lad.bar == 0.40
    assert lad.record_failure() == "failed"
    assert lad.record_failure() == "exhausted" and lad.bar == 0.40
    lad.start_round()                       # next round: one level tighter
    assert lad.bar == 0.35 and lad.fails == 0
    lad.start_round()
    lad.start_round()
    assert lad.bar == 0.30                  # never below the bottom


def test_ladder_without_carry_over_restarts():
    lad = Ladder([0.30, 0.35], fails_per_level=1, carry_over=False)
    lad.start_round()
    lad.record_failure()
    assert lad.bar == 0.35
    lad.start_round()
    assert lad.bar == 0.30


# ---------------------------------------------------------------- selection

class C:
    def __init__(self, val_err, ns, order):
        self.val_err, self.ns, self.order = val_err, ns, order


def test_progress_matches_the_worked_example():
    assert progress(0.30) == pytest.approx(-0.5 * math.log(0.84))
    picked = choose([C(0.30, 5, 0), C(0.25, 40, 1), C(0.45, 1, 2)])
    assert picked.order == 0                                     # A wins, as in the plan
    assert progress(0.5) == 0 and progress(0.7) == 0
    assert alpha(0.3) == pytest.approx(0.5 * math.log(0.7 / 0.3))


def test_choose_uses_ns_floor_and_tie_breaks():
    # Below the floor every rule counts as 1 ns, so the lower error wins.
    assert choose([C(0.30, 0.2, 0), C(0.20, 0.9, 1)], ns_floor=1.0).order == 1
    assert choose([C(0.30, 2, 0), C(0.30, 2, 1)]).order == 0


def test_required_error_inverts_progress_per_ns():
    target = progress_per_ns(0.30, 5)
    need = required_error(target, 12)
    assert progress_per_ns(need, 12) == pytest.approx(target)
    assert required_error(1e9, 5) is None


# ---------------------------------------------------------------- metrics

def test_auc_and_detection_hand_cases():
    y = np.array([1, 1, -1, -1])
    assert auc(np.array([4, 3, 2, 1.0]), y) == 1.0
    assert auc(np.array([1, 2, 3, 4.0]), y) == 0.0
    assert auc(np.array([1, 1, 1, 1.0]), y) == 0.5
    s = np.array([5, 3, 4, 1, 0, 2.0])
    yy = np.array([1, 1, -1, -1, -1, 1])
    assert detection_at_fp(s, yy, 0.0) == pytest.approx(1 / 3)    # only the 5 before the first FP
    assert detection_at_fp(s, yy, 1 / 3) == 1.0                  # one FP allowed: 5, 4(FP), 3, 2
    assert detection_at_fp(np.ones(4), y, 0.4) == 0.0              # tied group cannot be split


# ---------------------------------------------------------------- cascade

def test_cascade_scores_order_by_exit_point_and_respect_face_budget():
    rng = np.random.default_rng(0)
    T, n = 8, 400
    y = np.where(np.arange(n) < 150, 1, -1)
    H = np.where(rng.random((T, n)) < np.where(y > 0, 0.75, 0.3), 1, -1).astype(np.int8)
    alphas = rng.uniform(0.2, 0.8, T)
    F = running_scores(H, alphas)
    cuts = default_cuts(T)
    assert cuts == [1, 2, 4]
    stages = calibrate(F, y, cuts, face_loss=0.06)
    score, used = apply_cascade(F, stages, alphas)
    faces_lost = np.mean(used[y > 0] < T)
    assert faces_lost <= 0.06 + 1e-9
    order = np.argsort(score)
    # Every image that exits earlier ranks below every image that exits later.
    exits = used[order]
    assert all(a <= b for a, b in zip(exits, exits[1:]))
    survivors = used == T
    assert np.allclose(np.argsort(score[survivors], kind="stable"),
                       np.argsort(F[-1, survivors], kind="stable"))
    assert mean_ns_per_image(used, [1.0] * T) == pytest.approx(used.mean())


def test_calibrate_with_tiny_budget_rejects_nothing():
    F = running_scores(np.ones((4, 5), dtype=np.int8), np.ones(4))
    stages = calibrate(F, np.ones(5), [1, 2], face_loss=0.01)
    assert all(st["theta"] is None for st in stages)


# ---------------------------------------------------------------- brief

def test_brief_contains_contract_bar_and_feedback():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (50, 19, 19), dtype=np.uint8)
    y = np.where(np.arange(50) < 20, 1, -1)
    view, shown = preview(images, y, rng.random(50), rng)
    text = brief(round_t=2, rounds=10, attempt=1, bar=0.33, deadline_s=40, preview=view,
                 history=history_text([]), feedback="Your result: X")
    assert "double weak_score(const vj_image *im);" in text
    assert "0.330" in text and "Round 3 of 10, attempt 2" in text and "Your result: X" in text
    assert describe("/* rule: eyes */ double weak_score(...)") == "rule: eyes"
    assert describe("double f;") == "(no description comment)"


def test_memory_and_history_show_code_and_outcomes():
    from coordinator.brief import memory_text
    entries = [{"round": 0, "attempt": 0, "outcome": "missed the bar 0.300", "val_err": 0.31, "ns": 1.2,
                "source": "/* rule: eye band */\ndouble weak_score(const vj_image *im) {\n  return 1;\n}"},
               {"round": 1, "attempt": 0, "outcome": "stopped at the deadline with no submission",
                "val_err": None, "ns": None, "source": None}]
    text = memory_text(entries)
    assert "round 1, attempt 1: missed the bar 0.300 (validation error 0.310" in text
    assert "`/* rule: eye band */ double weak_score(const vj_image *im) { return 1; }`" in text
    assert "round 2, attempt 1: stopped at the deadline" in text
    ens = [{"round": 0, "agent": "base", "val_err": 0.14, "alpha": 0.9, "ns": 0.1, "description": "d"}]
    hist = history_text(ens, ["double weak_score(const vj_image *im) { return 2; }"])
    assert "1 | base |" in hist and "- round 1: `double weak_score" in hist


def test_preview_draws_four_of_each_class_by_weight_and_hides_the_rest():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (200, 19, 19), dtype=np.uint8)
    y = np.where(np.arange(200) < 60, 1, -1)
    w = np.zeros(200)
    w[[1, 2, 3, 4, 5]] = 1.0                   # only 5 faces and 5 non-faces can be drawn
    w[[100, 101, 102, 103, 104]] = 1.0
    correct = np.arange(200) % 2 == 0
    text, shown = preview(images, y, w, np.random.default_rng(1), correct, np.full(200, "", dtype=object))
    assert len(shown) == 8 and sum(y[i] > 0 for i in shown) == 4
    assert set(shown) <= {1, 2, 3, 4, 5, 100, 101, 102, 103, 104}
    assert "gets it WRONG" in text and "gets it RIGHT" in text and "x average" in text
    _, again = preview(images, y, w, np.random.default_rng(1), correct)
    _, other = preview(images, y, w, np.random.default_rng(2), correct)
    assert again == shown and other != shown   # reproducible per seed, new draw per attempt
    text0, _ = preview(images, y, w, np.random.default_rng(1))
    assert "no rules yet" in text0 and "minus" not in text0   # no averaged difference map any more


def test_augment_keeps_labels_and_makes_nine_copies():
    from common.data import augment
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (6, 19, 19), dtype=np.uint8)
    y = np.array([1, 1, -1, -1, -1, 1], dtype=np.int8)
    ax, ay, src, names = augment(images, y, ("flip", "rotate", "zoom", "shift"))
    assert len(ax) == 6 * 9
    assert np.array_equal(ay, y[src])
    flip = ax[list(names).index("flip")]
    assert np.array_equal(flip, images[0][:, ::-1])
    right = ax[list(names).index("shift+x")]
    assert np.array_equal(right[:, 1:], images[0][:, :-1])   # content moved one pixel right
    for removed in ("invert", "blur"):
        with pytest.raises(ValueError):
            augment(images, y, (removed,))
