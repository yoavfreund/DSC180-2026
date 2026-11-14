"""Build checks and the coordinator end to end, with fake agents (no LLM)."""
import json
import os

import numpy as np
import pytest

from config import AgentSpec, Config
from coordinator.coordinator import Coordinator
from ensemble import learner_outputs, running_scores
from harness.build import LearnerError, build_learner, run_scores, time_ns, write_images
from harness.threshold import fit_stump

RULE = "/* rule: band */ double weak_score(const vj_image *im) { return vj_haar2_tb(im, 2, 5, 15, 3); }"


@pytest.fixture
def images_file(tmp_path):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (300, 19, 19), dtype=np.uint8)
    path = tmp_path / "imgs.bin"
    write_images(imgs, path)
    return imgs, path


def test_c_scores_match_numpy_reference(images_file, tmp_path):
    imgs, path = images_file
    b = build_learner(RULE, tmp_path / "b", tmp_path / "cache")
    s = run_scores(b.binary, path, len(imgs))
    x = imgs.astype(np.float64)
    var = x.reshape(len(x), -1).var(1)
    std = np.where(var > 1, np.sqrt(np.maximum(var, 1)), 1.0)
    ref = (x[:, 5:8, 2:17].sum((1, 2)) - x[:, 8:11, 2:17].sum((1, 2))) / std
    assert np.allclose(s, ref, atol=1e-9)
    ns = time_ns(b.binary, path, len(imgs), reps=3)
    assert 0 < ns < 1000


@pytest.mark.parametrize("source, message", [
    ("#include <stdio.h>\ndouble weak_score(const vj_image *im) { return 0; }", "#include"),
    ("void *fopen(const char*, const char*);\n"
     "double weak_score(const vj_image *im) { return fopen(\"x\", \"r\") != 0; }", "not allowed: fopen"),
    ("int system(const char*);\ndouble weak_score(const vj_image *im) { return system(\"ls\"); }", "system"),
    ("double weak_score(const vj_image *im) { __asm__(\"nop\"); return 0; }", "assembly"),
    ("double weak_score(const vj_image *im) { return 0 }", "compile error"),
    ("int main(void) { return 0; }\ndouble weak_score(const vj_image *im) { return 0; }", "link error"),
    ("double other(const vj_image *im) { return 0; }", "must define"),
    ("/*" + "x" * 9000 + "*/ double weak_score(const vj_image *im) { return 0; }", "limit"),
])
def test_build_rejects_unsafe_or_broken_source(source, message, tmp_path):
    with pytest.raises(LearnerError, match=message):
        build_learner(source, tmp_path / "b", tmp_path / "cache")


@pytest.mark.parametrize("source, message", [
    ("double weak_score(const vj_image *im) { return sqrt(-1.0 - im->px[0][0]); }", "NaN"),
    ("double weak_score(const vj_image *im) { volatile int i = 0; for (;;) i++; return 0; }", "longer"),
    ("double weak_score(const vj_image *im) { volatile int *p = 0; return *p; }", "crashed"),
])
def test_bad_runtime_behavior_fails_the_candidate(source, message, images_file, tmp_path):
    imgs, path = images_file
    b = build_learner(source, tmp_path / "b", tmp_path / "cache")
    with pytest.raises(LearnerError, match=message):
        run_scores(b.binary, path, len(imgs), timeout_s=2)


def _audit(run_dir, kind):
    return [json.loads(l) for l in (run_dir / "audit.jsonl").read_text().splitlines()
            if json.loads(l)["type"] == kind]


def _group_gone(run_dir):
    """True if no worker started by the run is still alive."""
    out = os.popen("ps -axo command").read()
    return "scripted_worker" not in out and "haar_worker" not in out


def test_coordinator_accepts_good_rule_kills_hung_agent_rejects_unsafe(synthetic_dataset, tmp_path):
    cfg = Config(rounds=1, agents=[AgentSpec("good", "scripted", mode="good"),
                                   AgentSpec("hang", "scripted", mode="hang"),
                                   AgentSpec("bad", "scripted", mode="forbidden"),
                                   AgentSpec("crash", "scripted", mode="crash")],
                 deadline_s=2, fails_per_level=1)
    run = tmp_path / "run"
    summary = Coordinator(cfg, synthetic_dataset, run, tmp_path / "work", echo=lambda *_: None).run()
    assert summary["accepted"] == 1
    results = {r["agent"]: r for r in _audit(run, "AgentResult")}
    assert results["good"]["status"] == "scored"
    assert results["hang"]["killed_at_deadline"] and results["hang"]["status"] == "no submission"
    assert results["bad"]["status"] == "rejected" and "fopen" in results["bad"]["error"]
    assert results["crash"]["exit_code"] == 3
    assert _group_gone(run)

    # Independent check of the accepted rule's validation error and the weight update.
    d = synthetic_dataset
    entry = json.loads((run / "manifest.jsonl").read_text().splitlines()[0])
    assert entry["agent"] == "good" and entry["bar"] == 0.30 and entry["val_err"] < 0.30
    imgs_path = tmp_path / "all.bin"
    write_images(d.images, imgs_path)
    b = build_learner((run / entry["source"]).read_text(), tmp_path / "check", tmp_path / "cache")
    scores = run_scores(b.binary, imgs_path, d.n)
    w0 = np.full(d.n, 1 / d.n)
    stump = fit_stump(scores[d.train_idx], d.y[d.train_idx], w0[d.train_idx])
    pred = stump.predict(scores)
    val_err = np.mean(pred[d.val_idx] != d.y[d.val_idx])
    assert entry["val_err"] == pytest.approx(val_err)
    a = 0.5 * np.log((1 - val_err) / val_err)
    w1 = w0 * np.exp(-a * d.y * pred)
    assert np.allclose(np.load(run / "round_000" / "weights_out.npy"), w1 / w1.sum())

    # The saved ensemble reproduces the coordinator's predictions.
    entries, H = learner_outputs(run, d.images)
    assert np.array_equal(H[0], pred)
    assert running_scores(H, [e["alpha"] for e in entries])[-1] == pytest.approx(a * pred)


def test_submission_counts_even_if_agent_is_killed_later(synthetic_dataset, tmp_path):
    cfg = Config(rounds=1, agents=[AgentSpec("late", "scripted", mode="submit_hang")], deadline_s=1.5)
    run = tmp_path / "run"
    summary = Coordinator(cfg, synthetic_dataset, run, tmp_path / "work", echo=lambda *_: None).run()
    r = _audit(run, "AgentResult")[0]
    assert r["killed_at_deadline"] and r["status"] == "scored"
    assert summary["accepted"] == 1


def test_exhausted_ladder_uses_fallback_or_skips(synthetic_dataset, tmp_path):
    for fallback, accepted in ((True, 1), (False, 0)):
        cfg = Config(rounds=1, agents=[AgentSpec("hang", "scripted", mode="hang")], deadline_s=0.5,
                     ladder=[0.30, 0.49], fails_per_level=1, fallback=fallback)
        run = tmp_path / f"run-{fallback}"
        summary = Coordinator(cfg, synthetic_dataset, run, tmp_path / "work", echo=lambda *_: None).run()
        assert summary["accepted"] == accepted
        fails = _audit(run, "AttemptFailed")
        assert [f["result"] for f in fails] == ["raised", "exhausted"]
        if fallback:
            assert summary["fallback_rules"] == 1
            assert json.loads((run / "manifest.jsonl").read_text())["agent"] == "fallback-haar"
        else:
            assert _audit(run, "RoundSkipped")


def test_resubmitting_the_same_rule_is_marked_duplicate(synthetic_dataset, tmp_path):
    # Round 2: the same rule now has error near 0.5, so attempts fail and repeats are duplicates.
    cfg = Config(rounds=2, agents=[AgentSpec("good", "scripted", mode="good")], deadline_s=1,
                 ladder=[0.30, 0.49], fails_per_level=2, fallback=False)
    run = tmp_path / "run"
    Coordinator(cfg, synthetic_dataset, run, tmp_path / "work", echo=lambda *_: None).run()
    statuses = [r["status"] for r in _audit(run, "AgentResult") if r["round"] == 1]
    assert statuses[0] == "scored" and set(statuses[1:]) == {"duplicate"}


def test_marginal_ns_charges_expensive_rules_and_not_cheap_ones(images_file, tmp_path):
    from harness.build import baseline_binary, marginal_ns
    imgs, path = images_file
    cheap = build_learner(RULE, tmp_path / "cheap", tmp_path / "cache").binary
    loop = build_learner("double weak_score(const vj_image *im) { double s = 0; "
                         "for (int y = 0; y < 19; y++) for (int x = 0; x < 19; x++) "
                         "s += sqrt(im->px[y][x] + x); return s; }", tmp_path / "loop", tmp_path / "cache").binary
    (c_ns, c_raw), (l_ns, l_raw) = marginal_ns([cheap, loop], path, 256, baseline_binary(tmp_path / "cache"))
    assert c_ns < 3.0                 # a Haar feature is close to free over the call overhead
    assert l_ns > 20 * max(c_ns, 1.0)  # 361 square roots are not
    assert c_raw >= c_ns and l_raw >= l_ns
