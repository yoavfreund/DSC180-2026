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
    """True if no worker started by THIS run is still alive (other runs may be going)."""
    out = os.popen("ps -axo command").read()
    return str(run_dir) not in out


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
    assert c_ns < 10.0                # a Haar feature is close to free (absolute ns vary by machine;
                                      # measured 4.7 on an M-series Mac, so 3.0 failed there)
    assert l_ns > 20 * max(c_ns, 1.0)  # 361 square roots are not
    assert c_raw >= c_ns and l_raw >= l_ns


def test_memory_records_outcomes_across_attempts_and_rounds(synthetic_dataset, tmp_path):
    cfg = Config(rounds=2, agents=[AgentSpec("good", "scripted", mode="good"),
                                   AgentSpec("bad", "scripted", mode="forbidden")],
                 deadline_s=1, ladder=[0.30, 0.49], fails_per_level=1, fallback=False)
    coord = Coordinator(cfg, synthetic_dataset, tmp_path / "run", tmp_path / "work", echo=lambda *_: None)
    coord.run()
    good, bad = coord.memory["good"], coord.memory["bad"]
    assert good[0]["outcome"].startswith("ACCEPTED at bar 0.300") and good[0]["source"]
    assert all(not e["pending"] for e in good + bad)
    assert all(e["outcome"].startswith("rejected: calls functions that are not allowed") for e in bad)
    assert len(coord.accepted_code) == len(coord.ensemble) >= 1


def _write_pgms(folder, n, value):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (folder / f"{i:03d}.pgm").write_bytes(b"P5\n19 19\n255\n" + bytes([value]) * 361)


def test_audit_split_is_stratified_disjoint_and_never_weighted(tmp_path, synthetic_dataset):
    from common.data import load_dataset
    _write_pgms(tmp_path / "train" / "face", 10, 200)
    _write_pgms(tmp_path / "train" / "non-face", 20, 50)
    _write_pgms(tmp_path / "validate" / "face", 5, 200)
    _write_pgms(tmp_path / "validate" / "non-face", 5, 50)
    d = load_dataset(tmp_path, audit_frac=0.2)
    assert len(d.audit_idx) == 6 and int((d.y[d.audit_idx] > 0).sum()) == 2
    assert not set(d.audit_idx) & set(d.train_idx) and not set(d.audit_idx) & set(d.val_idx)
    assert len(d.train_idx) + len(d.audit_idx) == 30

    # In a run, audit images keep weight 0 and are reported separately.
    sd = synthetic_dataset
    sd.audit_idx, sd.train_idx = sd.train_idx[::5], np.setdiff1d(sd.train_idx, sd.train_idx[::5])
    cfg = Config(rounds=1, agents=[AgentSpec("good", "scripted", mode="good")], deadline_s=1)
    run = tmp_path / "run"
    Coordinator(cfg, sd, run, tmp_path / "work", echo=lambda *_: None).run()
    w = np.load(run / "round_000" / "weights_out.npy")
    assert np.all(w[sd.audit_idx] == 0) and w.sum() == pytest.approx(1.0)
    assert "audit_auc" in _audit(run, "RoundAccepted")[0]


def test_weight_cap_bounds_every_weight():
    from coordinator.coordinator import _cap_weights, _ess
    rng = np.random.default_rng(0)
    w = rng.exponential(size=1000) ** 6
    w /= w.sum()
    capped = _cap_weights(w, 20 / 1000)
    assert capped.max() <= 20 / 1000 * (1 + 1e-6) and capped.sum() == pytest.approx(1.0)
    assert np.all(np.argsort(capped[w < np.quantile(w, 0.5)]) == np.argsort(w[w < np.quantile(w, 0.5)]))
    assert _ess(capped) > _ess(w)


def test_confidence_margin_blocks_a_rule_that_passes_without_it(synthetic_dataset, tmp_path):
    for confidence, accepted_at_first_try in ((0.0, True), (100.0, False)):
        cfg = Config(rounds=1, agents=[AgentSpec("good", "scripted", mode="good")], deadline_s=1,
                     ladder=[0.30, 0.49], fails_per_level=1, fallback=False, confidence=confidence)
        run = tmp_path / f"run-{confidence}"
        Coordinator(cfg, synthetic_dataset, run, tmp_path / "work", echo=lambda *_: None).run()
        assert (not _audit(run, "AttemptFailed")) == accepted_at_first_try


def test_haar_search_runs_once_per_round(synthetic_dataset, tmp_path):
    cfg = Config(rounds=1, agents=[AgentSpec("base", "haar")], deadline_s=60,
                 ladder=[0.30, 0.49], fails_per_level=2, fallback=False, confidence=100.0)
    run = tmp_path / "run"
    Coordinator(cfg, synthetic_dataset, run, tmp_path / "work", echo=lambda *_: None).run()
    results = _audit(run, "AgentResult")
    assert len(results) == 4 and results[0]["status"] == "scored" and not results[0].get("reused")
    assert all(r["status"] == "duplicate" and r["reused"] for r in results[1:])


def test_run_stops_when_rounds_keep_getting_skipped(synthetic_dataset, tmp_path):
    cfg = Config(rounds=10, agents=[AgentSpec("hang", "scripted", mode="hang")], deadline_s=0.3,
                 ladder=[0.49], fails_per_level=1, fallback=False)
    summary = Coordinator(cfg, synthetic_dataset, tmp_path / "run", tmp_path / "work", echo=lambda *_: None).run()
    assert summary["stop_reason"].startswith("stalled: 2 rounds")
    assert len(_audit(tmp_path / "run", "RoundSkipped")) == 2



def test_copies_stay_on_the_training_side(tmp_path):
    from common.data import load_dataset
    _write_pgms(tmp_path / "train" / "face", 10, 200)
    _write_pgms(tmp_path / "train" / "non-face", 20, 50)
    _write_pgms(tmp_path / "validate" / "face", 5, 200)
    _write_pgms(tmp_path / "validate" / "non-face", 5, 50)
    d = load_dataset(tmp_path, audit_frac=0.2, augment_kinds=("flip", "shift"))
    originals = 30 - len(d.audit_idx)                      # 24 training originals outside the audit share
    copies = [i for i in d.train_idx if d.variant[i]]
    assert len(copies) == originals * 5                    # 1 flip + 4 shifts of each
    assert all(not d.variant[i] for i in np.concatenate([d.val_idx, d.audit_idx]))
    audit_names = {d.ids[i] for i in d.audit_idx}
    assert not any(d.ids[i].split("#")[0] in audit_names for i in copies)


def test_balanced_init_and_search_sample(synthetic_dataset, tmp_path):
    from agents.common import SEARCH_ROWS, TrainData, write_private
    cfg = Config(rounds=1, agents=[AgentSpec("good", "scripted", mode="good")], deadline_s=1, balanced_init=True)
    coord = Coordinator(cfg, synthetic_dataset, tmp_path / "run", tmp_path / "work", echo=lambda *_: None)
    d, w = synthetic_dataset, coord.w
    for split in (d.train_idx, d.val_idx):
        assert w[split][d.y[split] > 0].sum() == pytest.approx(w[split][d.y[split] < 0].sum())
    big = np.repeat(d.images[:400], 25, axis=0)              # 10000 rows > SEARCH_ROWS
    yy = np.repeat(d.y[:400], 25)
    ww = np.ones(len(yy)); ww[yy > 0] = 0                    # all weight on non-faces
    write_private(tmp_path / "p", big, yy, ww)
    s = TrainData.load(tmp_path / "p", search=True)
    assert s.sampled and s.n == SEARCH_ROWS and np.all(s.y < 0) and np.allclose(s.w, 1 / SEARCH_ROWS)
    write_private(tmp_path / "q", d.images[:100], d.y[:100], np.ones(100))
    assert not TrainData.load(tmp_path / "q", search=True).sampled


def _fake_claude(monkeypatch, modes):
    """Claude agents whose sessions are scripted workers: no LLM is called."""
    import sys
    real = Coordinator._command

    def command(self, agent, out, private, t, attempt, preview, history):
        if agent.kind != "claude":
            return real(self, agent, out, private, t, attempt, preview, history)
        return [sys.executable, "-m", "agents.scripted_worker", "--mode", modes[agent.name], "--out", str(out)]
    monkeypatch.setattr(Coordinator, "_command", command)


def test_run_stops_when_every_claude_session_fails(synthetic_dataset, tmp_path, monkeypatch):
    prompt = tmp_path / "p.md"
    prompt.write_text("test prompt")
    _fake_claude(monkeypatch, {"x": "api_error", "y": "api_error"})
    cfg = Config(rounds=3, agents=[AgentSpec("x", "claude", prompt=prompt), AgentSpec("y", "claude", prompt=prompt),
                                   AgentSpec("base", "haar")], deadline_s=30)
    run = tmp_path / "run"
    summary = Coordinator(cfg, synthetic_dataset, run, tmp_path / "work", echo=lambda *_: None).run()
    assert summary["stop_reason"].startswith("Claude sessions failed (2 of 2 in round 0, attempt 0)")
    assert "usage limit" in summary["stop_reason"]
    assert summary["accepted"] == 0          # the Haar rule of that attempt is not kept
    assert {e["agent"] for e in _audit(run, "SessionFailed")} == {"x", "y"}


def test_run_stops_when_some_claude_sessions_keep_failing(synthetic_dataset, tmp_path, monkeypatch):
    prompt = tmp_path / "p.md"
    prompt.write_text("test prompt")
    _fake_claude(monkeypatch, {"x": "api_error", "y": "good"})
    cfg = Config(rounds=5, agents=[AgentSpec("x", "claude", prompt=prompt), AgentSpec("y", "claude", prompt=prompt)],
                 deadline_s=30)
    summary = Coordinator(cfg, synthetic_dataset, tmp_path / "run", tmp_path / "work", echo=lambda *_: None).run()
    assert summary["accepted"] == 1          # round 0 went on (one failure); round 1 is the second in a row
    assert summary["stop_reason"].startswith("Claude sessions failed (1 of 2 in round 1")


def test_failed_attempt_feedback_shows_the_noise_margin(synthetic_dataset, tmp_path):
    cfg = Config(rounds=1, agents=[AgentSpec("good", "scripted", mode="good")], confidence=1.0)
    c = Coordinator(cfg, synthetic_dataset, tmp_path / "run", tmp_path / "work", echo=lambda *_: None)
    c.ess_val = 100.0                        # margin at 0.04: sqrt(0.04 * 0.96 / 100) = 0.0196
    c._feedback_failed({"good": {"status": "scored", "val_err": 0.04, "train_err": 0.03, "ns": 0.1}}, 0.05, "raised")
    text = c.feedback["good"]
    assert "validation error + noise margin below 0.050" in text
    assert "0.040 + 0.020 = 0.060, which is not below 0.050" in text
