"""The coordinator: plain Python, no LLM. It runs the boosting loop.

    for each round:                                  outer loop
        bar moves one level down (carry-over)
        while True:                                  inner loop: attempts
            a rule seen this round is under the bar? -> accept it, next round
            start every agent as its own process; wait up to deadline_s; kill the rest
            build, score and time every submission (the coordinator re-does all of it)
            a new rule is under the bar?             -> accept it, next round
            count a failure; after fails_per_level failures the bar moves up one level
            top of the ladder exhausted?             -> fallback Haar search, or skip the round

Agents run as separate OS processes in their own process group. At the deadline
the whole group is killed, so no agent session or `claude` process outlives it.
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agents.common import write_private
from common.audit import AuditLog
from common.hashing import sha256_array
from common.metrics import auc, detection_at_fp
from coordinator import brief as B
from coordinator.ladder import Ladder
from coordinator.selection import alpha, choose, progress_per_ns, required_error
import harness.build as hb
from harness.build import LearnerError, baseline_binary, build_learner, marginal_ns, run_scores, write_images
from harness.threshold import fit_stump

REPO = Path(__file__).resolve().parent.parent
# Variables that tie a process to an outer Claude Code session; agents must start clean sessions.
SESSION_VARS = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION",
                "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_PID")
FALLBACK_TIMEOUT_S = 180


@dataclass
class Candidate:
    agent: str
    round: int
    attempt: int
    order: int
    source: str
    sha256: str
    description: str
    theta: float
    polarity: int
    train_err: float
    val_err: float
    pred: np.ndarray       # +1 / -1 on every image (training and validation)
    binary: Path
    ns: float = float("nan")       # extra ns per call over the empty rule (see harness.build.marginal_ns)
    ns_raw: float = float("nan")   # measured ns per call including the call overhead
    fallback: bool = False


class Coordinator:
    def __init__(self, cfg, data, run_dir, work_dir, echo=print):
        self.cfg, self.data = cfg, data
        self.run_dir, self.work_dir = Path(run_dir), Path(work_dir)
        self.echo = echo
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLog(self.run_dir / "audit.jsonl")
        harness = hb.snapshot_harness(self.run_dir / "harness")   # this run builds only from its own copy
        self.cache = self.run_dir / "cache"
        (self.run_dir / "learners").mkdir()
        data_dir = self.run_dir / "data"
        data_dir.mkdir()
        self.all_images = data_dir / "all_images.bin"     # coordinator only; agents never see this file
        write_images(data.images, self.all_images)
        self.train_images = data_dir / "train_images.bin"  # training rows only; linked into each round
        write_images(data.images[data.train_idx], self.train_images)
        rng = np.random.default_rng(cfg.seed)
        timing_idx = np.sort(rng.choice(data.n, size=min(cfg.timing_images, data.n), replace=False))
        self.timing_images = data_dir / "timing_images.bin"
        self.n_timing = write_images(data.images[timing_idx], self.timing_images)
        self.baseline = baseline_binary(self.cache)

        self.w = np.zeros(data.n)                          # audit images (if any) keep weight 0 forever
        active = np.concatenate([data.train_idx, data.val_idx])
        self.w[active] = 1.0 / len(active)
        if cfg.balanced_init:
            # Within training and within validation: half the weight on faces, half on non-faces.
            for split in (data.train_idx, data.val_idx):
                for label in (1, -1):
                    rows = split[data.y[split] == label]
                    self.w[rows] = (len(split) / len(active)) / (2 * len(rows))
            self.w /= self.w.sum()
        self.ess_val = float(len(data.val_idx))
        self.F = np.zeros(data.n)                          # ensemble score sum(alpha * h) on every image
        self.ladder = Ladder(cfg.ladder, cfg.fails_per_level, cfg.carry_over)
        self.ensemble = []
        self.feedback = {a.name: "" for a in cfg.agents}
        self.memory = {a.name: [] for a in cfg.agents}   # each agent's own submissions, shown in its prompt
        self.accepted_code = []                           # source of each accepted rule, for the prompt
        self.stats = {a.name: {"sessions": 0, "submitted": 0, "rejected": 0, "no_submission": 0,
                               "killed_at_deadline": 0, "wins": 0, "alpha_total": 0.0, "cost_usd": 0.0}
                      for a in cfg.agents}
        self.llm_sessions = 0
        self.order = 0
        self.env = {k: v for k, v in os.environ.items() if k not in SESSION_VARS}
        self.env["PYTHONPATH"] = str(REPO)
        self.env["FACE_HARNESS_DIR"] = str(harness)

    # ------------------------------------------------------------------ outer loop

    def run(self):
        cfg = self.cfg
        self.audit.emit("RunStart", rounds=cfg.rounds, agents=[a.__dict__ for a in cfg.agents],
                        ladder=cfg.ladder, fails_per_level=cfg.fails_per_level, carry_over=cfg.carry_over,
                        deadline_s=cfg.deadline_s, ns_floor=cfg.ns_floor, fallback=cfg.fallback,
                        model=cfg.model, effort=cfg.effort, max_turns=cfg.max_turns,
                        n_train=len(self.data.train_idx), n_val=len(self.data.val_idx),
                        n_audit=len(self.data.audit_idx), confidence=cfg.confidence, weight_cap=cfg.weight_cap,
                        n_train_copies=int(sum(1 for v in self.data.variant[self.data.train_idx] if v)),
                        balanced_init=cfg.balanced_init)
        stop_reason = "all rounds done"
        skipped_in_a_row = 0
        for t in range(cfg.rounds):
            outcome = self._round(t)
            if outcome == "stop":
                stop_reason = "LLM session budget reached"
                break
            # A skipped round leaves the weights unchanged, so the next round faces the same problem.
            skipped_in_a_row = skipped_in_a_row + 1 if outcome == "skipped" else 0
            if skipped_in_a_row >= cfg.max_skips_in_a_row:
                stop_reason = f"stalled: {skipped_in_a_row} rounds in a row had no rule under the bar"
                self.echo(stop_reason)
                break
        return self._finish(stop_reason)

    # ------------------------------------------------------------------ one round

    def _round(self, t):
        d, cfg = self.data, self.cfg
        self.ladder.start_round()
        round_dir = self.run_dir / f"round_{t:03d}"
        private = round_dir / "private"
        tr = d.train_idx
        write_private(private, d.images[tr], d.y[tr], self.w[tr], seed=cfg.seed * 1000 + t,
                      shared_images=self.train_images)                                       # training only
        np.save(round_dir / "weights_in.npy", self.w)
        self.ess_val = _ess(self.w[d.val_idx])
        self.audit.emit("RoundStart", round=t, bar=self.ladder.bar, weights_sha256=sha256_array(self.w),
                        ess_val=self.ess_val, ess_train=_ess(self.w[tr]))
        self.echo(f"round {t}: bar {self.ladder.bar:.3f}")

        seen, attempt = [], 0
        self.round_cache = {}                     # source sha256 -> Candidate or LearnerError, this round only
        self.det_source = {}                      # deterministic agents run once per round: name -> source
        while True:
            pick = self._passing(seen)
            if pick:
                return self._accept(t, pick, seen)
            n_llm = sum(a.kind == "claude" for a in cfg.agents)
            if n_llm and self.llm_sessions + n_llm > cfg.max_llm_sessions:
                self.audit.emit("BudgetStop", round=t, llm_sessions=self.llm_sessions)
                return "stop"

            new, outcomes = self._attempt(t, attempt, round_dir)
            seen.extend(new)
            pick = self._passing(seen)
            if pick:
                return self._accept(t, pick, seen)

            bar = self.ladder.bar
            result = self.ladder.record_failure()
            self.audit.emit("AttemptFailed", round=t, attempt=attempt, bar=bar, result=result,
                            new_bar=self.ladder.bar)
            self.echo(f"  attempt {attempt}: nothing under {bar:.3f} -> {result}"
                      + (f", bar now {self.ladder.bar:.3f}" if result == "raised" else ""))
            self._feedback_failed(outcomes, bar, result)
            attempt += 1
            if result == "exhausted":
                if cfg.fallback:
                    fb = self._fallback(t, round_dir)
                    if fb is not None and fb.val_err < self.ladder.bar:
                        return self._accept(t, fb, seen + [fb])
                self.audit.emit("RoundSkipped", round=t, reason="ladder exhausted")
                self.echo(f"  round {t} skipped")
                return "skipped"

    def margin(self, err):
        """confidence x the standard error of a weighted validation error, from the effective sample size."""
        e = min(max(err, 0.0), 1.0)
        return self.cfg.confidence * float(np.sqrt(max(e * (1 - e), 1e-4) / max(self.ess_val, 1.0)))

    def _passing(self, seen):
        under = [c for c in seen if c.val_err + self.margin(c.val_err) < self.ladder.bar]
        return choose(under, self.cfg.ns_floor) if under else None

    # ------------------------------------------------------------------ one attempt

    def _attempt(self, t, attempt, round_dir):
        cfg = self.cfg
        d = self.data
        attempt_dir = round_dir / f"attempt_{attempt:02d}"
        history = B.history_text(self.ensemble, self.accepted_code)
        tr = d.train_idx
        rng = np.random.default_rng([cfg.seed, t, attempt])        # a new draw each attempt, reproducible
        correct = (np.sign(self.F[tr] + 1e-12) == d.y[tr]) if self.ensemble else None
        preview, shown = B.preview(d.images[tr], d.y[tr], self.w[tr], rng, correct, d.variant[tr])
        self.audit.emit("AttemptStart", round=t, attempt=attempt, bar=self.ladder.bar,
                        preview_ids=[d.ids[tr[i]] for i in shown])
        procs = {}
        start = time.monotonic()
        for agent in cfg.agents:
            out = attempt_dir / agent.name
            out.mkdir(parents=True)
            if agent.kind == "haar" and agent.name in self.det_source:
                # Same data and weights give the same search result: reuse it, start nothing.
                (out / "submission.c").write_text(self.det_source[agent.name], encoding="utf-8")
                procs[agent.name] = [None, None, 0.0]
                continue
            cmd = self._command(agent, out, round_dir / "private", t, attempt, preview, history)
            log = open(out / "worker.log", "w")
            procs[agent.name] = [subprocess.Popen(cmd, cwd=REPO, env=self.env, stdout=log,
                                                  stderr=subprocess.STDOUT, start_new_session=True),
                                 log, None]
            self.stats[agent.name]["sessions"] += 1    # reused deterministic results skip this
            if agent.kind == "claude":
                self.llm_sessions += 1

        deadline = start + cfg.deadline_s
        while time.monotonic() < deadline:
            running = False
            for entry in procs.values():
                if entry[2] is None and entry[0] is not None and entry[0].poll() is not None:
                    entry[2] = time.monotonic() - start
                running |= entry[2] is None
            if not running:
                break
            time.sleep(0.1)
        for entry in procs.values():
            if entry[0] is None:              # reused deterministic result
                continue
            killed = entry[2] is None
            _kill_group(entry[0])
            entry[1].close()
            if killed:
                entry[2] = "killed"

        # All agents are stopped, so the timing below is not disturbed by them.
        candidates, outcomes, scored = [], {}, []
        for agent in cfg.agents:
            out = attempt_dir / agent.name
            proc, _, finish = procs[agent.name]
            o = {"agent": agent.name, "killed_at_deadline": finish == "killed",
                 "seconds": None if finish == "killed" else round(finish, 1),
                 "exit_code": None if finish == "killed" or proc is None else proc.returncode}
            if proc is None:
                o["reused"] = True
            o.update(_read_result(out))
            if o["killed_at_deadline"]:
                self.stats[agent.name]["killed_at_deadline"] += 1
            self.stats[agent.name]["cost_usd"] += o.get("total_cost_usd") or 0.0
            sub = out / "submission.c"
            if not sub.exists():
                o["status"] = "no submission"
                self.stats[agent.name]["no_submission"] += 1
            else:
                self.stats[agent.name]["submitted"] += 1
                source = sub.read_text(encoding="utf-8")
                if agent.kind == "haar":
                    self.det_source[agent.name] = source
                key = hashlib.sha256(source.encode("utf-8")).hexdigest()
                try:
                    if key in self.round_cache:     # same source, same weights: same result
                        c = self.round_cache[key]
                        if isinstance(c, LearnerError):
                            raise c
                        o["status"] = "duplicate"
                    else:
                        c = self._evaluate(agent.name, t, attempt, source, out / "eval")
                        self.round_cache[key] = c
                        candidates.append(c)
                        o["status"] = "scored"
                    scored.append((o, c))
                except LearnerError as e:
                    self.round_cache.setdefault(key, e)
                    o.update(status="rejected", error=str(e)[:600])
                    self.stats[agent.name]["rejected"] += 1
            outcomes[agent.name] = o

        self._time(candidates)
        for o, c in scored:
            o.update(val_err=c.val_err, train_err=c.train_err, ns=c.ns, ns_raw=c.ns_raw,
                     progress_per_ns=progress_per_ns(c.val_err, c.ns, cfg.ns_floor), description=c.description)
        sources = {c.agent: c for o, c in scored}
        for agent in cfg.agents:
            o = outcomes[agent.name]
            self.audit.emit("AgentResult", round=t, attempt=attempt, **o)
            self.echo("    " + _outcome_line(o))
            c = sources.get(agent.name)
            self.memory[agent.name].append({
                "round": t, "attempt": attempt, "outcome": _first_outcome(o),
                "pending": o["status"] in ("scored", "duplicate"),
                "val_err": c.val_err if c else None, "ns": c.ns if c else None,
                "source": c.source if c else None, "sha256": c.sha256 if c else None})
        return candidates, outcomes

    def _command(self, agent, out, private, t, attempt, preview, history):
        py = sys.executable
        if agent.kind == "haar":
            return [py, "-m", "agents.haar_worker", "--private", str(private), "--out", str(out)]
        if agent.kind == "scripted":
            return [py, "-m", "agents.scripted_worker", "--mode", agent.mode, "--out", str(out)]
        text = B.brief(round_t=t, rounds=self.cfg.rounds, attempt=attempt, bar=self.ladder.bar,
                       margin=self.margin(0.35),
                       deadline_s=self.cfg.deadline_s, preview=preview, history=history,
                       feedback=self.feedback[agent.name], memory=B.memory_text(self.memory[agent.name]))
        (out / "brief.md").write_text(text, encoding="utf-8")
        cwd = self.work_dir / agent.name / f"r{t:03d}_a{attempt:02d}"   # outside the repository
        cwd.mkdir(parents=True, exist_ok=True)
        return [py, "-m", "agents.claude_worker", "--system", str(Path(agent.prompt).resolve()),
                "--brief", str(out / "brief.md"), "--private", str(private), "--out", str(out),
                "--cache", str(self.cache), "--cwd", str(cwd), "--bar", str(self.ladder.bar),
                "--ns-floor", str(self.cfg.ns_floor), "--model", self.cfg.model,
                "--effort", self.cfg.effort, "--max-turns", str(self.cfg.max_turns)]

    def _evaluate(self, agent, t, attempt, source, eval_dir):
        """Re-do everything the agent claims: build, score every image, fit on training, judge on validation."""
        d = self.data
        build = build_learner(source, eval_dir, self.cache)
        scores = run_scores(build.binary, self.all_images, d.n)
        tr, va = d.train_idx, d.val_idx
        stump = fit_stump(scores[tr], d.y[tr], self.w[tr])
        pred = stump.predict(scores)
        wv = self.w[va]
        val_err = float(np.dot(wv, pred[va] != d.y[va]) / wv.sum())
        self.order += 1
        return Candidate(agent, t, attempt, self.order, source, build.sha256, B.describe(source),
                         stump.theta, stump.polarity, stump.error, val_err, pred, build.binary)

    def _time(self, candidates):
        """Extra ns over the empty rule, timed in turn with the baseline, on a small cached image set."""
        if not candidates:
            return
        results = marginal_ns([c.binary for c in candidates], self.timing_images, self.n_timing,
                              self.baseline, rounds=self.cfg.timing_rounds, reps=self.cfg.timing_reps)
        for c, (ns, raw) in zip(candidates, results):
            c.ns, c.ns_raw = ns, raw

    def _fallback(self, t, round_dir):
        out = round_dir / "fallback"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "agents.haar_worker", "--private", str(round_dir / "private"),
               "--out", str(out)]
        try:
            subprocess.run(cmd, cwd=REPO, env=self.env, timeout=FALLBACK_TIMEOUT_S,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            c = self._evaluate("fallback-haar", t, -1, (out / "submission.c").read_text(), out / "eval")
            self._time([c])
        except (subprocess.TimeoutExpired, FileNotFoundError, LearnerError) as e:
            self.audit.emit("FallbackFailed", round=t, error=str(e)[:300])
            return None
        c.fallback = True
        self.audit.emit("FallbackResult", round=t, val_err=c.val_err, ns=c.ns, description=c.description)
        self.echo(f"  fallback Haar search: validation error {c.val_err:.4f}, {c.ns:.2f} ns")
        return c

    # ------------------------------------------------------------------ accept a rule

    def _accept(self, t, c, seen):
        d = self.data
        a = alpha(c.val_err)
        w_in = sha256_array(self.w)
        self.w = self.w * np.exp(-a * d.y * c.pred)
        self.w /= self.w.sum()
        if self.cfg.weight_cap:
            self.w = _cap_weights(self.w, self.cfg.weight_cap / np.count_nonzero(self.w))
        self.F += a * c.pred
        np.save(self.run_dir / f"round_{t:03d}" / "weights_out.npy", self.w)

        src_name = f"learners/round_{t:03d}.c"
        (self.run_dir / src_name).write_text(c.source, encoding="utf-8")
        ppn = progress_per_ns(c.val_err, c.ns, self.cfg.ns_floor)
        entry = {"round": t, "agent": c.agent, "attempt": c.attempt, "bar": self.ladder.bar,
                 "val_err": c.val_err, "train_err": c.train_err, "alpha": a, "ns": c.ns, "ns_raw": c.ns_raw,
                 "progress_per_ns": ppn, "theta": c.theta, "polarity": c.polarity,
                 "source": src_name, "source_sha256": c.sha256, "description": c.description,
                 "fallback": c.fallback}
        with open(self.run_dir / "manifest.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        self.ensemble.append(entry)
        self.accepted_code.append(c.source)
        if c.agent in self.stats:
            self.stats[c.agent]["wins"] += 1
            self.stats[c.agent]["alpha_total"] += a

        tr, va = d.train_idx, d.val_idx
        metrics = {"ensemble_train_err": float(np.mean(np.sign(self.F[tr] + 1e-12) != d.y[tr])),
                   "ensemble_val_err": float(np.mean(np.sign(self.F[va] + 1e-12) != d.y[va])),
                   "ensemble_val_auc": auc(self.F[va], d.y[va])}
        if len(d.audit_idx):
            au = d.audit_idx
            metrics.update(audit_auc=auc(self.F[au], d.y[au]),
                           audit_err=float(np.mean(np.sign(self.F[au] + 1e-12) != d.y[au])),
                           audit_detection_at_1pct_fp=detection_at_fp(self.F[au], d.y[au], 0.01))
        self.audit.emit("RoundAccepted", **{k: v for k, v in entry.items() if k != "source"},
                        weights_in_sha256=w_in, weights_out_sha256=sha256_array(self.w), **metrics)
        self.echo(f"  accepted {c.agent}: validation error {c.val_err:.4f}, {c.ns:.2f} ns, alpha {a:.3f} | "
                  f"ensemble validation AUC {metrics['ensemble_val_auc']:.4f}"
                  + (f", audit AUC {metrics['audit_auc']:.4f}" if "audit_auc" in metrics else ""))
        self._feedback_round(t, c, seen)
        return "accepted"

    # ------------------------------------------------------------------ feedback to agents

    def _feedback_failed(self, outcomes, bar, result):
        for entries in self.memory.values():
            for e in entries:
                if e["pending"]:
                    e["outcome"], e["pending"] = f"missed the bar {bar:.3f}", False
        after = {"failed": f"The bar stays at {self.ladder.bar:.3f}.",
                 "raised": f"The bar is now {self.ladder.bar:.3f}.",
                 "exhausted": "The bar is at its ceiling; this round is ending."}[result]
        for name, o in outcomes.items():
            self.feedback[name] = (f"Last attempt: no rule from any agent had validation error below "
                                   f"{bar:.3f}. {after}\nYour result: {_agent_view(o)}")

    def _feedback_round(self, t, win, seen):
        for name, entries in self.memory.items():
            for e in entries:
                if e["round"] != t:
                    continue
                if name == win.agent and e["sha256"] == win.sha256:
                    e["outcome"] = f"ACCEPTED at bar {self.ladder.bar:.3f}"
                elif e["pending"]:
                    bar = self.ladder.bar
                    e["outcome"] = (f"under the bar {bar:.3f} but not chosen (round went to {win.agent})"
                                    if e["val_err"] is not None and e["val_err"] < bar else f"missed the bar {bar:.3f}")
                e["pending"] = False
        win_ppn = progress_per_ns(win.val_err, win.ns, self.cfg.ns_floor)
        head = (f"Round {t + 1} kept {win.agent}'s rule: validation error {win.val_err:.3f}, "
                f"{win.ns:.2f} ns, progress per ns {win_ppn:.5f} (bar {self.ladder.bar:.3f}). "
                f"The weights have changed, so the next round's hard images are different.")
        for agent in self.cfg.agents:
            mine = [c for c in seen if c.agent == agent.name]
            if agent.name == win.agent:
                self.feedback[agent.name] = head + "\nYou won this round."
            elif not mine:
                self.feedback[agent.name] = head + "\nYou submitted no usable rule this round."
            else:
                best = max(mine, key=lambda c: progress_per_ns(c.val_err, c.ns, self.cfg.ns_floor))
                need = required_error(win_ppn, best.ns, self.cfg.ns_floor)
                why = (f"At {best.ns:.2f} ns you needed validation error {need:.3f} or lower to win."
                       if need is not None else f"At {best.ns:.2f} ns no error is low enough; make it faster.")
                self.feedback[agent.name] = (head + f"\nYour best rule this round: validation error "
                                             f"{best.val_err:.3f}, {best.ns:.2f} ns. {why}")

    # ------------------------------------------------------------------ end of run

    def _finish(self, stop_reason):
        d = self.data
        va = d.val_idx
        summary = {"stop_reason": stop_reason, "accepted": len(self.ensemble),
                   "fallback_rules": sum(e["fallback"] for e in self.ensemble),
                   "llm_sessions": self.llm_sessions, "agents": self.stats}
        if self.ensemble:
            summary.update(val_auc=auc(self.F[va], d.y[va]),
                           val_detection_at_fp={str(r): detection_at_fp(self.F[va], d.y[va], r)
                                                for r in (0.001, 0.01, 0.1)},
                           ns_total=sum(e["ns"] for e in self.ensemble))
            if len(d.audit_idx):
                au = d.audit_idx
                summary.update(audit_auc=auc(self.F[au], d.y[au]),
                               audit_detection_at_fp={str(r): detection_at_fp(self.F[au], d.y[au], r)
                                                      for r in (0.001, 0.01, 0.1)})
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        self.audit.emit("RunEnd", **summary)
        return summary


def _ess(w):
    """Effective sample size of a weight vector: (sum w)^2 / sum w^2."""
    w = np.asarray(w, dtype=np.float64)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300))


def _cap_weights(w, cap, iters=100):
    """Clip weights at `cap` and renormalize until no weight is above it (MadaBoost-style robustness)."""
    w = w.copy()
    for _ in range(iters):
        if w.max() <= cap * (1 + 1e-9):
            break
        w = np.minimum(w, cap)
        w /= w.sum()
    return w


def _kill_group(proc):
    """Stop a worker and everything it started (its process group), even if the worker already exited."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)   # whatever is left in the group, such as a `claude` child
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


def _read_result(out):
    path = Path(out) / "result.json"
    if not path.exists():
        return {}
    try:
        r = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    keep = ("num_turns", "total_cost_usd", "is_error", "subtype", "error", "api_error_status")
    return {("worker_" + k if k == "error" else k): r.get(k) for k in keep if r.get(k) is not None}


def _first_outcome(o):
    status = o.get("status")
    if status == "rejected":
        return "rejected: " + o.get("error", "").splitlines()[0][:150]
    if status == "duplicate":
        return "a duplicate of a rule already scored this round"
    if status == "no submission":
        return "stopped at the deadline with no submission" if o.get("killed_at_deadline") else "no submission"
    return "scored"


def _agent_view(o):
    if o.get("status") == "duplicate":
        return (f"you resubmitted a rule already scored this round (validation error {o['val_err']:.3f}). "
                "The weights do not change within a round, so the same rule always gets the same error.")
    if o.get("status") == "scored":
        return (f"validation error {o['val_err']:.3f} (training error {o['train_err']:.3f}), "
                f"{o['ns']:.2f} ns per image.")
    if o.get("status") == "rejected":
        return "your submission was rejected: " + o.get("error", "")[:400]
    if o.get("killed_at_deadline"):
        return "you were stopped at the deadline before you submitted a rule. Submit earlier."
    return "you finished without submitting a rule."


def _outcome_line(o):
    when = "killed at deadline" if o["killed_at_deadline"] else f"{o['seconds']} s"
    if o.get("status") in ("scored", "duplicate"):
        return (f"{o['agent']}: {'(duplicate) ' if o['status'] == 'duplicate' else ''}val err {o['val_err']:.4f}, train err {o['train_err']:.4f}, "
                f"{o['ns']:.2f} ns ({when})")
    return f"{o['agent']}: {o.get('status')} ({when})" + (f": {o['error'][:120]}" if o.get("error") else "")
