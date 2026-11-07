# coordinator/coordinator.py
import json
import math
import os
import multiprocessing as mp
import importlib.util
from pathlib import Path
import time
import statistics

import numpy as np

from agents.learner_agent import LearnerAgent, TrainRequest
from agents.evaluator import Evaluator
from agents.combiner import combine
from common import messages as M
from common.audit import AuditLog
from common.hashing import sha256_array, new_id
from common.splits import seeded_split
from coordinator.specs import parse_specs, DEFAULT_SPECS
from common.images import extract_features

EPS = 1e-6

# Schema policy names -> combiner modes
POLICIES = {"majority": "majority", "soft_vote": "soft", "adaboost_weighted": "adaboost"}


def _train_worker(learners, req, connection):
    try:
        connection.send(learners.train(req))
    finally:
        connection.close()


def _proposal_worker(adapter_path, history, feedback, k, connection):
    try:
        spec = importlib.util.spec_from_file_location("boosting_llm_adapter", adapter_path)
        if spec is None or spec.loader is None:
            raise ValueError("could not load LLM adapter")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        connection.send((True, module.propose_raw(history, feedback, k, timeout=20)))
    except Exception as exc:
        # Exceptions from providers can contain secrets, so do not log their text.
        connection.send((False, f"adapter failed ({type(exc).__name__})"))
    finally:
        connection.close()


def _train_until(learners, requests, deadline):
    """Isolate native training so an over-budget model can be terminated."""
    ctx = mp.get_context("fork")
    pending = []
    for req in requests:
        recv, send = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_train_worker, args=(learners, req, send))
        proc.start()
        send.close()
        pending.append((proc, recv))
    statuses = [None] * len(pending)
    try:
        while time.monotonic() < deadline and any(p.is_alive() for p, _ in pending):
            for i, (proc, recv) in enumerate(pending):
                if statuses[i] is None and recv.poll():
                    try:
                        statuses[i] = recv.recv()
                    except EOFError:
                        pass
            time.sleep(min(.01, max(0, deadline - time.monotonic())))
        for i, (proc, recv) in enumerate(pending):
            if statuses[i] is None and recv.poll():
                try:
                    statuses[i] = recv.recv()
                except EOFError:
                    pass
    finally:
        for proc, recv in pending:
            if proc.is_alive():
                proc.terminate()
            proc.join()
            recv.close()
    return statuses


def _benchmark(learners, handle, X, library, deadline):
    """Median end-to-end per-image time on a shared small batch, with warm-up."""
    indices = np.linspace(0, len(X) - 1, min(len(X), 32), dtype=int)
    sample = np.ascontiguousarray(np.rint(X[indices, :361] * 255), dtype=np.uint8).reshape(-1, 19, 19)
    def run():
        start = time.perf_counter()
        features = extract_features(sample, library)
        learners.predict(handle, features)
        return (time.perf_counter() - start) / len(sample)
    run()
    timings = []
    for _ in range(3):
        if time.monotonic() >= deadline:
            break
        timings.append(run())
    return (statistics.median(timings), len(timings)) if len(timings) == 3 else (None, len(timings))


class Coordinator:
    def __init__(self, cfg, learners: LearnerAgent, evaluator: Evaluator,
                 audit: AuditLog, X, y, X_test, y_test, dataset_ref, run_dir):
        self.cfg = cfg
        self.learners = learners
        self.evaluator = evaluator
        self.audit = audit
        self.X, self.y = X, y
        self.X_test, self.y_test = X_test, y_test   # used only in finalize, never for selection
        self.dataset_ref = dataset_ref
        self.run_dir = run_dir
        self.n = len(y)

        # The coordinator owns all round state.
        self.w = np.full(self.n, 1.0 / self.n)
        self.history = []     # [{"spec", "err"}], fed to the LLM
        self.ensemble = []    # [{"alpha", "handle", "spec", "learner_id"}]

    # ---------- outer protocol ----------

    def run(self):
        for t in range(self.cfg.T):
            best = self._round(t)
            if best is None or best["err"] >= 0.5:
                self.audit.emit({"type": "RoundSkipped", "round": t,
                                 "reason": "no candidate better than chance"})
                continue
            self._update_weights(t, best)
        return self._finalize()

    # ---------- one boosting round ----------

    def _round(self, t):
        deadline = time.monotonic() + self.cfg.time_limit
        split_seed = getattr(self.cfg, "seed", 0) + t
        train_idx, val_idx = seeded_split(self.n, split_seed, self.cfg.val_ratio)
        split = {"type": "seeded", "seed": split_seed, "ratio": self.cfg.val_ratio,
                 "val_sha256": sha256_array(val_idx)}
        round_dir = os.path.join(self.run_dir, f"round_{t:03d}")
        os.makedirs(round_dir, exist_ok=True)
        np.save(os.path.join(round_dir, "weights_in.npy"), self.w)
        w_sha = sha256_array(self.w)

        best, feedback, fallback_used = None, None, False
        for attempt in range(self.cfg.max_refine):
            if time.monotonic() >= deadline:
                break
            audit_context = self.audit.summary()
            context = (feedback or "") + "\nRecent audit records:\n" + json.dumps(audit_context)
            specs, source, raw, why = self._propose(context, deadline)
            if source == "fallback" and fallback_used:
                self.audit.emit({"type": "RefinementStopped", "round": t,
                                 "reason": "fallback already attempted in this round"})
                break
            fallback_used |= source == "fallback"
            self.audit.emit(M.coordinator_decision(
                rationale=f"round {t}, attempt {attempt}: {why}",
                source=source, specs=specs, combiner=self.cfg.combiner,
                feedback_ref=feedback, raw_llm_output=raw))

            # One learner per spec; learner_id is stable across the message trail.
            reqs = []
            for k, spec in enumerate(specs):
                learner_id = f"r{t:03d}_a{attempt}_k{k}"
                reqs.append(TrainRequest(
                    request_id=new_id(), learner_id=learner_id, spec=spec,
                    train_idx=train_idx, w=self.w, round_dir=round_dir))
            for r in reqs:
                self.audit.emit(M.train_request(
                    request_id=r.request_id, learner_id=r.learner_id, spec=r.spec,
                    data_ref=self.dataset_ref, w_sha256=w_sha, round_t=t))

            statuses = _train_until(self.learners, reqs, deadline)

            results, failures = [], []
            for req, st in zip(reqs, statuses):
                if st is None:
                    failures.append(f"{req.spec}: worker failed or timed out")
                    self.audit.emit({"type": "TrainResult", "request_id": req.request_id,
                                     "learner_id": req.learner_id, "status": "timeout_or_error"})
                    continue
                self.audit.emit(M.train_result(
                    request_id=req.request_id, learner_id=req.learner_id, status=st))
                if not st.ok:
                    failures.append(f"{req.spec}: {st.error}")
                    continue

                # Predictions on all rows: needed for the weight update.
                pred_res = self.learners.predict(st.handle, self.X)
                predict_s, timing_count = _benchmark(self.learners, st.handle, self.X,
                                                       self.learners._lib._name, deadline)
                predict_id = new_id()
                self.audit.emit(M.predict_request(
                    request_id=predict_id, learner_id=req.learner_id,
                    model_ref=st.handle.artifact_path, data_ref=self.dataset_ref))
                self.audit.emit(M.predict_result(
                    request_id=predict_id, learner_id=req.learner_id,
                    pred_path=None, status="ok"))

                # Validation error: the only error used for selection and alpha.
                eval_req_id = new_id()
                self.audit.emit(M.evaluate_request(
                    learner_id=req.learner_id, eval_target="learner", split=split, cost={},
                    metrics=("err",),
                    request_id=eval_req_id))
                ev = self.evaluator.score(pred_res.labels, self.y, self.w, val_idx)
                self.audit.emit(M.evaluate_result(
                    request_id=eval_req_id, learner_id=req.learner_id,
                    eval_target="learner", metrics={"err": ev.err}))

                if self.learners.variants is not None:
                    robust_errors = []
                    for variant in self.learners.variants:
                        variant_pred = self.learners.predict(st.handle, variant[val_idx]).labels
                        vals = self.y[val_idx]
                        vw = self.w[val_idx]
                        robust_errors.append(float(np.dot(vw, variant_pred != vals) / vw.sum()))
                    self.audit.emit({"type": "ValidationRobustness", "learner_id": req.learner_id,
                                     "weighted_validation_error": float(np.mean(robust_errors))})

                results.append({"learner_id": req.learner_id, "spec": req.spec,
                                "handle": st.handle, "pred": pred_res.labels,
                                "err": ev.err, "eval": ev, "predict_s": predict_s,
                                "timing_count": timing_count})

            if not results:
                feedback = "All candidates failed:\n" + "\n".join(failures)
                continue

            eligible = [r for r in results if r["err"] < 0.5]
            pool = eligible or results
            lowest = min(r["err"] for r in pool)
            # Prefer faster inference within the configured error tolerance.
            contenders = [r for r in pool if r["err"] <= lowest + self.cfg.speed_tolerance]
            timed = [r for r in contenders if r["predict_s"] is not None]
            cand = min(timed, key=lambda r: (r["predict_s"], r["err"])) if len(timed) == len(contenders) else min(contenders, key=lambda r: r["err"])
            self.audit.emit({"type": "CandidateSelection", "round": t,
                             "learner_id": cand["learner_id"],
                             "validation_error": cand["err"], "predict_s": cand["predict_s"],
                             "timing_samples": cand["timing_count"],
                             "speed_tolerance": self.cfg.speed_tolerance})
            if best is None or cand["err"] < best["err"]:
                best = cand
            if best["err"] <= 0.5 - self.cfg.gamma:      # good enough: stop refining
                break
            feedback = self.evaluator.describe(cand["spec"], cand["eval"])

        return best

    def _propose(self, feedback, deadline):
        raw = None
        try:
            ctx = mp.get_context("fork")
            recv, send = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_proposal_worker,
                               args=(str(Path(getattr(self.cfg, "llm_adapter", None) or
                                                  Path(__file__).with_name("llm.py")).resolve()),
                                     self.history[-5:], feedback, self.cfg.K, send))
            proc.start()
            send.close()
            try:
                if not recv.poll(max(0, deadline - time.monotonic())):
                    raise TimeoutError("proposal deadline exceeded")
                ok, result = recv.recv()
                if not ok:
                    raise RuntimeError(result)
                raw = result
            finally:
                if proc.is_alive():
                    proc.terminate()
                proc.join()
                recv.close()
            specs = parse_specs(raw, self.cfg.K)
            return specs, "llm", raw, "LLM proposal"
        except Exception as e:
            fallback = [DEFAULT_SPECS[i % len(DEFAULT_SPECS)] for i in range(self.cfg.K)]
            return fallback, "fallback", raw, f"fallback: {e}"

    # ---------- weights, alpha, and bookkeeping ----------

    def _update_weights(self, t, best):
        round_dir = os.path.join(self.run_dir, f"round_{t:03d}")
        w_in_sha = sha256_array(self.w)

        e = min(max(best["err"], EPS), 1 - EPS)
        alpha = 0.5 * math.log((1 - e) / e)

        # Update over ALL rows, using predictions on all rows.
        self.w = self.w * np.exp(-alpha * self.y * best["pred"])
        self.w /= self.w.sum()
        np.save(os.path.join(round_dir, "weights_out.npy"), self.w)
        w_out_sha = sha256_array(self.w)

        self.ensemble.append({"alpha": alpha, "handle": best["handle"],
                              "spec": best["spec"], "learner_id": best["learner_id"]})
        self.history.append({"spec": best["spec"], "err": best["err"]})

        self.audit.emit(M.round_update(
            round_t=t, selected_learner_id=best["learner_id"], err_clamped=e,
            alpha=alpha, w_in_sha256=w_in_sha, w_out_sha256=w_out_sha))
        with open(os.path.join(self.run_dir, "manifest.jsonl"), "a") as f:
            f.write(json.dumps({"round": t, "learner_id": best["learner_id"],
                                "spec": best["spec"], "alpha": alpha,
                                "err": best["err"],
                                "model_ref": os.path.relpath(best["handle"].artifact_path, self.run_dir),
                                "model_sha256": best["handle"].sha256}) + "\n")

    def _finalize(self):
        if not self.ensemble or self.X_test is None or self.y_test is None:
            return {"accepted_learners": len(self.ensemble)}
        summary = {}
        for policy, mode in POLICIES.items():
            combine_id = new_id()    # one id shared by the request, the combine, and the result

            self.audit.emit(M.combine_request(
                policy=policy,
                learner_weights=[m["alpha"] for m in self.ensemble],
                prediction_refs=[m["handle"].artifact_path for m in self.ensemble],
                request_id=combine_id))

            y_hat = combine(self.ensemble, self.learners, self.X_test, mode)
            acc = float(np.mean(y_hat == self.y_test))

            self.audit.emit(M.evaluate_result(
                request_id=combine_id,
                learner_id=f"ensemble:{policy}",
                eval_target="ensemble",
                metrics={"test_accuracy": acc}))
            summary[policy] = acc
        return summary
