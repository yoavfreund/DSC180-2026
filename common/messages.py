# common/messages.py
# Schema message builders. Each returns a dict; AuditLog.emit adds the envelope.
from dataclasses import asdict, is_dataclass
from typing import Optional

from common.hashing import new_id


def _as_dict(m):
    if m is None:
        return {}
    return asdict(m) if is_dataclass(m) else dict(m)


def coordinator_decision(rationale, source, specs, combiner, feedback_ref,
                         raw_llm_output, budget=None):
    if source not in ("llm", "rule", "fallback"):
        raise ValueError(f"bad decision source {source!r}")
    return {
        "type": "CoordinatorDecision",
        "decision_id": new_id(),
        "rationale": rationale,
        "source": source,
        "actions": {
            "methods": sorted({s["family"] for s in specs}),
            "n_learners": len(specs),
            "rounds": None,
            "combiner": combiner,
            "budget": budget or {},
            "candidates": specs,              # extension: the exact specs proposed
        },
        "feedback_ref": feedback_ref,         # extension: evaluator text fed back in
        "raw_llm_output": raw_llm_output,     # extension: needed for replay
    }


def train_request(request_id, learner_id, spec, data_ref, w_sha256, round_t):
    return {
        "type": "TrainRequest",
        "request_id": request_id,
        "learner_id": learner_id,
        "method": spec["family"],
        "data_ref": data_ref,
        "sample_weights": {"ref": f"round_{round_t:03d}/weights_in.npy",
                           "sha256": w_sha256},
        "hyperparams": {k: v for k, v in spec.items() if k != "family"},
    }


def train_result(request_id, learner_id, status):
    # status is a TrainStatus from LearnerAgent.train
    return {
        "type": "TrainResult",
        "request_id": request_id,
        "learner_id": learner_id,
        "model_ref": status.handle.artifact_path if status.ok else None,
        "model_sha256": status.handle.sha256 if status.ok else None,
        "status": "ok" if status.ok else "error",
        "error": status.error,
        "elapsed_s": status.elapsed_s,
    }


def predict_request(request_id, learner_id, model_ref, data_ref):
    return {
        "type": "PredictRequest",
        "request_id": request_id,
        "learner_id": learner_id,
        "model_ref": model_ref,
        "data_ref": data_ref,
    }


def predict_result(request_id, learner_id, pred_path, status):
    return {
        "type": "PredictResult",
        "request_id": request_id,
        "learner_id": learner_id,
        "predictions": pred_path,
        "status": status,                     # "ok" | "error"
    }


def evaluate_request(learner_id, eval_target, split, cost, request_id=None,
                     metrics=("err", "fpr", "fnr", "w_pos_val")):
    return {
        "type": "EvaluateRequest",
        "request_id": request_id or new_id(),
        "learner_id": learner_id,
        "eval_target": eval_target,           # "learner" | "ensemble"
        "metrics": list(metrics),
        "split_protocol": split,
        "cost": cost,
    }


def evaluate_result(request_id, learner_id, eval_target, metrics):
    return {
        "type": "EvaluateResult",
        "request_id": request_id,
        "learner_id": learner_id,
        "eval_target": eval_target,
        "metrics": _as_dict(metrics),
    }


def combine_request(policy, learner_weights, prediction_refs, request_id=None):
    return {
        "type": "CombineRequest",
        "request_id": request_id or new_id(),
        "policy": policy,                     # majority | soft_vote | adaboost_weighted
        "learner_weights": learner_weights,
        "prediction_refs": prediction_refs,
    }


def combine_result(request_id, out_path, out_sha256, ok):
    return {
        "type": "CombineResult",
        "request_id": request_id,
        "predictions": out_path,
        "predictions_sha256": out_sha256,
        "status": "ok" if ok else "error",
    }


def round_update(round_t, selected_learner_id, err_clamped, alpha,
                 w_in_sha256, w_out_sha256):
    # Extension to the schema: one record per accepted round.
    return {
        "type": "RoundUpdate",
        "round": round_t,
        "selected_learner_id": selected_learner_id,
        "err_clamped": err_clamped,
        "alpha": alpha,
        "weights_in_sha256": w_in_sha256,
        "weights_out_sha256": w_out_sha256,
    }
