import datetime
import json
import threading


class AuditLog:
    """Append JSON messages from concurrent threads in a single process."""
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()

    def emit(self, message):
        record = {"schema_version": 1,
                  "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  **message}
        with self._lock, open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def summary(self, limit=18):
        """Return bounded aggregate records, excluding raw outputs and secrets."""
        try:
            with self._lock, open(self.path, encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
        except FileNotFoundError:
            return []
        allowed = {"CoordinatorDecision": ("source", "actions"),
                   "TrainResult": ("learner_id", "status", "elapsed_s"),
                   "EvaluateResult": ("learner_id", "eval_target", "metrics"),
                   "ValidationRobustness": ("learner_id", "weighted_validation_error"),
                   "CandidateSelection": ("learner_id", "validation_error", "predict_s"),
                   "RoundSkipped": ("round", "reason"),
                   "RefinementStopped": ("round", "reason")}
        selected = [r for r in records if r.get("type") in allowed][-limit:]
        return [{"type": r["type"], **{k: r[k] for k in allowed[r["type"]] if k in r}}
                for r in selected]
