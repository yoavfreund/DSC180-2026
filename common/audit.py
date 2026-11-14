import datetime
import json
import math


def _clean(value):
    """Make a value JSON-safe: numpy scalars to Python, non-finite floats to strings."""
    if hasattr(value, "item") and not isinstance(value, (list, dict)):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


class AuditLog:
    """Append-only JSONL log. One record per event, written by the coordinator process only."""

    def __init__(self, path):
        self.path = path

    def emit(self, type_, **fields):
        record = {"schema_version": 2,
                  "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "type": type_, **_clean(fields)}
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, default=str) + "\n")
