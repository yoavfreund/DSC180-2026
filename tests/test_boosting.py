import json
import math
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import numpy as np

from common.audit import AuditLog
from common.splits import seeded_split
from config import Config
from coordinator.coordinator import Coordinator, _train_until


class SlowLearner:
    def train(self, request):
        time.sleep(5)
        return request


class BoostingProtocolTest(unittest.TestCase):
    def test_repeated_fallback_is_not_retrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            learners = SimpleNamespace(variants=None, _lib=SimpleNamespace(_name="unused"),
                                       predict=lambda handle, X: SimpleNamespace(labels=np.ones(len(X), dtype=int)))
            evaluator = SimpleNamespace(score=lambda *args: SimpleNamespace(err=.495),
                                        describe=lambda *args: "weighted validation error: .495")
            coordinator = Coordinator(Config(K=1, max_refine=3), learners, evaluator,
                                      AuditLog(Path(tmp, "audit")), np.zeros((10, 714)),
                                      np.ones(10, dtype=int), None, None, "data", tmp)
            status = SimpleNamespace(ok=True, handle=SimpleNamespace(artifact_path="model", sha256="hash"),
                                     error=None, elapsed_s=.1)
            with mock.patch.object(coordinator, "_propose", return_value=([{"family": "stump"}], "fallback", None, "fallback")), \
                 mock.patch("coordinator.coordinator._train_until", return_value=[status]) as train, \
                 mock.patch("coordinator.coordinator._benchmark", return_value=(.01, 3)):
                coordinator._round(0)
            self.assertEqual(train.call_count, 1)
            records = [json.loads(line) for line in Path(tmp, "audit").read_text().splitlines()]
            self.assertIn("RefinementStopped", [r["type"] for r in records])

    def test_training_worker_is_stopped_at_deadline(self):
        start = time.monotonic()
        result = _train_until(SlowLearner(), ["candidate"], start + .2)
        self.assertEqual(result, [None])
        self.assertLess(time.monotonic() - start, 2)

    def test_validation_error_drives_all_row_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "round_000").mkdir()
            labels = np.array([1, -1, 1, -1])
            coordinator = Coordinator(Config(T=1), None, None, AuditLog(Path(tmp, "audit")),
                                      np.zeros((4, 1)), labels, None, None, "data", tmp)
            best = {"err": .25, "pred": np.array([1, 1, -1, -1]),
                    "handle": SimpleNamespace(artifact_path=str(Path(tmp, "round_000", "model")), sha256="hash"),
                    "spec": {}, "learner_id": "candidate"}
            coordinator._update_weights(0, best)
            alpha = .5 * math.log(3)
            expected = np.exp(-alpha * labels * best["pred"])
            expected /= expected.sum()
            np.testing.assert_allclose(coordinator.w, expected)
            self.assertAlmostEqual(coordinator.w.sum(), 1)
            self.assertAlmostEqual(coordinator.ensemble[0]["alpha"], alpha)
            log = json.loads(Path(tmp, "audit").read_text())
            self.assertEqual(log["err_clamped"], .25)

    def test_seeded_round_splits(self):
        train, val = seeded_split(100, 42, .67)
        train2, val2 = seeded_split(100, 43, .67)
        self.assertEqual((len(train), len(val)), (33, 67))
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertFalse(np.array_equal(val, val2))
        np.testing.assert_array_equal(train, seeded_split(100, 42, .67)[0])


if __name__ == "__main__":
    unittest.main()
