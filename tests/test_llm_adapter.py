import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from common.audit import AuditLog
from config import Config
from coordinator.coordinator import Coordinator


class AdapterTest(unittest.TestCase):
    def coordinator(self, directory, adapter):
        return Coordinator(Config(K=2, llm_adapter=str(adapter)), None, None,
                           AuditLog(Path(directory) / "audit"), np.zeros((2, 1)),
                           np.array([1, -1]), None, None, "data", directory)

    def test_custom_adapter_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.py"
            path.write_text('def propose_raw(history, feedback, k, timeout=5):\n'
                            '    assert k == 2 and feedback == "feedback"\n'
                            '    return \'[{"family":"stump","features":"lbp"},'
                            '{"family":"stump","features":"pixels"}]\'\n')
            specs, source, raw, _ = self.coordinator(tmp, path)._propose("feedback", time.monotonic() + 5)
            self.assertEqual(source, "llm")
            self.assertEqual(specs[0]["features"], "lbp")
            self.assertEqual(len(json.loads(raw)), 2)

    def test_adapter_failure_uses_fallback_without_exposing_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.py"
            path.write_text('def propose_raw(history, feedback, k, timeout=5):\n'
                            '    raise ValueError("private secret")\n')
            specs, source, _, why = self.coordinator(tmp, path)._propose(None, time.monotonic() + 5)
            self.assertEqual(source, "fallback")
            self.assertEqual(len(specs), 2)
            self.assertNotIn("private secret", why)


if __name__ == "__main__":
    unittest.main()
