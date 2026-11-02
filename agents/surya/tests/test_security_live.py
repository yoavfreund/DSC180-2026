"""Real Linux sandbox probes. No sign-in, model request, or validation call."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from agents.surya.security import LinuxAgent


class LiveSecurityTests(unittest.TestCase):
    def test_actual_codex_sandbox_blocks_reads_network_and_training_writes(self):
        private = Path(os.environ.get("DSC180_PRIVATE_WIN", Path.home() / "dsc180-private-win"))
        self.check_directory(private)

    def test_windows_backed_repository_filesystem(self):
        self.check_directory(Path(__file__).resolve().parent)

    def check_directory(self, parent):
        private = Path(os.environ.get("DSC180_PRIVATE_WIN", Path.home() / "dsc180-private-win"))
        binary = private / "runtime/codex"
        with tempfile.TemporaryDirectory(prefix="sirio-live-security-", dir=parent) as root:
            root = Path(root)
            workspace = root / "run-01"
            (workspace / "learner").mkdir(parents=True)
            (workspace / "train").mkdir()
            (workspace / "train/labels.txt").write_text("000001 1\n")
            (workspace / "AGENTS.md").write_text("Use only this training data.")
            (workspace / "run.json").write_text("{}")
            fake_private = root / "private"
            key = fake_private / "staged/validate/labels.txt"
            key.parent.mkdir(parents=True)
            key.write_text("private-key-sentinel")
            runtime = LinuxAgent(workspace, fake_private, REPO, binary=binary)
            report = runtime.check()
            self.assertTrue(all(report.values()), report)
            self.assertEqual(key.read_text(), "private-key-sentinel")
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
