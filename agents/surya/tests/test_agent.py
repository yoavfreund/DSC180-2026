"""Launcher lifecycle tests with no model calls or validation calls."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "agent.py"
spec = importlib.util.spec_from_file_location("sirio_agent", SOURCE)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.here = self.root / "agents/surya"
        self.here.mkdir(parents=True)
        (self.here / "system_prompt.md").write_text("Build your own cascade.")
        self.private = self.root / "private"
        for split in ("train", "validate"):
            data = self.private / "staged" / split
            (data / "images").mkdir(parents=True)
            (data / "labels.txt").write_text("000001 1\n")
        patches = [patch.object(agent, "HERE", self.here),
                   patch.object(agent.stage, "PRIVATE", self.private),
                   patch.object(agent.stage, "WORK", self.root / "unused"),
                   patch.object(agent.stage, "doctor"),
                   patch.object(agent.stage, "build_workspace", side_effect=self.build)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.args = argparse.Namespace(run_id="run-01", reuse=False, validate=False, check=False,
                                       distro="Ubuntu", max_validations=3)

    def build(self, name, run_id, *args):
        workspace = agent.stage.WORK / name / run_id
        (workspace / "learner").mkdir(parents=True)

    def test_prepares_source_under_sirio_and_routes_windows_through_guarded_launcher(self):
        workspace = agent.prepare(self.args)
        self.assertEqual(workspace, self.here / "runs/run-01")
        self.assertFalse(list((workspace / "learner").iterdir()))
        launcher = (workspace / "start-codex.ps1").read_text()
        self.assertIn("wsl.exe", launcher)
        self.assertIn("agent.py", launcher)
        self.assertNotIn("& codex", launcher)
        self.assertTrue((workspace / ".sirio-run-root").exists())
        self.assertEqual(json.loads((workspace / "run.json").read_text())["max_validations"], 3)

    def test_reuse_preserves_source_and_rejects_budget_change(self):
        workspace = agent.prepare(self.args)
        source = workspace / "learner/train.c"
        source.write_text("my own learner")
        with self.assertRaises(SystemExit):
            agent.prepare(self.args)
        self.args.reuse = True
        agent.prepare(self.args)
        self.assertEqual(source.read_text(), "my own learner")
        self.args.max_validations = 4
        with self.assertRaisesRegex(SystemExit, "budget"):
            agent.prepare(self.args)

    def test_run_id_cannot_escape_sirio(self):
        for value in ("../andrew", "/tmp/run", "run/other", ""):
            with self.assertRaises(ValueError):
                agent.run_path(value)

    def test_reuse_keeps_frozen_prompt_after_default_prompt_changes(self):
        workspace = agent.prepare(self.args)
        archived = workspace / 'system_prompt_run01.md'
        self.assertEqual(archived.read_text(), 'Build your own cascade.')
        (self.here / 'system_prompt.md').write_text('A different experiment.')
        self.args.reuse = True
        agent.prepare(self.args)
        self.assertIn('Build your own cascade.', (workspace / 'AGENTS.md').read_text())
        self.assertNotIn('A different experiment.', (workspace / 'AGENTS.md').read_text())
        archived.write_text('Edited after freezing.')
        with self.assertRaisesRegex(SystemExit, 'prompt changed'):
            agent.prepare(self.args)

    def test_native_command_keeps_model_literal_and_does_not_override_read_profile(self):
        from types import SimpleNamespace
        workspace = agent.prepare(self.args)
        self.args.model = "model'withquote"
        runtime = SimpleNamespace(command=lambda *args: list(args))
        command = agent.model_command(runtime, self.args, workspace)
        self.assertIn("--ignore-user-config", command)
        self.assertNotIn("--sandbox", command)
        self.assertIn("model'withquote", command)
        self.assertIn("--output-last-message", command)
        self.assertIn("--ephemeral", command)

    def test_security_policy_denies_host_reads_and_external_tools(self):
        import tomllib
        from agents.surya.security import settings, toml
        workspace = agent.prepare(self.args)
        policy = settings(workspace, self.private, self.root, 3)
        self.assertEqual(policy["approval_policy"], "never")
        self.assertEqual(policy["web_search"], "disabled")
        self.assertFalse(policy["permissions"]["learner"]["network"]["enabled"])
        files = policy["permissions"]["learner"]["filesystem"]
        self.assertEqual(files[":root"], "deny")
        self.assertEqual(files[str(workspace / "learner")], "write")
        self.assertEqual(files[str(workspace / "train")], "read")
        self.assertNotIn(str(workspace), files)
        self.assertTrue(policy["features"]["skip_host_skill_discovery"])
        self.assertTrue(policy["features"]["code_mode_host"])
        self.assertFalse(any(value for key, value in policy["features"].items()
                             if key not in ("skip_host_skill_discovery", "code_mode_host")))
        self.assertEqual(list(policy["mcp_servers"]), ["validation"])
        self.assertEqual(policy["mcp_servers"]["validation"]["enabled_tools"], ["run_validation"])
        self.assertEqual(policy["mcp_servers"]["validation"]["tools"],
                         {"run_validation": {"approval_mode": "approve"}})
        self.assertIn("sirio/run-01", policy["mcp_servers"]["validation"]["args"])
        self.assertEqual(tomllib.loads("policy=" + toml(policy))["policy"], policy)

    def test_launch_stops_before_starting_model_if_isolation_check_fails(self):
        workspace = agent.prepare(self.args)
        self.args.model = None
        with patch.object(agent, "LinuxAgent") as runtime, patch.object(agent.subprocess, "Popen") as process:
            runtime.return_value.check.side_effect = SystemExit("isolation failed")
            with self.assertRaisesRegex(SystemExit, "isolation failed"):
                agent.launch(self.args, workspace)
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
