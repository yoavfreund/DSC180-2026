"""Native Linux Codex policy and no-model, fail-closed access probes."""
import json
import os
from pathlib import Path
import subprocess
import tempfile


def toml(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ",".join(toml(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(key) + "=" + toml(item) for key, item in value.items()) + "}"
    raise TypeError(type(value))


def settings(workspace, private, repo, max_calls):
    workspace, private, repo = map(lambda p: Path(p).resolve(), (workspace, private, repo))
    learner = workspace / "learner"
    return {
        "default_permissions": "learner",
        "approval_policy": "never",
        "cli_auth_credentials_store": "file",
        "model_provider": "openai",
        "web_search": "disabled",
        "project_doc_max_bytes": 0,
        "project_root_markers": [".sirio-run-root"],
        "projects": {str(workspace): {"trust_level": "untrusted"}, str(repo): {"trust_level": "untrusted"}},
        "features": {"apps": False, "plugins": False, "remote_plugin": False,
                     "multi_agent": False, "memories": False, "hooks": False,
                     "shell_snapshot": False, "code_mode_host": True,
                     "multi_agent_v2": False, "browser_use": False,
                     "browser_use_external": False, "browser_use_full_cdp_access": False,
                     "computer_use": False, "in_app_browser": False,
                     "in_app_chat": False, "in_app_local_automation": False,
                     "image_generation": False, "view_image": False,
                     "artifact": False, "workspace_dependencies": False,
                     "skill_search": False, "skill_mcp_dependency_install": False,
                     "plugin_discovery_prefetch": False, "tool_suggest": False,
                     "skip_host_skill_discovery": True},
        "apps": {"_default": {"enabled": False}},
        "history": {"persistence": "none"},
        "permissions": {"learner": {
            "filesystem": {":root": "deny", ":minimal": "read", ":slash_tmp": "deny",
                           str(workspace / "train"): "read",
                           str(workspace / "AGENTS.md"): "read", str(learner): "write"},
            "network": {"enabled": False}}},
        "shell_environment_policy": {"inherit": "none", "set": {
            "PATH": "/usr/bin:/bin", "HOME": str(learner / ".home"),
            "TMPDIR": str(learner / ".tmp"), "LANG": "C.UTF-8"}},
        "mcp_servers": {"validation": {
            "command": "/usr/bin/python3",
            "args": [str(repo / "harness/run_validation_WIN.py"), str(workspace),
                     "--run-name", f"sirio/{workspace.name}", "--require-cascade",
                     "--max-calls", str(max_calls), "--serve"],
            "env": {"DSC180_PRIVATE_WIN": str(private)}, "required": True,
            "enabled_tools": ["run_validation"], "startup_timeout_sec": 60,
            "tools": {"run_validation": {"approval_mode": "approve"}},
            "tool_timeout_sec": 4500}},
    }


class LinuxAgent:
    def __init__(self, workspace, private, repo, max_calls=20, binary=None):
        self.workspace, self.private, self.repo = map(Path, (workspace, private, repo))
        self.binary = Path(binary or os.environ.get("DSC180_CODEX_BIN", self.private / "runtime/codex"))
        if not self.binary.is_file():
            raise SystemExit("Native Linux Codex is required; no Windows fallback. Run in WSL: "
                             "python3 agents/surya/install_runtime.py")
        with self.binary.open("rb") as executable:
            if executable.read(4) != b"\x7fELF":
                raise SystemExit("DSC180_CODEX_BIN must point to a native Linux ELF binary")
        self.home = self.private / "agents/sirio/codex-home"
        self.home.mkdir(parents=True, exist_ok=True)
        self.home.chmod(0o700)
        (self.home / "tmp").mkdir(exist_ok=True)
        self.env = {"CODEX_HOME": str(self.home), "HOME": str(self.home),
                    "PATH": str(self.binary.parent) + ":/usr/bin:/bin", "LANG": "C.UTF-8",
                    "TMPDIR": str(self.home / "tmp")}
        # Authentication uses the isolated client's auth store, never inherited
        # API-key environment variables or the Windows user's home/session state.
        self.policy = settings(self.workspace, self.private, self.repo, max_calls)
        # Codex's Linux sandbox re-execs this immutable runtime inside bwrap.
        # Nested execution still inherits the outer filesystem/network limits.
        self.policy["permissions"]["learner"]["filesystem"][str(self.binary.resolve())] = "read"

    def command(self, *arguments):
        result = [str(self.binary)]
        for key, value in self.policy.items():
            result += ["-c", key + "=" + toml(value)]
        return [*result, *map(str, arguments)]

    def check(self):
        """Exercise the actual Codex command sandbox, without calling a model."""
        learner = self.workspace / "learner"
        for directory in (".tmp", ".home"):
            (learner / directory).mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="access-probe-", dir=self.home) as temp:
            outside = Path(temp) / "outside.txt"
            outside.write_text("private sentinel")
            link = learner / ".access-probe-link"
            if link.exists() or link.is_symlink():
                raise SystemExit("Remove unexpected .access-probe-link before checking")
            link.symlink_to(outside)
            policy_marker = self.workspace / ".access-probe-write"
            if policy_marker.exists():
                link.unlink()
                raise SystemExit("Unexpected .access-probe-write; inspect it before launching")
            policy_marker.write_bytes(b"trusted policy sentinel")
            payload = {"outside": str(outside), "symlink": str(link),
                       "repository": str(self.repo / "DATA_SPLIT.md"),
                       "private_key": str(self.private / "staged/validate/labels.txt"),
                       "training": str(self.workspace / "train/labels.txt"),
                       "train_write": str(self.workspace / "train/.access-probe-write"),
                       "policy_write": str(policy_marker), "metadata": str(self.workspace / "run.json"),
                       "host_process": f"/proc/{os.getpid()}/environ",
                       "learner": str(learner)}
            probe = r'''
import json, os, pathlib, socket, subprocess, sys
p = json.loads(sys.argv[1]); results = {}
def can_read(path):
    try:
        with open(path, 'rb'): pass
        return True
    except OSError: return False
def can_write(path):
    try:
        with open(path, 'w') as f: f.write('probe')
        os.unlink(path)
        return True
    except OSError: return False
for name in ('outside', 'symlink', 'repository', 'private_key'):
    results[name + '_blocked'] = not can_read(p[name])
results['training_readable'] = can_read(p['training'])
results['training_readonly'] = not can_write(p['train_write'])
results['run_metadata_unreadable'] = not can_read(p['metadata'])
# bwrap may expose writable synthetic parent directories. A write there is
# harmless if it cannot modify the real host file; the trusted parent checks.
can_write(p['policy_write'])
results['learner_writable'] = can_write(str(pathlib.Path(p['learner']) / '.access-probe-write'))
results['host_process_environment_blocked'] = not can_read(p['host_process'])
results['credentials_not_inherited'] = not any(k in os.environ for k in ('OPENAI_API_KEY','CODEX_API_KEY'))
try:
    s = socket.socket(); s.settimeout(1)
    results['network_blocked'] = s.connect_ex(('1.1.1.1', 443)) != 0
    s.close()
except OSError: results['network_blocked'] = True
source = pathlib.Path(p['learner']) / '.access-probe.c'
binary = pathlib.Path(p['learner']) / '.access-probe'
try:
    source.write_text('int main(void) { return 0; }\n')
    compiled = subprocess.run(['/usr/bin/cc', str(source), '-o', str(binary)], capture_output=True)
    results['compiler_works'] = compiled.returncode == 0 and subprocess.run([str(binary)]).returncode == 0
finally:
    source.unlink(missing_ok=True); binary.unlink(missing_ok=True)
print(json.dumps(results))
sys.exit(0 if all(results.values()) else 1)
'''
            try:
                probe_env = {**self.env, **self.policy["shell_environment_policy"]["set"]}
                for name in ("OPENAI_API_KEY", "CODEX_API_KEY"):
                    probe_env.pop(name, None)
                result = subprocess.run(self.command("sandbox", "--permission-profile", "learner",
                                        "--cd", learner, "--", "/usr/bin/python3", "-c", probe,
                                        json.dumps(payload)), env=probe_env, cwd=learner,
                                        capture_output=True, text=True, timeout=90)
            finally:
                link.unlink(missing_ok=True)
                policy_intact = policy_marker.read_bytes() == b"trusted policy sentinel" if policy_marker.exists() else False
                policy_marker.unlink(missing_ok=True)
            if result.returncode:
                raise SystemExit("Agent isolation check failed; refusing to launch.\n" + result.stdout + result.stderr)
            try:
                report = json.loads(result.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError):
                raise SystemExit("Agent isolation probe returned no valid report; refusing to launch")
            report["run_metadata_unchanged_on_host"] = policy_intact
            if not report or not all(value is True for value in report.values()):
                raise SystemExit("Agent isolation probe did not pass; refusing to launch")
            return report
