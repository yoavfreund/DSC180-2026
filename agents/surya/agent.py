#!/usr/bin/env python3
"""Surya's Codex launcher, based on Andrew's agent.py lifecycle.

Run with Ubuntu WSL's Python from the repo root:
  python3 agents/surya/agent.py run-01 --prepare-only
  python3 agents/surya/agent.py run-01 --reuse

The editable run lives in agents/surya/runs/<run>/, with source in learner/.
Native Linux Codex runs with a deny-by-default file policy and no tool network.
The trusted validation MCP runs outside that policy with a fixed interface.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from pipeline import stage_WIN as stage  # noqa: E402
from agents.surya.security import LinuxAgent  # noqa: E402

AGENT = "sirio"


def run_path(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run ID must contain only letters, digits, underscores and hyphens")
    return HERE / "runs" / run_id


def windows_path(path):
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def ps_literal(value):
    return "'" + value.replace("'", "''") + "'"


def write_launchers(workspace, args):
    """Route Windows shortcuts back through the protected Linux launcher."""
    command = ["-d", args.distro, "--", "python3", str(HERE / "agent.py"), args.run_id,
               "--max-validations", str(args.max_validations)]
    script = ("param([switch]$Check)\n$ErrorActionPreference = 'Stop'\n"
              "$arguments = @(" + ",".join(map(ps_literal, command)) + ")\n"
              "if ($Check) { $arguments += '--check' } else { $arguments += '--reuse' }\n"
              "& wsl.exe @arguments\nexit $LASTEXITCODE\n")
    for name in ("start-codex.ps1", "run-codex.ps1"):
        (workspace / name).write_text(script)
    (workspace / ".codex").mkdir(exist_ok=True)
    (workspace / ".codex/config.toml").write_text(
        "# Run agent.py or start-codex.ps1 to apply the protected native Linux policy.\n"
        "# Opening this folder directly in a desktop chat does not apply that policy.\n")
    (workspace / ".sirio-run-root").write_text("Protected launcher project boundary.\n")


def model_command(runtime, args, workspace):
    # All security policy comes from trusted code as CLI overrides. Never pass
    # --sandbox: it would replace the stronger named read-restriction profile.
    command = runtime.command("exec", "--ignore-user-config", "--ephemeral", "--cd", workspace / "learner",
                              "--skip-git-repo-check", "--json", "--color", "never",
                              "--output-last-message", workspace / "final.md")
    if args.model:
        command += ["--model", args.model]
    continuation = (" Continue the implementation already present in this learner directory. "
                    "Inspect its source, README, and local results before making changes. "
                    "Preserve working code and finish any missing validation through the validation tool. "
                    "Do not restart from scratch or repeat local training unless a concrete issue requires it."
                    if args.reuse else "")
    command += [(workspace / "AGENTS.md").read_text() +
                "\nBegin. Your current directory is already learner/. Write the Makefile and source here, "
                "and read training data from ../train/. "
                "Use the validation tool within its budget and summarize the final design and results." + continuation]
    return command


def prepare(args):
    workspace = run_path(args.run_id)
    stage.doctor()
    for split in ("train", "validate"):
        directory = stage.PRIVATE / "staged" / split
        if not (directory / "images").is_dir() or not (directory / "labels.txt").is_file():
            raise SystemExit("Staged data is missing. Run: python3 pipeline/stage_WIN.py prepare")
    if workspace.exists():
        if not (args.reuse or args.validate or args.check):
            raise SystemExit(f"{workspace} already exists. Use --reuse to continue its learner.")
        if (workspace.is_symlink() or (workspace / "learner").is_symlink()
                or (workspace / "train").is_symlink() or not (workspace / "learner").is_dir()):
            raise SystemExit("Existing workspace must be a regular run directory with learner/.")
        # A resumed run must keep its validation budget, rather than resetting it.
        metadata_path = workspace / "run.json"
        if not metadata_path.exists():
            raise SystemExit("Existing workspace is missing run.json; inspect it before reuse.")
        metadata = json.loads(metadata_path.read_text())
        if metadata["max_validations"] != args.max_validations:
            raise SystemExit("--max-validations must match the budget recorded in run.json")
    else:
        if args.validate or args.check:
            raise SystemExit("Run does not exist; create it with --prepare-only first")
        # The staging helper lays out WORK/name/run: HERE/runs/<run>.
        stage.WORK = HERE
        stage.build_workspace("runs", args.run_id, args.distro, args.max_validations)
        metadata = {"agent": AGENT, "run_id": args.run_id,
                    "max_validations": args.max_validations, "distro": args.distro}
        (workspace / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Reusing a run must not silently replace its instructions with a later
    # experiment's default prompt. The archived source is immutable per run.
    prompt_path = workspace / ("system_prompt_" + args.run_id.replace("-", "") + ".md")
    if not prompt_path.exists():
        prompt_path.write_bytes((HERE / "system_prompt.md").read_bytes())
    prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    if metadata.get("prompt_sha256", prompt_hash) != prompt_hash:
        raise SystemExit("Run prompt changed after preparation; use a new run ID")
    metadata.update(prompt_file=prompt_path.name, prompt_sha256=prompt_hash)
    (workspace / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_launchers(workspace, args)
    prompt = prompt_path.read_text()
    prompt += ("\n## Protected Linux commands\n\n"
               "You run natively inside Ubuntu WSL, with learner/ as your current directory. "
               "Compile with `make`; training data is at `../train/`.\n"
               "Use the validation MCP server's run_validation tool for hidden validation.\n"
               f"Your total validation budget for this run is {args.max_validations} calls.\n")
    (workspace / "AGENTS.md").write_text(prompt)
    return workspace


def save_validation_log(workspace):
    source = stage.PRIVATE / "runs" / AGENT / workspace.name / "runs.jsonl"
    if source.is_file():
        shutil.copyfile(source, workspace / "runs.jsonl")


def launch(args, workspace):
    runtime = LinuxAgent(workspace, stage.PRIVATE, REPO, args.max_validations)
    report = runtime.check()
    (workspace / "security_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Agent access checks passed: " + ", ".join(report), flush=True)
    command = model_command(runtime, args, workspace)
    # Keep the original's console/transcript lifecycle; JSON events record
    # tool activity while the native OS sandbox enforces access boundaries.
    with open(workspace / "console.log", "a") as console, open(workspace / "tool_log.jsonl", "a") as tools:
        process = subprocess.Popen(command, env=runtime.env, cwd=workspace / "learner",
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                console.write(line)
                console.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type", "").startswith("item."):
                    tools.write(json.dumps(event) + "\n")
                    tools.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            return 130
        finally:
            save_validation_log(workspace)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--max-validations", type=int, default=20)
    parser.add_argument("--model", help="optional Codex model ID; otherwise use the CLI default")
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--reuse", action="store_true", help="continue editing an existing learner in a new session")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="create an empty learner workspace without a model call")
    mode.add_argument("--validate", action="store_true", help="validate an existing learner once, using its call budget")
    mode.add_argument("--check", action="store_true", help="probe actual agent isolation and inspect MCP configuration without a model call")
    mode.add_argument("--login", action="store_true", help="sign in to the isolated native Linux Codex client")
    args = parser.parse_args()
    if args.max_validations < 1:
        parser.error("--max-validations must be positive")
    if args.login:
        runtime = LinuxAgent(HERE / "runs/auth", stage.PRIVATE, REPO)
        return subprocess.call(runtime.command("login", "--device-auth"), env=runtime.env)
    if not args.run_id:
        parser.error("run_id is required unless --login is used")
    try:
        workspace = prepare(args)
    except ValueError as error:
        parser.error(str(error))
    if args.prepare_only:
        print(f"Your implementation belongs in: {windows_path(workspace / 'learner')}")
        return 0
    if args.check:
        runtime = LinuxAgent(workspace, stage.PRIVATE, REPO, args.max_validations)
        report = runtime.check()
        (workspace / "security_check.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return subprocess.call(runtime.command("mcp", "get", "validation", "--json"), env=runtime.env)
    if args.validate:
        from harness.run_validation_WIN import Validator
        validator = Validator(workspace, f"{AGENT}/{args.run_id}",
                              max_calls=args.max_validations, require_cascade=True)
        print(validator.run())
        save_validation_log(workspace)
        return 0
    return launch(args, workspace)


if __name__ == "__main__":
    sys.exit(main())
