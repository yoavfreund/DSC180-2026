#!/usr/bin/env python3
"""Andrew's agent: writes and tunes a Viola-Jones face classifier in C.

Isolation, strongest layer first:
  1. Absence: the agent runs in a staged workspace outside the repo holding
     only train images + labels and an empty learner/. No .git, no validate,
     no test set.
  2. Enforcement: built-in tools limited to Read/Write/Edit/Glob/Grep/Bash,
     only our MCP server, no user settings; Bash runs in Claude Code's sandbox
     with no network and reads limited to the workspace; a PreToolUse hook
     denies file-tool paths outside the workspace and network/git commands.
  3. Instruction: the system prompt states the rules and why.
  4. Audit: every tool call is logged to tool_log.jsonl.
  5. Backstop: the hidden test set, scored once by people at the end.

Validation goes only through the run_validation tool (harness/run_validation.py),
which runs outside the agent's sandbox and returns rounded aggregate metrics.

Usage:
  .venv/bin/python agent.py RUN_ID [--max-turns N] [--max-validations N] [--model M]
  .venv/bin/python agent.py RUN_ID --redteam     # prove the isolation holds
"""
import argparse
import asyncio
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, HookMatcher, ResultMessage,
                              SystemMessage, TextBlock, query)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "harness"))
from run_validation import PRIVATE, Validator, make_server  # noqa: E402

AGENT = "andrew"
WORK = Path(os.environ.get("DSC180_WORK", Path.home() / "dsc180-work"))
BUILTIN_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]
VALIDATION_TOOL = "mcp__harness__run_validation"
PATH_FIELDS = ("file_path", "path", "notebook_path")
# Bash commands that have no business in this task. The sandbox already blocks
# network and outside reads; this keeps the attempts out of the transcript.
# Matched only in command position, so Python's open(...) inside a script is fine.
BLOCKED_COMMANDS = re.compile(
    r"(?:^|[;&|(`\n]|\$\()\s*(?:sudo\s+|env\s+(?:\S+=\S*\s+)*)?"
    r"(?:\S*/)?(git|curl|wget|ssh|scp|sftp|rsync|nc|ncat|telnet|ftp|pip3?|brew|npm|open)(?=\s|$|;|&|\|)")
# Claude Code keeps session scratch dirs for this repo here; they can hold
# copies of data, so sandboxed commands may not read them.
SCRATCH_ROOT = f"/private/tmp/claude-{os.getuid()}/{str(REPO).replace('/', '-')}"


def clip(value, limit=1500):
    """Shorten long strings inside a log record without breaking its JSON."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"...[{len(value) - limit} more chars]"
    if isinstance(value, dict):
        return {k: clip(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [clip(v, limit) for v in value]
    return value


class Guard:
    """PreToolUse/PostToolUse hooks: keep file tools inside the workspace, log everything."""

    def __init__(self, workspace, log_path):
        self.ws = Path(os.path.realpath(workspace))
        self.log_path = log_path

    def _log(self, **record):
        record["time"] = datetime.datetime.now().isoformat(timespec="seconds")
        with open(self.log_path, "a") as f:
            f.write(json.dumps(clip(record), default=str) + "\n")

    def _inside(self, p):
        real = Path(os.path.realpath(self.ws / os.path.expanduser(p)))
        return real == self.ws or self.ws in real.parents

    def _deny_reason(self, tool, args):
        if tool == VALIDATION_TOOL:
            return None
        if tool not in BUILTIN_TOOLS:
            return f"tool {tool} is not allowed"
        if tool == "Bash":
            if args.get("dangerouslyDisableSandbox"):
                return "running outside the sandbox is not allowed"
            if BLOCKED_COMMANDS.search(args.get("command", "")):
                return "network, git and package-manager commands are not allowed"
            return None
        for field in PATH_FIELDS:
            if args.get(field) and not self._inside(args[field]):
                return f"{args[field]} is outside the workspace"
        if tool == "Glob" and (args.get("pattern", "").startswith(("/", "~")) or ".." in args.get("pattern", "")):
            return "glob patterns must stay inside the workspace"
        return None

    async def pre(self, input_data, tool_use_id, context):
        tool, args = input_data["tool_name"], input_data.get("tool_input", {})
        reason = self._deny_reason(tool, args)
        self._log(event="pre", tool=tool, input=args, decision="deny" if reason else "allow", reason=reason)
        if reason:
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                           "permissionDecisionReason": reason}}
        return {}

    async def post(self, input_data, tool_use_id, context):
        self._log(event="post", tool=input_data["tool_name"], response=str(input_data.get("tool_response")))
        return {}

    async def post_failure(self, input_data, tool_use_id, context):
        self._log(event="post_failure", tool=input_data["tool_name"], error=str(input_data.get("error")))
        return {}


def build_options(workspace, validator, guard, prompt_file, max_turns, model):
    ws = str(Path(os.path.realpath(workspace)))
    return ClaudeAgentOptions(
        cwd=ws,
        system_prompt=(HERE / prompt_file).read_text(),
        tools=BUILTIN_TOOLS,
        allowed_tools=BUILTIN_TOOLS + [VALIDATION_TOOL],
        disallowed_tools=["WebFetch", "WebSearch", "Agent", "Task", "NotebookEdit", "Skill"],
        mcp_servers={"harness": make_server(validator)},
        strict_mcp_config=True,
        setting_sources=[],
        permission_mode="dontAsk",
        max_turns=max_turns,
        model=model,
        sandbox={
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "network": {"allowedDomains": []},
            "filesystem": {"denyRead": ["/Users", "/Volumes", SCRATCH_ROOT],
                           "allowRead": [ws], "allowWrite": [ws]},
        },
        hooks={"PreToolUse": [HookMatcher(hooks=[guard.pre])],
               "PostToolUse": [HookMatcher(hooks=[guard.post])],
               "PostToolUseFailure": [HookMatcher(hooks=[guard.post_failure])]},
    )


class Console:
    """Print to the terminal and keep a transcript, saved with the run as console.log."""

    def __init__(self, path):
        self.path = path

    def __call__(self, text):
        print(text, flush=True)
        with open(self.path, "a") as f:
            f.write(text + "\n")


def copy_sources(src, dst, max_bytes=1_000_000):
    """Copy the learner's text files (source, Makefile, notes) into the repo; skip binaries and big files."""
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.is_symlink() or p.stat().st_size > max_bytes:
            continue
        if b"\0" in p.read_bytes()[:8192]:  # compiled programs, models, other binary output
            continue
        target = dst / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)


def check_init(message, say):
    """Abort if the session exposes any tool beyond the allowlist (e.g. user MCP connectors)."""
    tools = set(message.data.get("tools", []))
    extra = tools - set(BUILTIN_TOOLS) - {VALIDATION_TOOL}
    if extra:
        raise SystemExit(f"ABORT: session exposes tools outside the allowlist: {sorted(extra)}")
    if VALIDATION_TOOL not in tools:
        raise SystemExit("ABORT: run_validation tool is missing from the session")
    say(f"[init] tools: {sorted(tools)}")


async def run(args):
    run_name = f"{AGENT}/{args.run_id}"
    subprocess.run([sys.executable, str(REPO / "pipeline" / "stage.py"), "workspace", AGENT, args.run_id],
                   check=True)
    workspace = WORK / AGENT / args.run_id
    validator = Validator(workspace, run_name, max_calls=args.max_validations, require_cascade=True)
    guard = Guard(workspace, validator.run_dir / "tool_log.jsonl")
    say = Console(validator.run_dir / "console.log")
    prompt_file = "redteam_prompt.md" if args.redteam else "system_prompt.md"
    options = build_options(workspace, validator, guard, prompt_file, args.max_turns, args.model)
    task = ("Run the isolation checks described in your instructions and report each result."
            if args.redteam else "Begin. Your workspace is the current directory.")

    async for message in query(prompt=task, options=options):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            check_init(message, say)
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    say(block.text)
        elif isinstance(message, ResultMessage):
            say(f"[done] {message.subtype}, turns={message.num_turns}")

    out = HERE / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    if (workspace / "learner").exists():
        copy_sources(workspace / "learner", out / "learner")
    say(f"[saved] {out}")
    for name in ("runs.jsonl", "tool_log.jsonl", "console.log"):
        if (validator.run_dir / name).exists():
            shutil.copyfile(validator.run_dir / name, out / name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-validations", type=int, default=20)
    ap.add_argument("--model", default=None, help="model alias or ID; default is the CLI's default")
    ap.add_argument("--redteam", action="store_true", help="run the isolation self-test instead of the task")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
