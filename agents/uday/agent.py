#!/usr/bin/env python3
"""Uday's agent: writes and tunes a boosted-cascade face classifier in C.

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

The workspace also gets a starter learner skeleton (starter/learner) and a local
checker (starter/check_local.py), both built from train data only; disable with
--no-starter. --variant appends one assigned cascade design from variants.py to
the system prompt.

Usage:
  .venv/bin/python agent.py RUN_ID [--max-turns N] [--max-validations N] [--model M]
                                   [--max-budget-usd D] [--variant NAME] [--no-starter]
  .venv/bin/python agent.py RUN_ID --redteam     # prove the isolation holds
  .venv/bin/python agent.py --list-variants
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
import time
from pathlib import Path

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, HookMatcher, ResultMessage,
                              SystemMessage, TextBlock, ToolResultBlock, ToolUseBlock, UserMessage, query)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "harness"))
from run_validation import PRIVATE, Validator, make_server  # noqa: E402

sys.path.insert(0, str(HERE))
from variants import VARIANTS, variant_note  # noqa: E402

AGENT = "andrew"
STARTER = HERE / "starter"
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

    def __init__(self, workspace, log_path, max_turns=None):
        self.ws = Path(os.path.realpath(workspace))
        self.log_path = log_path
        self.max_turns = max_turns
        self.calls = 0  # tool calls so far; a rough stand-in for turns used (a turn can hold several calls)
        self.writes = {}  # file path -> times written/edited since the last Bash command (loop detector)

    def _nudge(self):
        """Budget warning added to the model's context near the end of the turn budget (approximate)."""
        if not self.max_turns:
            return None
        left = self.max_turns - self.calls
        if left == max(5, self.max_turns // 6) or 0 < left <= 3:
            return (f"Budget notice: about {max(left, 0)} turn(s) may remain. Make sure learner/ builds and "
                    "passes check_local.py, call run_validation if you have not yet, and then write your "
                    "final summary.")
        return None

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

    def _loop_note(self, tool, tool_input):
        """If the same file keeps being rewritten without anything being run, tell the model to stop and test."""
        if tool == "Bash":
            self.writes.clear()
            return None
        if tool in ("Write", "Edit") and isinstance(tool_input, dict):
            path = str(tool_input.get("file_path") or tool_input.get("path") or "")
            self.writes[path] = self.writes.get(path, 0) + 1
            n = self.writes[path]
            if n >= 3:
                return (f"Loop warning: you have written {Path(path).name} {n} times without running any command. "
                        "Stop rewriting it. Run `make 2>&1 | head -30` now, then `python3 check_local.py --quick`, "
                        "and fix whatever the output reports. Fill in real logic, not empty functions or comments.")
        return None

    async def post(self, input_data, tool_use_id, context):
        self.calls += 1
        self._log(event="post", tool=input_data["tool_name"], response=str(input_data.get("tool_response")))
        notes = [n for n in (self._loop_note(input_data["tool_name"], input_data.get("tool_input")), self._nudge()) if n]
        if notes:
            text = " ".join(notes)
            self._log(event="nudge", text=text)
            return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": text}}
        return {}

    async def post_failure(self, input_data, tool_use_id, context):
        self.calls += 1
        self._log(event="post_failure", tool=input_data["tool_name"], error=str(input_data.get("error")))
        return {}


def build_options(workspace, validator, guard, system_prompt, max_turns, model, max_budget_usd=None):
    ws = str(Path(os.path.realpath(workspace)))
    return ClaudeAgentOptions(
        cwd=ws,
        system_prompt=system_prompt,
        tools=BUILTIN_TOOLS,
        allowed_tools=BUILTIN_TOOLS + [VALIDATION_TOOL],
        disallowed_tools=["WebFetch", "WebSearch", "Agent", "Task", "NotebookEdit", "Skill"],
        mcp_servers={"harness": make_server(validator)},
        strict_mcp_config=True,
        setting_sources=[],
        permission_mode="dontAsk",
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
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
        self.t0 = time.monotonic()

    def __call__(self, text):
        minutes, seconds = divmod(int(time.monotonic() - self.t0), 60)
        line = f"[{minutes:02d}:{seconds:02d}] {text}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


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


def load_prompt(args):
    """System prompt text: the redteam prompt, or the task prompt plus the assigned variant."""
    if args.redteam:
        return (HERE / "redteam_prompt.md").read_text()
    return (HERE / "system_prompt.md").read_text() + variant_note(args.variant)


def install_starter(workspace, say):
    """Copy the starter learner and local checker into the workspace (our own files, train-only)."""
    if not STARTER.exists():
        say(f"[starter] {STARTER} not found; continuing with an empty learner/")
        return
    src = STARTER / "learner"
    if src.exists():
        for p in sorted(src.iterdir()):
            if p.is_file() and b"\0" not in p.read_bytes()[:8192]:  # sources only, never compiled binaries
                shutil.copyfile(p, workspace / "learner" / p.name)
    if (STARTER / "check_local.py").exists():
        shutil.copyfile(STARTER / "check_local.py", workspace / "check_local.py")
    say("[starter] installed learner skeleton and check_local.py")


def brief(tool_input, limit=140):
    """One-line summary of a tool call's input for the console transcript."""
    if isinstance(tool_input, dict):
        for key in ("command", "file_path", "path", "pattern"):
            if tool_input.get(key):
                return str(tool_input[key]).replace("\n", " ")[:limit]
    return str(tool_input).replace("\n", " ")[:limit]


def fmt_usage(usage):
    if isinstance(usage, dict):
        keys = ("input_tokens", "output_tokens", "cache_read_input_tokens")
        return " ".join(f"{k.replace('_input_tokens', '').replace('_tokens', '')}={usage[k]}" for k in keys if usage.get(k))
    return ""


def result_text(content, limit):
    """Flatten a tool result (a string or a list of text blocks) to one short line."""
    if isinstance(content, list):
        content = " ".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
    text = " ".join(str(content or "").split())
    return text if len(text) <= limit else text[:limit] + " ..."


def unchanged_from_starter(workspace):
    """Names of the starter's train.c / predict.c that are still byte-identical (apart from whitespace
    at the ends) in the workspace; these mean the agent never replaced the placeholder."""
    same = []
    for name in ("train.c", "predict.c"):
        a, b = STARTER / "learner" / name, workspace / "learner" / name
        if a.exists() and b.exists() and a.read_text().strip() == b.read_text().strip():
            same.append(name)
    return same


def check_init(message, say):
    """Abort if the session exposes any tool beyond the allowlist (e.g. user MCP connectors)."""
    tools = set(message.data.get("tools", []))
    extra = tools - set(BUILTIN_TOOLS) - {VALIDATION_TOOL}
    if extra:
        raise SystemExit(f"ABORT: session exposes tools outside the allowlist: {sorted(extra)}")
    if VALIDATION_TOOL not in tools:
        raise SystemExit("ABORT: run_validation tool is missing from the session")
    model = message.data.get("model")
    say(f"[init] model: {model or 'unknown'}")
    say(f"[init] tools: {sorted(tools)}")


async def run(args):
    run_name = f"{AGENT}/{args.run_id}"
    subprocess.run([sys.executable, str(REPO / "pipeline" / "stage.py"), "workspace", AGENT, args.run_id],
                   check=True)
    workspace = WORK / AGENT / args.run_id
    validator = Validator(workspace, run_name, max_calls=args.max_validations, require_cascade=True)
    guard = Guard(workspace, validator.run_dir / "tool_log.jsonl", max_turns=args.max_turns)
    say = Console(validator.run_dir / "console.log")
    if not args.redteam:
        say(f"[variant] {args.variant}")
        if not args.no_starter:
            install_starter(workspace, say)
    options = build_options(workspace, validator, guard, load_prompt(args), args.max_turns, args.model,
                            args.max_budget_usd)
    task = ("Run the isolation checks described in your instructions and report each result."
            if args.redteam else "Begin. Your workspace is the current directory.")

    async for message in query(prompt=task, options=options):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            check_init(message, say)
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    say(block.text)
                elif isinstance(block, ToolUseBlock):
                    say(f"  -> {block.name}: {brief(block.input)}")
            tokens = fmt_usage(getattr(message, "usage", None))
            if tokens:
                say(f"  (tokens {tokens})")
        elif isinstance(message, UserMessage) and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    if block.is_error:
                        say(f"  <- TOOL ERROR: {result_text(block.content, 400)}")
                    else:
                        say(f"  <- {result_text(block.content, 160)}")
        elif isinstance(message, ResultMessage):
            say(f"[done] {message.subtype}, turns={message.num_turns}")
            extra = [f"{name}={getattr(message, name)}" for name in ("duration_ms", "total_cost_usd")
                     if getattr(message, name, None) is not None]
            tokens = fmt_usage(getattr(message, "usage", None))
            if extra or tokens:
                say("[usage] " + " ".join(extra + ([f"tokens {tokens}"] if tokens else [])))

    if not args.redteam and not args.no_starter:
        same = unchanged_from_starter(workspace)
        if same:
            say(f"[WARNING] learner/{' and learner/'.join(same)} still match the starter placeholder. "
                "The agent's edits did not land, so ignore its summary: the saved learner is not its algorithm.")

    out = HERE / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    if (workspace / "learner").exists():
        copy_sources(workspace / "learner", out / "learner")
    (out / "variant.txt").write_text(f"{args.variant}\n")
    say(f"[saved] {out}")
    for name in ("runs.jsonl", "tool_log.jsonl", "console.log"):
        if (validator.run_dir / name).exists():
            shutil.copyfile(validator.run_dir / name, out / name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-validations", type=int, default=20)
    ap.add_argument("--model", default=None, help="model alias or ID; default is the CLI's default")
    ap.add_argument("--max-budget-usd", type=float, default=None,
                    help="stop the run once its API cost reaches this many US dollars")
    ap.add_argument("--redteam", action="store_true", help="run the isolation self-test instead of the task")
    ap.add_argument("--variant", default="free", choices=sorted(VARIANTS), help="assigned cascade design (variants.py)")
    ap.add_argument("--no-starter", action="store_true", help="start from an empty learner/ and no check_local.py")
    ap.add_argument("--list-variants", action="store_true", help="print the available variants and exit")
    args = ap.parse_args()
    if args.list_variants:
        for name, text in VARIANTS.items():
            print(f"{name:18} {text[:90] or '(no assigned design)'}")
        return
    if not args.run_id:
        ap.error("run_id is required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()