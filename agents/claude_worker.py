"""One Claude agent session for one attempt. The coordinator starts it and kills it at the deadline.

The agent has NO built-in tools: no file reads, no file writes, no shell. Everything
it needs (task, data summary, accepted rules, its own feedback) is in the prompt.
It acts only through two tools served by this process:

  try_rule(source)     build the rule, report weighted TRAINING error and speed
  submit_rule(source)  the same, then save it as this agent's submission

Only the last submission written before the deadline counts. Validation images
are never in this process.

Usage (normally run by the coordinator):
  python -m agents.claude_worker --system PROMPT.md --brief BRIEF.md --private DIR
      --out DIR --cache DIR --cwd DIR --bar 0.30 [--model M] [--effort E] [--max-turns N]
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage, SystemMessage,
                              TextBlock, ToolUseBlock, UserMessage, create_sdk_mcp_server, query, tool)

from agents.common import TrainData, check_on_train, write_result, write_submission
from harness.build import LearnerError

TRY_TOOL = "mcp__harness__try_rule"
SUBMIT_TOOL = "mcp__harness__submit_rule"
BUILTIN_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "WebSearch",
                 "Agent", "Task", "NotebookEdit", "Skill", "TodoWrite"]
SOURCE_SCHEMA = {"type": "object",
                 "properties": {"source": {"type": "string",
                                           "description": "complete C source defining weak_score"}},
                 "required": ["source"]}


def make_server(data, out_dir, cache_dir, bar, ns_floor, log):
    count = {"n": 0}

    async def check(source, submit):
        count["n"] += 1
        build_dir = out_dir / "builds" / f"{count['n']:02d}"
        try:
            text, _ = await asyncio.to_thread(check_on_train, source, data, build_dir, cache_dir, bar, ns_floor)
        except LearnerError as e:
            log({"event": "tool", "tool": "submit_rule" if submit else "try_rule", "ok": False, "error": str(e)[:300]})
            return {"content": [{"type": "text", "text": f"REJECTED (not submitted): {e}"}], "is_error": True}
        if submit:
            write_submission(out_dir, source)
            text = ("SUBMITTED. If time remains you may submit a better rule; the last submission "
                    "before the deadline counts.\n" + text)
        log({"event": "tool", "tool": "submit_rule" if submit else "try_rule", "ok": True,
             "report": text[:400]})
        return {"content": [{"type": "text", "text": text}]}

    @tool("try_rule", "Build a weak rule and report its weighted training error and speed. "
                      "Does not submit it.", SOURCE_SCHEMA)
    async def try_rule(args):
        return await check(args.get("source", ""), submit=False)

    @tool("submit_rule", "Build a weak rule, report its weighted training error and speed, and save "
                         "it as your submission. The last submission before the deadline counts.",
          SOURCE_SCHEMA)
    async def submit_rule(args):
        return await check(args.get("source", ""), submit=True)

    return create_sdk_mcp_server(name="harness", tools=[try_rule, submit_rule])


async def run(args):
    out_dir = Path(args.out)
    t0 = time.monotonic()
    transcript = open(out_dir / "transcript.jsonl", "a", encoding="utf-8")

    def log(record):
        record["t"] = round(time.monotonic() - t0, 2)
        transcript.write(json.dumps(record, default=str) + "\n")
        transcript.flush()

    data = TrainData.load(args.private)
    server = make_server(data, out_dir, Path(args.cache), args.bar, args.ns_floor, log)
    options = ClaudeAgentOptions(
        cwd=args.cwd,
        system_prompt=Path(args.system).read_text(encoding="utf-8"),
        tools=[],                                   # no built-in tools at all
        allowed_tools=[TRY_TOOL, SUBMIT_TOOL],
        disallowed_tools=BUILTIN_TOOLS,
        mcp_servers={"harness": server},
        strict_mcp_config=True,                     # ignore any other MCP configuration
        setting_sources=[],                         # ignore user/project settings and CLAUDE.md files
        permission_mode="dontAsk",                  # anything not allowed above is denied
        max_turns=args.max_turns,
        model=args.model or None,
        effort=args.effort or None,
    )
    result = {"status": "no result message"}
    async for message in query(prompt=Path(args.brief).read_text(encoding="utf-8"), options=options):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            # Proof of isolation for the audit: the tools this session actually had.
            log({"event": "init", "tools": message.data.get("tools"), "model": message.data.get("model")})
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    log({"event": "text", "text": block.text[:1500]})
                elif isinstance(block, ToolUseBlock):
                    log({"event": "tool_use", "name": block.name,
                         "source_chars": len(str(block.input.get("source", "")))})
        elif isinstance(message, UserMessage):
            pass  # tool results are logged by the tools themselves
        elif isinstance(message, ResultMessage):
            result = {"status": "finished", "subtype": message.subtype, "is_error": message.is_error,
                      "num_turns": message.num_turns, "duration_ms": message.duration_ms,
                      "total_cost_usd": message.total_cost_usd, "usage": message.usage,
                      "errors": message.errors, "api_error_status": message.api_error_status}
            log({"event": "result", **result})
    transcript.close()
    write_result(out_dir, **result)


def main():
    ap = argparse.ArgumentParser()
    for name in ("--system", "--brief", "--private", "--out", "--cache", "--cwd"):
        ap.add_argument(name, required=True)
    ap.add_argument("--bar", type=float, required=True)
    ap.add_argument("--ns-floor", type=float, default=1.0)
    ap.add_argument("--model", default="")
    ap.add_argument("--effort", default="")
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as e:  # a failed session is reported, not raised to the coordinator
        write_result(args.out, status="error", error=f"{type(e).__name__}: {e}"[:1000])
        sys.exit(1)


if __name__ == "__main__":
    main()
