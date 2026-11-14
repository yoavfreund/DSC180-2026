"""Train a boosted face detector whose weak rules are written by competing agents.

Examples:
  # No LLM: the brute-force Haar search is the only agent (a baseline, and a free pipeline test).
  .venv/bin/python main.py --run-name haar-01 --agent base=haar

  # Claude agents, each with its own .md, competing every round, plus the Haar baseline.
  .venv/bin/python main.py --run-name mix-01 --rounds 20 \\
      --agent andrew=claude:agents/prompts/andrew.md --agent base=haar

Agent syntax: NAME=KIND[:ARG]. KIND is claude (ARG = the agent's .md file), haar
(no ARG) or scripted (ARG = test mode; see agents/scripted_worker.py).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from common.data import load_dataset
from config import AgentSpec, Config
from coordinator.coordinator import Coordinator
from coordinator.ladder import flat_ladder, parse_ladder


def parse_agent(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("agent must look like NAME=KIND[:ARG]")
    name, rest = text.split("=", 1)
    kind, _, arg = rest.partition(":")
    try:
        if kind == "claude":
            return AgentSpec(name, kind, prompt=Path(arg))
        if kind == "scripted":
            return AgentSpec(name, kind, mode=arg)
        return AgentSpec(name, kind)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--agent", action="append", type=parse_agent, required=True, dest="agents")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--data", default=".", help="folder that holds train/ and validate/")
    ap.add_argument("--limit-per-class", type=int, help="smoke tests only")
    ap.add_argument("--run-dir", help="default: runs/RUN_NAME")
    ap.add_argument("--work-dir", help="agent session folders, outside the repo "
                                       "(default: ~/dsc180-work/agent-learners/RUN_NAME)")
    ap.add_argument("--ladder", type=parse_ladder, help="explicit bar levels, e.g. 0.30,0.36,0.40,0.43,0.45,0.47,0.49")
    ap.add_argument("--ladder-start", type=float, default=0.30)
    ap.add_argument("--ladder-step", type=float, default=0.03)
    ap.add_argument("--ladder-cap", type=float, default=0.49)
    ap.add_argument("--fails-per-level", type=int, default=5)
    ap.add_argument("--no-carry-over", action="store_true", help="restart the bar at the bottom every round")
    ap.add_argument("--deadline", type=float, default=40.0, help="seconds per attempt")
    ap.add_argument("--no-fallback", action="store_true", help="skip the round instead of a Haar fallback")
    ap.add_argument("--max-llm-sessions", type=int, default=300)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-turns", type=int, default=8)
    args = ap.parse_args()

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("warning: ANTHROPIC_API_KEY is set, so agents bill the API, not your Claude subscription",
              file=sys.stderr)
    ladder = args.ladder or flat_ladder(args.ladder_start, args.ladder_step, args.ladder_cap)
    cfg = Config(rounds=args.rounds, agents=args.agents, ladder=ladder, fails_per_level=args.fails_per_level,
                 carry_over=not args.no_carry_over, deadline_s=args.deadline, fallback=not args.no_fallback,
                 max_llm_sessions=args.max_llm_sessions, model=args.model, effort=args.effort,
                 max_turns=args.max_turns)
    repo = Path(__file__).resolve().parent
    run_dir = Path(args.run_dir) if args.run_dir else repo / "runs" / args.run_name
    work_dir = (Path(args.work_dir) if args.work_dir
                else Path.home() / "dsc180-work" / "agent-learners" / args.run_name)
    if repo == work_dir.resolve() or repo in work_dir.resolve().parents:
        sys.exit("--work-dir must be outside the repository")

    data = load_dataset(args.data, args.limit_per_class)
    print(f"{len(data.train_idx)} training images (agents), {len(data.val_idx)} validation images (hidden); "
          f"ladder {ladder}")
    summary = Coordinator(cfg, data, run_dir, work_dir).run()
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
