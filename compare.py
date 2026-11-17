"""Compare agent prompts under identical settings.

  # run every setup (skips runs that already finished), then print the tables
  .venv/bin/python compare.py run --name try1                       # small: 3 rounds, 1 repeat, 10 sessions/prompt
  .venv/bin/python compare.py run --name try2 --prompts sohan        # one prompt only
  .venv/bin/python compare.py run --name main --rounds 25 --max-sessions 100   # the main run
  # tables only, from finished runs
  .venv/bin/python compare.py report --name cmp1

Setups, each a normal main.py run under runs/:
  NAME-haar             brute-force Haar only (deterministic, so run once): the bar to beat
  NAME-team-rK          every prompt together, and the Haar search competes every round, repeat K
  NAME-agents-rK        every prompt together, no Haar search competing. The Haar search runs only
                        as the fallback, after every agent has failed the top of the ladder (0.49).
  NAME-agents-only-rK   every prompt together and NO Haar search at all (--no-fallback). A round
                        where every agent fails the top of the ladder is skipped; two skips in a
                        row end the run ("stalled"), and the report says so.
  NAME-solo-P-rK        only with --solo: prompt P alone against the Haar search, repeat K

Every run uses the same shared settings (SHARED below, plus anything after --), so the only
difference between setups is which agents compete. Runs go one at a time, because rules are timed
on this machine and parallel runs would slow each other. Compare runs from one machine only.

Session cap: --max-sessions is per prompt, so a run with N prompts may use N times as many
Claude sessions (every attempt starts one session per prompt). The Haar search uses none.
If Claude itself fails (usage limit, API errors), the run stops with that stop reason and the
comparison stops too, so no lost session is counted as a lost round.

The tables:
  1. Detector quality per setup, on the audit images (held out from everything, so honest):
     detection at 1%, 5% and 10% false positives and AUC, rules accepted, rules from the Haar
     fallback, total ns, cost, and why the run ended.
  2. Per prompt, head to head with the Haar search on the same weights (team runs, and solo runs
     if any): in how many rounds the prompt's best rule had more progress per ns than the search's.
  3. Per prompt, each multi-agent setup: rounds won and its share of the final vote weight (alpha).
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from common.metrics import FP_RATES

REPO = Path(__file__).resolve().parent
PROMPTS = REPO / "agents" / "prompts"
SHARED = ["--audit-frac", "0.2", "--weight-cap", "10", "--confidence", "1",      # README's tested best
          "--fails-per-level", "2", "--augment", "flip,rotate,zoom,shift"]
HAAR = "base"
SESSION_FAILED = "Claude sessions failed"   # start of the stop reason when Claude itself fails


def setups(name, prompts, repeats, solo=False):
    """(run name, agent args, extra main.py flags) for every run in the comparison."""
    out = [(f"{name}-haar", [f"{HAAR}=haar"], [])]
    for k in range(1, repeats + 1):
        claude = [f"{p}=claude:{PROMPTS / (p + '.md')}" for p in prompts]
        if solo:
            for p in prompts:
                out.append((f"{name}-solo-{p}-r{k}", [f"{p}=claude:{PROMPTS / (p + '.md')}", f"{HAAR}=haar"], []))
        out.append((f"{name}-team-r{k}", claude + [f"{HAAR}=haar"], []))
        out.append((f"{name}-agents-r{k}", claude, []))                         # Haar only as the fallback
        out.append((f"{name}-agents-only-r{k}", claude, ["--no-fallback"]))     # no Haar at all
    return out


def session_cap(agents, per_prompt):
    """Claude sessions allowed in one run: per_prompt for each Claude agent in it."""
    return per_prompt * max(1, sum("=claude:" in a for a in agents))


def run_all(args, extra):
    for p in args.prompts:
        if not (PROMPTS / f"{p}.md").is_file():
            sys.exit(f"no prompt file agents/prompts/{p}.md")
    todo = setups(args.name, args.prompts, args.repeats, args.solo)
    total = sum(session_cap(agents, args.max_sessions) for _, agents, _ in todo
                if any("=claude:" in a for a in agents))
    print(f"{len(todo)} runs, {args.rounds} rounds each; at most {total} Claude sessions in total "
          f"({args.max_sessions} per prompt per run)")
    for i, (run, agents, flags) in enumerate(todo, 1):
        summary = REPO / "runs" / run / "summary.json"
        if summary.is_file():
            print(f"[{i}/{len(todo)}] {run}: done already, skipped")
            continue
        if (REPO / "runs" / run).exists():
            print(f"[{i}/{len(todo)}] {run}: unfinished folder from a stopped run; delete runs/{run} to redo it")
            continue
        cmd = [sys.executable, str(REPO / "main.py"), "--run-name", run, "--rounds", str(args.rounds)]
        cmd += [x for a in agents for x in ("--agent", a)] + SHARED + flags
        cmd += ["--max-llm-sessions", str(session_cap(agents, args.max_sessions))] + extra
        print(f"[{i}/{len(todo)}] {run}", flush=True)
        if subprocess.run(cmd, cwd=REPO).returncode != 0:
            print(f"  {run} failed; continuing", file=sys.stderr)
            continue
        reason = json.loads(summary.read_text()).get("stop_reason", "") if summary.is_file() else ""
        if reason.startswith(SESSION_FAILED):
            sys.exit(f"  {run}: {reason}\nStopping the comparison. Delete runs/{run}, then run the same "
                     f"command again later: finished runs are skipped.")


def load(name):
    runs = {}
    for f in sorted((REPO / "runs").glob(f"{name}-*/summary.json")):
        runs[f.parent.name[len(name) + 1:]] = json.loads(f.read_text())
    return runs


def mean(xs, digits=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    m = statistics.fmean(xs)
    return f"{m:.{digits}f}" if len(xs) == 1 else f"{m:.{digits}f} ± {statistics.pstdev(xs):.{digits}f}"


def table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def head_to_head(run_dir):
    """Per round, each Claude agent's best scored rule against the Haar search's rule on the same
    weights, by progress per ns. Returns {agent: [rounds won, rounds compared]}."""
    best = {}                                          # (round, agent) -> best progress per ns
    for line in (run_dir / "audit.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("type") == "AgentResult" and e.get("status") == "scored":
            key = (e["round"], e["agent"])
            best[key] = max(best.get(key, float("-inf")), e["progress_per_ns"])
    out = {}
    for (t, agent), ppn in best.items():
        if agent == HAAR or (t, HAAR) not in best:
            continue
        won, n = out.get(agent, [0, 0])
        out[agent] = [won + (ppn > best[(t, HAAR)]), n + 1]
    return out


def report(name):
    runs = load(name)
    if not runs:
        sys.exit(f"no finished runs named {name}-* under runs/")
    groups = {}
    for key, s in runs.items():
        groups.setdefault(key.rsplit("-r", 1)[0] if key != "haar" else "haar", []).append((key, s))

    def det(s, r):   # None for runs made before this rate was recorded
        return (s.get("audit_detection_at_fp") or {}).get(str(r))

    order = {"haar": 0, "team": 1, "agents": 2, "agents-only": 3}
    rows = []
    for g in sorted(groups, key=lambda g: (order.get(g, 4), g)):
        ss = [s for _, s in groups[g]]
        rows.append([g, len(ss)] + [mean([det(s, r) for s in ss]) for r in FP_RATES]
                    + [mean([s.get("audit_auc") for s in ss]),
                       mean([s["accepted"] for s in ss], 0), mean([s.get("fallback_rules", 0) for s in ss], 0),
                       mean([s.get("ns_total") for s in ss]),
                       f"{sum(a['cost_usd'] for s in ss for a in s['agents'].values()):.2f}",
                       "; ".join(sorted({s.get("stop_reason", "?") for s in ss}))])
    out = ["## 1. Detector quality (audit images)",
           table(["setup", "runs"] + [f"detection @{r:.0%} FP" for r in FP_RATES]
                 + ["AUC", "rules", "from Haar fallback", "total ns", "cost $", "why it ended"], rows)]

    h2h = {}
    for g in groups:
        if g == "team" or g.startswith("solo-"):
            for key, _ in groups[g]:
                for agent, (won, n) in head_to_head(REPO / "runs" / f"{name}-{key}").items():
                    h2h.setdefault((g if g == "team" else "solo", agent), []).append((won, n))
    if h2h:
        rows = []
        for (g, agent), xs in sorted(h2h.items()):
            won, n = sum(x[0] for x in xs), sum(x[1] for x in xs)
            rows.append([agent, g, won, n, f"{won / n:.3f}" if n else "-"])
        out += ["## 2. Head to head with the Haar search, same round and weights (more progress per ns)",
                table(["prompt", "setup", "rounds beaten", "rounds compared", "share"], rows)]

    n = 3
    for g, title in (("team", "Team runs, with the Haar search competing"),
                     ("agents", "Agents runs, Haar search only as the fallback"),
                     ("agents-only", "Agents-only runs, no Haar search at all")):
        team = [s for _, s in groups.get(g, [])]
        if not team:
            continue
        names = sorted({a for s in team for a in s["agents"]}, key=lambda a: (a == HAAR, a))
        rows = []
        for a in names:
            st = [s["agents"].get(a, {}) for s in team]
            alpha = [x.get("alpha_total", 0) / max(sum(v["alpha_total"] for v in s["agents"].values()), 1e-12)
                     for x, s in zip(st, team)]
            rows.append([a, sum(x.get("wins", 0) for x in st), mean(alpha), sum(x.get("rejected", 0) for x in st)])
        fallback = sum(s.get("fallback_rules", 0) for s in team)
        if fallback:   # rounds where every agent failed the top of the ladder
            rows.append(["(Haar fallback)", fallback, "-", "-"])
        out += [f"## {n}. {title} ({len(team)}): who won the rounds",
                table(["agent", "rounds won", "share of vote weight", "rejected"], rows)]
        n += 1

    text = "\n\n".join(out) + "\n"
    (REPO / "runs" / f"{name}-report.md").write_text(text)
    print(text + f"\nsaved runs/{name}-report.md")


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("--name", required=True, help="prefix for every run in this comparison")
    ap.add_argument("--prompts", nargs="+", default=sorted(p.stem for p in PROMPTS.glob("*.md")),
                    help="prompt names in agents/prompts/ (default: all)")
    ap.add_argument("--rounds", type=int, default=3, help="small by default; raise for the final numbers")
    ap.add_argument("--repeats", type=int, default=1, help="Claude runs vary; repeat each LLM setup for the final numbers")
    ap.add_argument("--max-sessions", type=int, default=10,
                    help="hard cap on Claude sessions per prompt in each run; a run with N prompts may use "
                         "N times this (each attempt starts one session per prompt)")
    ap.add_argument("--solo", action="store_true", help="also run each prompt alone against the Haar search")
    args = ap.parse_args(argv)
    if args.mode == "run":
        run_all(args, extra)
    report(args.name)


if __name__ == "__main__":
    main()
