"""Compare agent prompts under identical settings.

  # run every setup (skips runs that already finished), then print the tables
  .venv/bin/python compare.py run --name cmp1 --rounds 10 --repeats 2
  .venv/bin/python compare.py run --name cmp1 --prompts andrew sohan -- --deadline 60
  # tables only, from finished runs
  .venv/bin/python compare.py report --name cmp1

Setups, each a normal main.py run under runs/:
  NAME-haar             brute-force Haar only (deterministic, so run once): the bar to beat
  NAME-solo-P-rK        prompt P alone against the Haar search, repeat K
  NAME-team-rK          every prompt together against the Haar search, repeat K

Every run uses the same shared settings (SHARED below, plus anything after --), so the only
difference between setups is which agents compete. Runs go one at a time, because rules are timed
on this machine and parallel runs would slow each other. Compare runs from one machine only.

The tables:
  1. Detector quality per setup, on the audit images (held out from everything, so honest):
     detection at 1% false positives and AUC, plus rules accepted, total ns and cost.
  2. Per prompt, solo runs: the share of rounds the prompt beat the Haar search.
  3. Per prompt, team runs: rounds won and its share of the final vote weight (alpha).
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PROMPTS = REPO / "agents" / "prompts"
SHARED = ["--audit-frac", "0.2", "--weight-cap", "10", "--confidence", "1"]   # README's tested best
HAAR = "base"


def setups(name, prompts, repeats):
    """(run name, agent args) for every run in the comparison."""
    out = [(f"{name}-haar", [f"{HAAR}=haar"])]
    for k in range(1, repeats + 1):
        for p in prompts:
            out.append((f"{name}-solo-{p}-r{k}", [f"{p}=claude:{PROMPTS / (p + '.md')}", f"{HAAR}=haar"]))
        out.append((f"{name}-team-r{k}",
                    [f"{p}=claude:{PROMPTS / (p + '.md')}" for p in prompts] + [f"{HAAR}=haar"]))
    return out


def run_all(args, extra):
    for p in args.prompts:
        if not (PROMPTS / f"{p}.md").is_file():
            sys.exit(f"no prompt file agents/prompts/{p}.md")
    todo = setups(args.name, args.prompts, args.repeats)
    for i, (run, agents) in enumerate(todo, 1):
        if (REPO / "runs" / run / "summary.json").is_file():
            print(f"[{i}/{len(todo)}] {run}: done already, skipped")
            continue
        if (REPO / "runs" / run).exists():
            print(f"[{i}/{len(todo)}] {run}: unfinished folder from a stopped run; delete runs/{run} to redo it")
            continue
        cmd = [sys.executable, str(REPO / "main.py"), "--run-name", run, "--rounds", str(args.rounds)]
        cmd += [x for a in agents for x in ("--agent", a)] + SHARED + extra
        print(f"[{i}/{len(todo)}] {run}", flush=True)
        if subprocess.run(cmd, cwd=REPO).returncode != 0:
            print(f"  {run} failed; continuing", file=sys.stderr)


def load(name):
    runs = {}
    for f in sorted((REPO / "runs").glob(f"{name}-*/summary.json")):
        runs[f.parent.name[len(name) + 1:]] = json.loads(f.read_text())
    return runs


def mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    m = statistics.fmean(xs)
    return f"{m:.3f}" if len(xs) == 1 else f"{m:.3f} ± {statistics.pstdev(xs):.3f}"


def table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def report(name):
    runs = load(name)
    if not runs:
        sys.exit(f"no finished runs named {name}-* under runs/")
    groups = {}
    for key, s in runs.items():
        groups.setdefault(key.rsplit("-r", 1)[0] if key != "haar" else "haar", []).append(s)

    def det(s):
        return (s.get("audit_detection_at_fp") or {}).get("0.01")

    rows = []
    for g in sorted(groups, key=lambda g: (g != "haar", g)):
        ss = groups[g]
        rows.append([g, len(ss), mean([det(s) for s in ss]), mean([s.get("audit_auc") for s in ss]),
                     mean([s["accepted"] for s in ss]), mean([s.get("ns_total") for s in ss]),
                     f"{sum(a['cost_usd'] for s in ss for a in s['agents'].values()):.2f}"])
    out = ["## 1. Detector quality (audit images)",
           table(["setup", "runs", "detection @1% FP", "AUC", "rules", "total ns", "cost $"], rows)]

    rows = []
    for g in sorted(g for g in groups if g.startswith("solo-")):
        p = g[len("solo-"):]
        share = [s["agents"][p]["wins"] / max(s["accepted"], 1) for s in groups[g]]
        st = [s["agents"][p] for s in groups[g]]
        rows.append([p, mean(share), sum(a["wins"] for a in st), sum(a["submitted"] for a in st),
                     sum(a["no_submission"] + a["killed_at_deadline"] for a in st)])
    if rows:
        out += ["## 2. Solo runs: how often each prompt beat the Haar search",
                table(["prompt", "share of rounds won", "rounds won", "rules submitted",
                       "no submission or killed"], rows)]

    team = groups.get("team", [])
    if team:
        names = sorted({a for s in team for a in s["agents"]}, key=lambda a: (a == HAAR, a))
        rows = []
        for a in names:
            st = [s["agents"].get(a, {}) for s in team]
            alpha = [x.get("alpha_total", 0) / max(sum(v["alpha_total"] for v in s["agents"].values()), 1e-12)
                     for x, s in zip(st, team)]
            rows.append([a, sum(x.get("wins", 0) for x in st), mean(alpha), sum(x.get("rejected", 0) for x in st)])
        out += [f"## 3. Team runs ({len(team)}): who won the rounds",
                table(["agent", "rounds won", "share of vote weight", "rejected"], rows)]

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
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2, help="Claude runs vary, so repeat each LLM setup")
    args = ap.parse_args(argv)
    if args.mode == "run":
        run_all(args, extra)
    report(args.name)


if __name__ == "__main__":
    main()
