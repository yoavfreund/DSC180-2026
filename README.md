# Agent-written weak rules for a boosted face detector

Branch `andrew-agent-learners` (Andrew's implementation). Built from `coordinating-agents`; the
shared evaluator `eval/` comes from `harness-validation`.

Every boosting round, several agents run in parallel. Each agent writes one weak rule in C. A
plain-Python coordinator (no LLM) builds, scores and times every rule, keeps the best one under
the current error bar, and reweights the images. After training, the chain of rules can be sliced
into a cascade.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests            # 53 tests, no LLM
eval/tests/run_tests.sh                        # shared C evaluator tests

# Baseline with no LLM: brute-force Haar search is the only agent.
.venv/bin/python main.py --run-name haar-01 --rounds 10 --agent base=haar

# Claude agents (each with its own .md) against the Haar baseline. Uses your Claude subscription;
# leave ANTHROPIC_API_KEY unset.
.venv/bin/python main.py --run-name mix-01 --rounds 10 \
    --agent andrew=claude:agents/prompts/andrew.md --agent base=haar

# After training: cascade (calibrated on validation faces), scores, shared evaluator.
.venv/bin/python cascade.py --run-dir runs/mix-01 --face-loss 0.01
.venv/bin/python predict.py --run-dir runs/mix-01 --images validate --scores s.txt --key k.txt --cascade
eval/eval s.txt k.txt --fp-rates 0.001,0.01,0.1
```

## One round

1. The bar moves one level down from where the last round ended (the first round starts at the bottom).
2. **Attempt:** every agent starts as its own OS process. After `--deadline` seconds (default 40)
   the coordinator kills each process group that is still running. The last rule an agent saved
   with `submit_rule` before then counts.
3. The coordinator re-does everything for each submission: it checks and compiles the source,
   scores every image, fits the threshold and polarity on the weighted **training** images, and
   measures weighted error on the hidden **validation** images.
4. It times all new rules in turn with an empty rule, on 256 images that stay in cache.
   `ns` = extra time over the empty rule; anything below 1 ns counts as 1 ns.
5. If any rule seen this round is under the bar, the coordinator keeps the one with the most
   boosting progress per ns: `-0.5 ln(1 - 4g^2) / ns`, where `g = 0.5 - error`. Then
   `alpha = 0.5 ln((1-e)/e)` (from error only), `w *= exp(-alpha y h)` on all images, and the next round starts.
6. Otherwise the attempt fails. After `--fails-per-level` failures (default 5) the bar moves up a
   level. Default ladder: 0.30, 0.33, ..., 0.48, 0.49 (`--ladder` sets any list, for example big
   steps first). When the top level is exhausted, a brute-force Haar search runs as a fallback
   (`--no-fallback` skips the round instead).

Agents get written feedback before the next attempt: their own result, and after a round, the
winner's numbers and the error they would have needed to win at their speed.

## What an agent can do

An LLM agent is a Claude Agent SDK session whose system prompt is that agent's `.md`. It has **no
built-in tools**: it cannot read or write files or run commands. Each session's start message
records its tool list in the audit. The prompt holds the task, the C contract
(`harness/vj_learner.h`), weighted statistics of the training images, the accepted rules and its
feedback. The agent has two tools:

- `try_rule(source)`: build the rule; report weighted training error, time, polarity.
- `submit_rule(source)`: the same, and save it as the submission.

A rule is `double weak_score(const vj_image *im)`. The image struct has the pixels, integral
images, mean and std, and the header has Haar, rectangle, pixel and LBP helpers. The build
rejects `#include`, `#pragma`, `asm` and `__attribute__`, and any call to a function outside
`<math.h>` (checked on the compiled object with `nm`, so hand-written declarations of `fopen`,
`system` or `malloc` fail). A crash, a hang, or a NaN or infinite score fails the candidate,
never the run.

Validation images exist only in the coordinator process. The CBCL test set is never loaded.

## Files

| Path | What it does |
|---|---|
| `main.py`, `config.py` | command line and defaults |
| `coordinator/coordinator.py` | the round and attempt loops, kill at deadline, scoring, weight update, audit |
| `coordinator/ladder.py` | the error bar: levels, failures per level, carry-over |
| `coordinator/selection.py` | progress per ns, alpha, error needed to win |
| `coordinator/brief.py` | the prompt each agent gets |
| `agents/claude_worker.py` | one Claude session with `try_rule` / `submit_rule` |
| `agents/haar_worker.py` | brute-force Haar search (baseline, fallback, LLM-free tests) |
| `agents/scripted_worker.py` | fake agents for tests (hang, crash, forbidden call, ...) |
| `agents/prompts/andrew.md` | Andrew's agent strategy (each student writes their own) |
| `harness/vj_learner.h`, `harness/runner.c` | the C contract and the scoring/timing program |
| `harness/build.py`, `harness/threshold.py` | source checks, compile, run, time; exact weighted threshold fit |
| `ensemble.py`, `cascade.py`, `predict.py` | reload a run, slice the cascade, write scores for `eval/` |
| `eval/` | shared evaluator: detection and precision at fixed FP rates (C, tested) |

A run folder holds `audit.jsonl` (every event), `manifest.jsonl` (accepted rules with source hash,
threshold, polarity, alpha, ns), `learners/round_XXX.c`, per-attempt agent folders (prompt,
transcript, submission, tool reports) and `summary.json`.

## Known limits

- The split is the team's folder split: 5,931 training and 1,046 validation images. Validation
  error picks rules and sets alpha, and the cascade is calibrated on it too, so validation numbers
  are optimistic. Only the test set gives real numbers.
- Agents see aggregate validation errors in their feedback, so many attempts can slowly fit validation.
- Timing runs on the machine doing the run. Compare runs from one machine only.
- With the 1 ns floor, a few Haar features are "free", so an agent can pack several into one rule.
  The floor value decides how many.
- Each rule votes +1 or −1, so early cascade cuts see only a few distinct running scores and can
  reject little without losing many faces. The cascade pays off once the chain is long.
- Process-group kill is POSIX only (macOS, Linux, or WSL on Windows).
