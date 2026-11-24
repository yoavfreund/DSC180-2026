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
.venv/bin/python run.py test      # unit tests + a 2-round pipeline run, no LLM, under a minute
.venv/bin/python run.py all       # Haar baseline end to end, no LLM, same result every run
.venv/bin/python run.py agents    # main agent comparison; needs a Claude sign-in, varies run to run
.venv/bin/python run.py clean     # delete everything the targets built
```

`run.py` only calls the scripts below in order. To run the steps yourself:

```bash
.venv/bin/python -m pytest -q tests            # 72 tests, no LLM
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
eval/eval s.txt k.txt --fp-rates 0.01,0.05,0.1
```

## Comparing prompts

`compare.py` runs every prompt in `agents/prompts/` under the same settings, one run at a time:
Haar only (the bar to beat); all prompts together with the Haar search competing every round
(team); all prompts together with the Haar search only as the fallback after every agent fails the
top of the ladder (agents); and all prompts together with no Haar search at all (agents-only,
`--no-fallback`: a round nobody passes is skipped, and two skips in a row end the run as
"stalled"). `--solo` adds each prompt alone against the Haar search. The tested settings
(audit 0.2, weight cap 10, confidence 1, 2 fails per level, flip/rotate/zoom/shift copies) are
built in. Defaults are small (3 rounds, 1 repeat, at most 10 Claude sessions per prompt in each
run, so a 4-prompt run may use 40); it prints the session cap before it starts.

```bash
.venv/bin/python compare.py run --name try1 --prompts sohan       # smallest
.venv/bin/python compare.py run --name try2                       # all prompts, small
.venv/bin/python compare.py run --name main --rounds 25 --max-sessions 100   # the main run
.venv/bin/python compare.py report --name try2                    # tables only
```

Anything after `--` goes to every `main.py` run (for example `-- --deadline 60`). It prints and
saves `runs/NAME-report.md`: detector quality on the audit images (detection at 1%, 5% and 10%
FP, AUC, rules, rules from the Haar fallback, total ns, cost, why the run ended), each prompt head
to head with the Haar search in the same rounds (team and solo runs), and rounds won and share of
the vote weight in each multi-agent setup. Finished runs are skipped, so a stopped comparison picks
up where it left off. If Claude itself fails (usage limit, API errors) the run ends with that
reason and the comparison stops, so lost sessions are never counted as lost rounds.

## Docker (recommended for Windows, works everywhere)

The image has Python 3.12, gcc and binutils. `compose.yaml` mounts `train/` and `validate/`
read-only and `runs/` read-write.

```bash
docker compose build
docker compose run --rm boost                                    # the tests
docker compose run --rm boost python main.py --run-name haar-01 --agent base=haar
```

Claude agents need your subscription sign-in inside the container:

1. On the host, run `claude setup-token` and copy the token it prints.
2. Put it in a file named `.env` next to `compose.yaml`: `CLAUDE_CODE_OAUTH_TOKEN=...`
   (`.env` is in `.gitignore` and `.dockerignore`; never commit it).
3. `docker compose run --rm boost python main.py --run-name mix-01 --agent andrew=claude:agents/prompts/andrew.md --agent base=haar`

Timing inside Docker on macOS and Windows runs in a Linux virtual machine. Compare ns values
within one run only.

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

The Haar search is deterministic, so it runs once per round; later attempts reuse its result.
Each run copies the harness C files into `runs/NAME/harness/` and builds only from that copy, so
editing the repository during a run cannot change it.

### Options for noisy late rounds (off by default)

AdaBoost moves the weight onto a few hard images. In a 20-round test the validation set's
effective size fell from 1,046 to about 60 images by round 6, so a weighted error had a standard
error near 0.06, twice the ladder step. Three options address this:

| Flag | What it does |
|---|---|
| `--audit-frac 0.2` | holds out 20% of `train/` from everything (no weight, no agent, no decision) and reports the ensemble's AUC and detection on it each round: an honest yardstick |
| `--confidence 1` | a rule passes only if validation error + 1 standard error < bar, so a lucky draw on few images does not pass |
| `--weight-cap 10` | caps every weight at 10 × the average (MadaBoost-style), so the weight cannot collapse onto a few images. Tested best: no stalls, effective validation size stays near 210 |

### Training copies (off by default)

`--augment flip,rotate,zoom,shift` adds 9 copies of every training image: a mirror, ±7° rotations,
0.9 and 1.1 zoom, and 1-pixel shifts. Copies come only from training images outside the audit share,
so validation and audit stay original. With `--audit-frac 0.2` the training set grows from 4,745 to
47,450 images. `--augment` also turns on `--balanced-init` (half the starting weight on faces). When
the training set is larger than 8,000 rows, the Haar search and `scan` run on a sample of 8,000 rows
drawn in proportion to weight.

30-round Haar-only tests (weight cap 10, confidence 1, audit 0.2), scored on the audit set:

| Copies | Training images | Audit AUC | Detection at 1% FP | Detection at 0.1% FP |
|---|---|---|---|---|
| none | 4,745 | 0.9989 | 0.976 | 0.896 |
| flip | 9,490 | 0.9975 | 0.966 | 0.831 |
| flip, rotate, zoom, shift | 47,450 | **0.9993** | **0.983** | 0.896 |
| all four + invert (removed) | 78,380 | 0.9945 (stalled at 13 rules) | 0.923 | 0.702 |

An `invert` option (the photographic negative of non-faces) was tried and removed. Inverted
non-faces never occur in real images, but they took 13% of the weight and were easy to separate,
so the search picked features that fit them (training error 0.15 on inverted copies, 0.50 on
validation). One run per setup: a 0.01 difference in detection is about 4 audit faces.

## What an agent can do

An LLM agent is a Claude Agent SDK session whose system prompt is that agent's `.md`. It has **no
built-in tools**: it cannot read or write files or run commands. Each session's start message
records its tool list in the audit. The prompt holds the task, the C contract
(`harness/vj_learner.h`), a preview of 8 training images (4 faces and 4 non-faces, drawn in
proportion to their current weight, a new draw each attempt, with whether the ensemble gets each
right), the accepted rules with their code, its feedback and its memory. It never sees the whole
training set; the tools use every training image but return only numbers. The agent has three tools:

- `scan(source, ranges)`: the source is a template that uses `P0`..`P3` for integers the agent
  is unsure of (positions, sizes). The harness tries every combination (at most 1,000) on the
  weighted training images and returns the 5 best concrete rules. The LLM picks the kind of
  rule; the search picks the numbers.
- `try_rules(sources)`: build 1 to 4 concrete rules in one call; report each one's weighted
  training error, time and polarity. Saves nothing.
- `submit_rule(source)`: build one concrete rule, report it, and save it as the submission.

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
| `agents/claude_worker.py` | one Claude session with `scan`, `try_rules`, `submit_rule` |
| `agents/scan.py` | the `scan` tool: every placement of a template, best concrete rules |
| `agents/haar_worker.py` | brute-force Haar search (baseline, fallback, LLM-free tests) |
| `agents/scripted_worker.py` | fake agents for tests (hang, crash, forbidden call, ...) |
| `agents/prompts/andrew.md` | Andrew's agent strategy (each student writes their own) |
| `agents/prompts/sohan.md` | Sohan's agent strategy: AND rules and Haar sizes the search skips |
| `compare.py` | runs every prompt under the same settings and prints comparison tables |
| `harness/vj_learner.h`, `harness/vj_scan.h`, `harness/runner.c` | the C contract, scan parameters, and the score/time/scan program |
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
- One Haar feature measures about 0.1 ns over an empty rule and counts as 1 ns. A two-feature rule
  measured about 3 ns, so under progress per ns a combination needs a much lower error to win.
  Early rounds therefore favor single features.
- Each rule votes +1 or −1, so early cascade cuts see only a few distinct running scores and can
  reject little without losing many faces. The cascade pays off once the chain is long.
- Process-group kill is POSIX only (macOS, Linux, or WSL on Windows).
