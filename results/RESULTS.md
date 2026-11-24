# Full-suite results, Nov 24 2026 (pooled re-split)

Code: branch `q1-face-detector` at `4d1a7e4` (Andrew's re-split and native-detector commits).
Run on one Apple M5 Pro laptop, one setup at a time, so the ns timings are comparable to each other.
Data: the new pooled split from `make_split.py` (seed 20261123). The test split was **not** touched.

Commands, in order:

```bash
python run.py test     # unit tests + 2-round smoke pipeline
python run.py all      # Haar baseline, 10 rounds, cascade, validation eval
python run.py agents   # compare.py run --name main --rounds 25 --max-sessions 100
# then, for each main-* run (same steps run.py all uses):
python cascade.py --run-dir runs/R --face-loss 0.01
python predict.py --run-dir runs/R --images validate --scores runs/R/scores.txt --key runs/R/key.txt --cascade
eval/eval runs/R/scores.txt runs/R/key.txt --fp-rates 0.01,0.05,0.1
```

## 1. Tests

- `pytest tests`: **77 passed**
- `eval/tests/run_tests.sh`: **all passed** (incl. 500 random cross-checks against the brute-force reference)
- 2-round smoke pipeline: ran end to end

## 2. Main comparison (25 rounds, 4 prompts: andrew, sohan, surya, uday)

Audit images: 20% of `train/` held out from everything (no weights, no agents, no rule picking),
413 faces. This is the honest yardstick. One audit face is about 0.0024 detection.

| setup | det @1% FP | det @5% FP | det @10% FP | AUC | rules | total ns | Claude sessions | cost $ |
|---|---|---|---|---|---|---|---|---|
| Haar only | 0.857 | 0.937 | 0.964 | 0.987 | 24 | 44.1 | 0 | 0.00 |
| team (agents + Haar every round) | 0.886 | **0.961** | **0.973** | 0.988 | 25 | 44.7 | 236 | 13.45 |
| agents (Haar only as fallback) | 0.886 | **0.961** | 0.966 | 0.989 | 25 | 46.8 | 264 | 15.11 |
| agents-only (no Haar at all) | **0.910** | 0.956 | **0.973** | **0.990** | 25 | 51.7 | 280 | 16.03 |

Validation images (364 faces, 682 non-faces), scored through the cascade with the compiled detector.
Validation also picks rules and calibrates the cascade, so these numbers are optimistic.

| setup | det @1% FP | det @5% FP | det @10% FP | AUC | rules per patch (cascade) |
|---|---|---|---|---|---|
| Haar only | 0.893 | 0.956 | 0.975 | 0.989 | 15.7 |
| team | **0.953** | **0.981** | **0.989** | **0.993** | 16.0 |
| agents | 0.929 | 0.967 | 0.981 | 0.990 | 16.6 |
| agents-only | 0.909 | 0.973 | 0.978 | 0.991 | 17.6 |

Every run finished all 25 rounds. None stalled, and no Haar fallback rule was needed in the agent setups.

### Who won rounds

| prompt | team: beat Haar head to head (of 25) | team: rounds won | agents: rounds won | agents-only: rounds won |
|---|---|---|---|---|
| andrew | 17 | 10 | 9 | 4 |
| sohan | 9 | 4 | 9 | 10 |
| surya | 8 | 2 | 6 | 9 |
| uday | 3 | 1 | 1 | 2 |
| Haar search | - | 8 | - | - |

Full tables, including vote-weight shares: [`main-report.md`](main-report.md).

## 3. Haar baseline from `run.py all` (10 rounds)

Validation, cascade: AUC 0.978, detection 0.640 / 0.893 / 0.945 at 1 / 5 / 10% FP,
7.5 rules per patch on average (10 in the full chain).

## What this does and doesn't show

- At every FP rate, all three agent setups match or beat Haar only on the audit images. The biggest gap is at 1% FP: +12 to +22 audit faces.
- The agent rules cost a bit more time: 44.7 to 51.7 ns total, against 44.1 for Haar only.
- Each setup ran **once**. The gaps between the three agent setups are a few audit faces, which is within run-to-run noise.
- These results use the new pooled split, so they can't be compared with anything from before Nov 24 (`FINAL_RESULTS.md`, the 11-17 slides).
- The test split is still locked. Real numbers come from one final test run.

## Files

- `main-report.md`: compare.py's report
- `runs/<setup>/`: `summary.json`, `manifest.jsonl` (accepted rules), `cascade.json`, `learners/round_XXX.c` (the C source of every accepted rule)
- `logs/`: full console output of every step
