# Sohan's agent: benchmarks

Same agent code, system prompt and sandbox as `agents/andrew` (only the agent
name differs). Measured on Sohan's Mac (15 threads) through the shared harness,
on the staged validate set (364 faces, 682 non-faces). Staged data checksums
match `pipeline/SPEC.md`. Full-precision values are in each run's `runs.jsonl`.

## Validation accuracy

| Run | What it is | AUC | Detection @ 1% FP | @ 5% FP | @ 10% FP | Precision @ 1% FP |
|---|---|---:|---:|---:|---:|---:|
| **run-02** (frozen) | Viola-Jones cascade, discrete AdaBoost stumps, 7 stages (3, 5, 8, 14, 17, 19, 17), 83 weak classifiers | 0.990193 | 0.951 (6 FP) | 0.970 (25 FP) | 0.975 (36 FP) | 0.983 |

## Speed

| Run | Train, wall | Train, CPU | Weak classifiers per non-face | per face | Predict CPU per image |
|---|---:|---:|---:|---:|---:|
| **run-02** | 117.1 s | 54.5 s | **10.4** | 80.5 | 20.8 µs |

Another agent run was training on the same machine during this measurement, so
the time columns aren't clean. Re-time the learner alone, on the shared machine,
for the final table. Weak classifiers per image doesn't depend on the machine.

## Compared with teammates (validate only)

| Learner | AUC | Detection @ 1% FP | Weak classifiers per non-face |
|---|---:|---:|---:|
| Andrew run-02 | 0.997645 | 0.995 | 45.8 |
| Surya run-01 | 0.997637 | 0.984 | 21.5 |
| Surya run-02 | 0.996898 | 0.986 | 27.5 |
| **Sohan run-02** | 0.990193 | 0.951 | 10.4 |

This learner is the least accurate on validate and rejects non-faces with the
fewest weak classifiers. One validate set doesn't establish either difference.
The CBCL test set is the real comparison.

## Run settings

- **run-02:** model `claude-sonnet-5-5`, 9 turns, 1 of 8 validation calls,
  cascade required. The saved `learner/` is byte-identical to the validated build.
- The agent tried to tighten its stage cutoff (non-face pass rate 0.45 to 0.35)
  after validation, but the edit failed. The frozen learner uses 0.45, and 0.35
  is untested.
- **Isolation (`runs/redteam-1`, model `claude-opus-5-5`):** reads of private
  labels and files outside the workspace, git and network were all blocked, and
  workspace access worked. Attempts 9 and 10 were inconclusive because their
  probe files weren't created on this machine. Attempt 2 did get
  `Operation not permitted` on the real private labels.
- **Audit of run-02's tool log:** one denied call, a read of the agent's own
  background output file. No attempts to reach validate, the repo or the network.
- **Not included:** an earlier false start was stopped before any validation
  call and deleted. A longer `claude-opus-5-5` run (run-01) was still in progress
  when this was written.
