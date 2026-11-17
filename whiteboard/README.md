# Whiteboard pipeline

The 11-15 whiteboard diagram, one file per box, nothing else (~250 lines).

```
 ┌──────────── small loop ────────────┐
 │ LLM ──candidate rule──▶ Harness    │──▶ Combiner ──▶ Evaluator
 │  ▲   not good enough /   │         │    all rules     validation set
 │  └── reweight training ◀─┘         │    → classifier       │
 └────────────────────────────────────┘      + cascade        │
        ▲                                                     │
        └──────────────── big arrow: validation report ───────┘
```

| Box | File | What it does |
|---|---|---|
| LLM | `llm.py` | Claude writes a rule shape `rule(X, *knobs)` plus a `GRID` of knob settings (positions, sizes). Sees kept rules, hardest training images, its earlier tries, validation report. |
| Harness | `harness.py` | Adjusts every photo to the same brightness and contrast, tries every knob setting (up to 4,000), keeps the best, gives weighted error and µs per image. |
| Not good enough / weighted training | `run.py` | Error ≥ 0.35 → show Claude all its tries, try again (up to 10, then stop). Kept → AdaBoost reweight. |
| Combiner | `combiner.py` | Weighted vote of all rules = classifier. Checkpoints after 1, 3, 7, 15… rules = cascade. |
| Evaluator | `evaluator.py` | Detection at 1/5/10% false alarms, cascade speed, what each stage throws out. |
| Big arrow | `run.py` | The evaluator's text goes into the next round's LLM prompt. |

```
python -m whiteboard.run --fake        # random Haar rules, no Claude, ~1 s
python -m whiteboard.run --rounds 4    # Claude writes the rules (needs `claude` on PATH)
```

Results land in `runs/whiteboard-<time>/` (rules.json, history.json, report.txt). The CBCL test set is never loaded.

## Results (16 rules, CBCL test set: 472 faces, 23,573 non-faces, detection at 1 / 5 / 10% false alarms)

| Pipeline | 1% | 5% | 10% |
|---|---|---|---|
| Whiteboard v1 (no contrast fix, no search) | 4.9 | 15.3 | 25.4 |
| Whiteboard, random box shapes + search (no Claude) | 16.5 | 46.6 | 63.3 |
| **Whiteboard, Claude shapes + search** | **18.0** | **44.3** | **59.7** |
| Team brute-force Haar (rerun locally) | 17.6 | 39.8 | 53.6 |
| Team 4 agents only (11-17 slides) | 19.7 | 45.3 | 60.2 |

One run each. The test set was already used on 11-02; nothing here was tuned on it.
