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
| LLM | `llm.py` | Claude writes `rule(img)` → one number. Sees kept rules, hardest training images, last failure, validation report. |
| Harness | `harness.py` | Runs the rule on all training images, picks the best threshold, gives weighted error and µs per image. |
| Not good enough / weighted training | `run.py` | Error ≥ bar → tell the LLM why, try again. Kept → AdaBoost reweight. |
| Combiner | `combiner.py` | Weighted vote of all rules = classifier. Checkpoints after 1, 3, 7, 15… rules = cascade. |
| Evaluator | `evaluator.py` | Detection at 1/5/10% false alarms, cascade speed, what each stage throws out. |
| Big arrow | `run.py` | The evaluator's text goes into the next round's LLM prompt. |

```
python -m whiteboard.run --fake        # random Haar rules, no Claude, ~1 s
python -m whiteboard.run --rounds 4    # Claude writes the rules (needs `claude` on PATH)
```

Results land in `runs/whiteboard-<time>/` (rules.json, history.json, report.txt). The CBCL test set is never loaded.
