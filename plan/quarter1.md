# Quarter 1 (DSC 180A): Learn, Implement, Replicate

**Term:** Fall 2026 · **Domain:** D31 Ensemble of Agents  
**Parent doc:** [curriculum.md](curriculum.md) · **Architecture:** [agent-architecture.md](agent-architecture.md)

## Goal

Gain background in ensemble methods and agent-based systems by **replicating** bagging and AdaBoost as communicating agents, matching sklearn baselines. End the quarter with a **Quarter 2 proposal** for a self-configuring agent ML system.

## Readings

From [description.md](description.md):

1. Freund, Yoav, and Robert E. Schapire. "A decision-theoretic generalization of on-line learning and an application to boosting." *Journal of Computer and System Sciences* 55.1 (1997): 119–139.
2. Breiman, Leo. "Bagging predictors." *Machine Learning* 24.2 (1996): 123–140.
3. Breiman, Leo. "Random forests." *Machine Learning* 45.1 (2001): 5–32.
4. Alafate, Julaiti, and Yoav S. Freund. "Faster boosting with smaller memory." *Advances in Neural Information Processing Systems* 32 (2019). *(skim)*
5. Freund, Yoav, and Robert E. Schapire. "Large margin classification using the perceptron algorithm." *Machine Learning* 37.3 (1999): 277–296.

Supporting material: sklearn docs for `BaggingClassifier`, `RandomForestClassifier`, and `AdaBoostClassifier`; the interfaces in [agent-architecture.md](agent-architecture.md).

## Week-by-week schedule

| Week | Theme | Focus | Deliverable |
| --- | --- | --- | --- |
| 1 | Foundations | Capstone norms; domain overview; assign papers [1]–[3]; set up shared repo | Repo skeleton; reading assignments |
| 2 | Averaged perceptron | Andrew Chen presents Freund and Schapire (1999) [5]; implements it or finds an implementation; runs it on a Kaggle dataset | Paper slides; results (accuracy, confidence, speed) |
| 3 | Bagging | Saisohan Shingade presents Breiman (1996) [2]; implements it or finds an implementation; runs it on a Kaggle dataset | Paper slides; results (accuracy, confidence, speed) |
| 4 | Random forests | Surya Setty presents Breiman (2001) [3]; implements it or finds an implementation; runs it on a Kaggle dataset | Paper slides; results (accuracy, confidence, speed) |
| 5 | Boosting stumps | Uday Lingampalli presents Freund and Schapire (1997) [1]; implements it or finds an implementation; runs it on a Kaggle dataset | Paper slides; results (accuracy, confidence, speed) |
| 6 | Boosting as agents | AdaBoost weight updates; sequential coordinator protocol | Coordinator stub that reweights examples |
| 7 | Boosting as agents | Weak-learner agents under boosting; train / combine loop with audit log | End-to-end AdaBoost agent pipeline |
| 8 | Boosting as agents | Match sklearn AdaBoost; plot error vs. rounds; compare to bagging | Replication table + learning curves |
| 9 | Synthesis | Ablations (communication pattern, `#agents`, weak learner); draft Q1 report | Ablation results; report draft |
| 10 | Proposal | Finalize Q1 report; write Q2 proposal; mentor feedback | **Q1 report** + **Q2 proposal** due |

Adjust dates to the official DSC 180A calendar; the sequence of themes stays fixed.

## Milestones

1. **M1 (end of week 2):** Shared repo set up; first paper presentation and Kaggle run done.
2. **M2 (end of week 5):** All four paper presentations done, with results on Kaggle datasets.
3. **M3 (end of week 8):** AdaBoost-as-agents replication accepted.
4. **M4 (end of week 10):** Q1 report and Q2 proposal submitted.

## Q1 Project success criteria

- Correct agent protocols for **bagging (parallel)** and **AdaBoost (sequential)**.
- Accuracy / error within a small tolerance of sklearn baselines, or documented and justified gaps.
- Persistent logs of agent messages and coordinator decisions.
- Report with method summary, architecture diagrams, results, and limitations.

## Q1 report rubric

| Section | What to include | Weight (guide) |
| --- | --- | --- |
| Problem & background | Ensemble ideas; why agents; citations [1]–[3] | 15% |
| Architecture | Diagrams for bagging and boosting agent protocols; link to [agent-architecture.md](agent-architecture.md) | 20% |
| Replication setup | Datasets, splits, seeds, sklearn baselines, tolerance | 15% |
| Results | Tables/curves vs. sklearn; ablations | 25% |
| Discussion | Gaps, failure modes, what Q2 should reuse | 15% |
| Reproducibility | How to re-run experiments from the repo | 10% |

## Q2 proposal checklist (due end of Q1)

The proposal must specify:

- [ ] Target loop: task in → method selection → combination → metrics → improve
- [ ] Evaluation suite (3–5 tabular tasks varying size / noise / imbalance)
- [ ] Non-goals (e.g. no vision/NLP in v1)
- [ ] How Q1 learner / coordinator / evaluator interfaces will be reused
- [ ] Success metrics vs. fixed AdaBoost and bagging baselines
- [ ] Risks (LLM flakiness, budget, team roles) and mitigations

## Weekly mentoring focus

| Weeks | Typical meeting agenda |
| --- | --- |
| 1 | Course overview (class 1 slides) |
| 2–5 | Student paper presentation; critique implementation and Kaggle results |
| 6–8 | Demo boosting agent run; critique results |
| 9–10 | Report and proposal feedback |

## Team notes

Work asynchronously in a shared repository. Come to the weekly hour with a short demo or decision list. The curriculum works with 3 or 4 students; split ownership across bagging pipeline, boosting pipeline, evaluation/logging, and documentation.
