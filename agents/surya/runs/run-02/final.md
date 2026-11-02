Implemented and validated the frozen **14/37/64/45-stage layout**, totaling **160 stumps** in a **4,922-byte model**.

Hidden validation, using **one call**:

- Detection: **98.6% / 98.9% / 98.9%** at ≤1% / 5% / 10% FP.
- Precision at ≤1% FP: **98.4%**.
- Mean weak evaluations: **27.5/non-face**, **158.4/face**.
- Training: **29.7 seconds**.

Exactly three constructions were used. The 99% detection preference and cumulative calibration-negative target were unmet and are documented.

Source, configuration, manifests, logs, and [final.model](final.model) are ready. Full measurements and limitations are in [RESULTS.md](RESULTS.md).