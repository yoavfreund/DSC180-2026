## 1. Detector quality (audit images)

| setup | runs | detection @1% FP | detection @5% FP | detection @10% FP | AUC | rules | from Haar fallback | total ns | cost $ | why it ended |
|---|---|---|---|---|---|---|---|---|---|---|
| haar | 1 | 0.857 | 0.937 | 0.964 | 0.987 | 24 | 3 | 44.118 | 0.00 | all rounds done |
| team | 1 | 0.886 | 0.961 | 0.973 | 0.988 | 25 | 0 | 44.691 | 13.45 | all rounds done |
| agents | 1 | 0.886 | 0.961 | 0.966 | 0.989 | 25 | 0 | 46.843 | 15.11 | all rounds done |
| agents-only | 1 | 0.910 | 0.956 | 0.973 | 0.990 | 25 | 0 | 51.699 | 16.03 | all rounds done |

## 2. Head to head with the Haar search, same round and weights (more progress per ns)

| prompt | setup | rounds beaten | rounds compared | share |
|---|---|---|---|---|
| andrew | team | 17 | 25 | 0.680 |
| sohan | team | 9 | 25 | 0.360 |
| surya | team | 8 | 25 | 0.320 |
| uday | team | 3 | 25 | 0.120 |

## 3. Team runs, with the Haar search competing (1): who won the rounds

| agent | rounds won | share of vote weight | rejected |
|---|---|---|---|
| andrew | 10 | 0.315 | 0 |
| sohan | 4 | 0.178 | 0 |
| surya | 2 | 0.084 | 0 |
| uday | 1 | 0.027 | 0 |
| base | 8 | 0.396 | 0 |

## 4. Agents runs, Haar search only as the fallback (1): who won the rounds

| agent | rounds won | share of vote weight | rejected |
|---|---|---|---|
| andrew | 9 | 0.338 | 0 |
| sohan | 9 | 0.420 | 0 |
| surya | 6 | 0.187 | 0 |
| uday | 1 | 0.055 | 0 |

## 5. Agents-only runs, no Haar search at all (1): who won the rounds

| agent | rounds won | share of vote weight | rejected |
|---|---|---|---|
| andrew | 4 | 0.138 | 0 |
| sohan | 10 | 0.384 | 0 |
| surya | 9 | 0.405 | 0 |
| uday | 2 | 0.072 | 0 |
