"""Evaluate local labeled predictions; no external dependencies."""
import math
import sys
labels = dict(line.split() for line in open(sys.argv[1]))
rows = [(float(s), int(labels[i])) for i, s in
        (line.split() for line in open(sys.argv[2])) if i in labels]
assert len(rows) == len(labels)
assert all(math.isfinite(s) for s, y in rows)
rows.sort(reverse=True)
p = sum(y for s, y in rows)
n = len(rows) - p
tp = fp = 0
auc = 0.0
points = [(0, 0)]
i = 0
while i < len(rows):
    j = i
    dt = df = 0
    while j < len(rows) and rows[j][0] == rows[i][0]:
        dt += rows[j][1]
        df += 1 - rows[j][1]
        j += 1
    auc += df * (tp + 0.5 * dt)
    tp += dt
    fp += df
    points.append((fp, tp))
    i = j
print(f'n={len(rows)} faces={p} negatives={n} AUC={auc/(p*n):.6f}')
for target in [.001, .01, .05, .1]:
    fp, tp = max((f, t) for f, t in points if f <= target*n)
    print(f'FPR target={target:.3f}: detection={tp/p:.4f}, precision={tp/max(1,tp+fp):.4f} ({tp} TP, {fp} FP)')
