"""Bagging (Breiman 1996) on the Kaggle Pima Indians Diabetes dataset.

Single tree vs. bagged trees, error vs. number of bootstrap copies, bagged
nearest neighbour (stable, should not move), vote-margin confidence, timings.
"""
import time, json
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import binomtest

COLS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin",
        "bmi", "pedigree", "age", "outcome"]
df = pd.read_csv("pima.csv", names=COLS)
X_all = df.drop(columns="outcome").to_numpy(float)
y_all = df["outcome"].to_numpy(int)
print("X:", X_all.shape, " diabetic:", y_all.sum(), " not:", (y_all == 0).sum())

N_SPLITS = 50
BAGS = [1, 5, 10, 25, 50, 100]


def timed(fn, repeat=1):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def tree():
    return DecisionTreeClassifier(random_state=0)


def knn():
    return KNeighborsClassifier(n_neighbors=1)


rec, curve, conf = [], [], []
for split in range(N_SPLITS):
    Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2,
                                          stratify=y_all, random_state=split)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd  # only kNN needs scaling

    # --- single tree vs bagged trees (B = 50) ---------------------------------
    for name, make, A, B in [("tree", tree, Xtr, Xte), ("knn", knn, Xtr_s, Xte_s)]:
        single, t_fit = timed(lambda: make().fit(A, ytr))
        pred, t_pred = timed(lambda: single.predict(B), 5)
        rec.append(dict(split=split, method=f"single {name}", error=100 * np.mean(pred != yte),
                        train_ms=t_fit * 1e3, pred_us=t_pred / len(yte) * 1e6))

        bag = BaggingClassifier(make(), n_estimators=50, random_state=split, n_jobs=1)
        bag, t_fit = timed(lambda: bag.fit(A, ytr))
        pred, t_pred = timed(lambda: bag.predict(B), 5)
        rec.append(dict(split=split, method=f"bagged {name} (50)", error=100 * np.mean(pred != yte),
                        train_ms=t_fit * 1e3, pred_us=t_pred / len(yte) * 1e6))

        if name == "tree":
            # vote split -> confidence
            votes = np.mean([t.predict(Xte) for t in bag.estimators_], axis=0)  # fraction voting "diabetic"
            margin = np.abs(votes - 0.5) * 2  # 0 = 25-25 split, 1 = unanimous
            for m, p, t in zip(margin, pred, yte):
                conf.append(dict(split=split, margin=m, correct=int(p == t)))

    # --- error vs number of bootstrap copies (trees) ---------------------------
    for b in BAGS:
        bag = BaggingClassifier(tree(), n_estimators=b, random_state=split, n_jobs=1).fit(Xtr, ytr)
        curve.append(dict(split=split, bags=b, error=100 * np.mean(bag.predict(Xte) != yte)))

runs, curve, conf = pd.DataFrame(rec), pd.DataFrame(curve), pd.DataFrame(conf)


def ci(x):
    x = np.asarray(x)
    return f"{x.mean():.2f} ± {1.96 * x.std(ddof=1) / np.sqrt(len(x)):.2f}"


table = runs.groupby("method", sort=False).agg(**{
    "Error % (95% CI)": ("error", ci),
    "Train (ms)": ("train_ms", lambda x: f"{x.median():.1f}"),
    "Predict (µs / patient)": ("pred_us", lambda x: f"{x.median():.1f}")})
print("\n== Main table (50 random 80/20 splits) ==\n", table.to_string())

print("\n== Error vs number of bootstrap copies (trees) ==")
curve_t = curve.groupby("bags")["error"].agg(ci)
print(curve_t.to_string())

p = runs.pivot(index="split", columns="method", values="error")
w = (p["bagged tree (50)"] < p["single tree"]).sum(); l = (p["bagged tree (50)"] > p["single tree"]).sum()
t = N_SPLITS - w - l
print(f"\nBagged tree beats single tree in {w} splits, ties {t}, loses {l}. Sign test p = {binomtest(w, w + l).pvalue:.2g}")
wk = (p["bagged knn (50)"] < p["single knn"]).sum(); lk = (p["bagged knn (50)"] > p["single knn"]).sum()
print(f"Bagged kNN beats single kNN in {wk} splits, ties {N_SPLITS - wk - lk}, loses {lk}.")

print("\n== Confidence: accuracy by vote margin (bagged trees, 50 votes) ==")
conf["bucket"] = pd.cut(conf["margin"], [-0.01, 0.2, 0.5, 0.8, 1.0],
                        labels=["close (≤60-40)", "60-40 to 75-25", "75-25 to 90-10", "lopsided (>90-10)"])
conf_t = conf.groupby("bucket", observed=True)["correct"].agg(n="size", accuracy=lambda c: f"{100 * c.mean():.1f}%")
conf_t["share"] = (conf_t["n"] / len(conf) * 100).round(1).astype(str) + "%"
print(conf_t.to_string())

table.to_csv("results_table.csv"); curve_t.to_csv("curve.csv"); conf_t.to_csv("confidence.csv")
json.dump(dict(w=int(w), l=int(l), t=int(t)), open("signtest.json", "w"))
