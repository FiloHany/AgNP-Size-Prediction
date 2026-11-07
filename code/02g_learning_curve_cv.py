"""
STEP 2g - Learning curve with fixed real test folds: in each outer fold (10-fold CV x 3 repeats) the framework is
trained on a random subset of m records of the training fold and tested on the complete (real) test fold.
Original framework: autoencoder imputation + {copula, noise, none}. Writes ../results/E_learning_curve_cv.csv
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_absolute_error
import framework_v3 as F

d = pd.read_csv("../data/arc_data_v3_30.csv")
X, y = d[F.FEATS].to_numpy(float), d["Nano size"].to_numpy(float)
SIZES = [9, 13, 18, 22, 27]
CFG = {a: F.DEFAULT | dict(imputer="Autoencoder", aug=a, mult=(0.0 if a == "none" else 1.0), physics=False,
                            log_target=False, indicators=False) for a in ["copula", "noise", "none"]}


def run(a, m, r, tr, te):
    sub = np.sort(np.random.default_rng(F.SEED + 100 * r + m).choice(tr, size=min(m, len(tr)), replace=False))
    return a, m, r, te, F.fit_framework(X[sub], y[sub], CFG[a], seed=F.SEED + r).predict(X[te])


tasks = [(a, m, r, tr, te) for a in CFG for m in SIZES for r in range(3) for tr, te in F.outer_folds(len(y), 10, F.SEED + r)]
res = Parallel(n_jobs=2)(delayed(run)(*t) for t in tasks)
pred = {}
for a, m, r, te, p in res:
    pred.setdefault((a, m, r), np.zeros(len(y)))[te] = p
rows = [{"augmentation": a, "n_train": m, "repeat": r, "R2": r2_score(y, p), "MAE": mean_absolute_error(y, p)}
        for (a, m, r), p in pred.items()]
out = pd.DataFrame(rows)
out.to_csv("../results/E_learning_curve_cv.csv", index=False)
print(out.groupby(["augmentation", "n_train"]).R2.agg(["mean", "std"]).round(3))
