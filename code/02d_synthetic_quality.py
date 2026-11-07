"""
STEP 2d - Table 3: quality of the synthetic records, computed inside each training fold
(10-fold CV x 3 repeats; autoencoder imputation fitted on the training fold; 1 synthetic row per real row).
Statistics on min-max scaled columns (10 inputs + size), averaged over columns:
mean difference, SD difference, Kolmogorov-Smirnov statistic, correlation difference,
accuracy = 1 / (1 + sum of the four), and the median distance from each synthetic row to its
closest real training row (DCR) relative to the median distance between real rows.
Writes ../results/S_synthetic_quality.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
import framework_v3 as F

d = pd.read_csv("../data/arc_data_v3_30.csv")
X, y = d[F.FEATS].to_numpy(float), d["Nano size"].to_numpy(float)
rows = []
for rep in range(3):
    for tr, te in F.outer_folds(len(y), 10, F.SEED + rep):
        Xi = F.make_imputer("Autoencoder").fit(X[tr]).transform(X[tr])
        R = np.column_stack([Xi, y[tr]])
        lo, span = R.min(0), np.ptp(R, 0) + 1e-12
        Rs = (R - lo) / span
        for m in ["noise", "copula"]:
            rng = np.random.default_rng(F.SEED + rep)
            Xa, ya = F.augment(Xi, y[tr], 0, m, 1.0, rng)
            S = (np.column_stack([Xa, ya])[len(tr):] - lo) / span
            dm = np.mean(np.abs(Rs.mean(0) - S.mean(0)))
            ds = np.mean(np.abs(Rs.std(0) - S.std(0)))
            ks = np.mean([ks_2samp(Rs[:, j], S[:, j]).statistic for j in range(Rs.shape[1])])
            cr = np.nan_to_num(np.corrcoef(Rs, rowvar=False)); cs = np.nan_to_num(np.corrcoef(S, rowvar=False))
            dc = np.mean(np.abs(cr - cs))
            rr = cdist(Rs, Rs); np.fill_diagonal(rr, np.inf)
            dcr = np.median(cdist(S, Rs).min(1)) / np.median(rr.min(1))
            rows.append(dict(method=m, mean_diff=dm, std_diff=ds, ks=ks, corr_diff=dc,
                             accuracy=1 / (1 + dm + ds + ks + dc), dcr_ratio=dcr))
r = pd.DataFrame(rows).groupby("method").agg(["mean", "std"]).round(3)
r.to_csv("../results/S_synthetic_quality.csv")
print(r.to_string())
