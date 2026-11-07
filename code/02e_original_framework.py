"""
STEP 2e - The framework exactly as described in the original manuscript (no physics descriptors, no indicators,
no log target): autoencoder imputation + {Gaussian copula, noise, none} + RF / LightGBM / DNN / stacking.
Used for the minimally revised manuscript (Tables 5-6, Figs. 6-7).
Writes ../results/E_*.csv
"""
import os
import numpy as np
import pandas as pd
import framework_v3 as F

d = pd.read_csv("../data/arc_data_v3_30.csv")
X, y = d[F.FEATS].to_numpy(float), d["Nano size"].to_numpy(float)
STUDY = d["study"].to_numpy()
REP = int(os.environ.get("REPEATS", 10))
CFG = {a: F.DEFAULT | dict(imputer="Autoencoder", aug=a, mult=(0.0 if a == "none" else 1.0), physics=False,
                            log_target=False, indicators=False) for a in ["copula", "noise", "none"]}
ST = "Stacked ensemble (proposed)"
rows, wrows, oof = [], [], {}
for a, cfg in CFG.items():
    out, w, means, _ = F.evaluate(X, y, cfg, repeats=REP, k=10)
    rows += [{"augmentation": a, "model": m, **v} for m, v in out.items()]
    w = pd.DataFrame(w)[F.BASE]
    for b in F.BASE:
        wrows.append({"augmentation": a, "base model": b, "mean": w[b].mean(), "std": w[b].std()})
    oof[a] = means[ST]
    print(a, round(out[ST]["R2"], 3), flush=True)
pd.DataFrame(rows).to_csv("../results/E_original_framework.csv", index=False)
pd.DataFrame(wrows).to_csv("../results/E_weights.csv", index=False)
pd.DataFrame({"Ref": d.Ref, "measured_nm": y, **{f"pred_{a}": v for a, v in oof.items()}}).to_csv("../results/E_oof.csv", index=False)
lo, lo_oof = F.evaluate_groups(X, y, STUDY, CFG["copula"])
pd.DataFrame([{"model": m, **v} for m, v in lo.items()]).to_csv("../results/E_loso_copula.csv", index=False)
print("LOSO copula", round(lo[ST]["R2"], 3), flush=True)
lc = []
for a, cfg in CFG.items():
    for n in [10, 15, 20, 25, 30]:
        for r in range(1 if n == 30 else 4):
            idx = np.arange(30) if n == 30 else np.sort(np.random.default_rng(F.SEED + r).choice(30, n, replace=False))
            o = F.evaluate(X[idx], y[idx], cfg, repeats=3 if n == 30 else 1, k=min(10, n))[0]
            lc.append({"augmentation": a, "n_records": n, "draw": r, "R2": o[ST]["R2"], "MAE": o[ST]["MAE"]})
    print("LC", a, flush=True)
pd.DataFrame(lc).to_csv("../results/E_learning_curve.csv", index=False)
