"""
STEP 2c - The framework exactly as named in the paper's title, evaluated on the 30 records:
autoencoder imputation + augmentation (copula or noise) + physics features + stacked ensemble.
Writes ../results/T_framework_as_titled.csv (all five models, 10-fold CV x 10 repeats).
"""
import os
import pandas as pd
import framework_v3 as F

d = pd.read_csv("../data/arc_data_v3_30.csv")
X, y = d[F.FEATS].to_numpy(float), d["Nano size"].to_numpy(float)
REP = int(os.environ.get("REPEATS", 10))
rows = []
for a, label in [("copula", "Autoencoder + Gaussian copula"), ("noise", "Autoencoder + noise augmentation")]:
    cfg = F.DEFAULT | dict(imputer="Autoencoder", aug=a, mult=1.0, physics=True, log_target=False, indicators=False)
    out = F.evaluate(X, y, cfg, repeats=REP, k=10)[0]
    rows += [{"framework": label, "model": m, **v} for m, v in out.items()]
    print(label, round(out["Stacked ensemble (proposed)"]["R2"], 3), flush=True)
pd.DataFrame(rows).to_csv("../results/T_framework_as_titled.csv", index=False)
