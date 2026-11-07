"""
STEP 2f - Nested CV restricted to the framework of the original manuscript:
imputer {Autoencoder, Mean, KNN} x augmentation {none, noise, copula}; no physics descriptors, no log target,
no indicators (9 configurations). Outer 10-fold x 3 repeats, inner 3-fold. Runs in-process (n_jobs=1).
Writes ../results/G_nested_original*.csv
"""
import itertools
import pandas as pd
import framework_v3 as F

SUB = [dict(imputer=i, aug=a, mult=(0.0 if a == "none" else 1.0), physics=False, log_target=False, indicators=False)
       for i, a in itertools.product(["Autoencoder", "Mean", "KNN"], ["none", "noise", "copula"])]
F.select_config.__defaults__ = (3, SUB)
d = pd.read_csv("../data/arc_data_v3_30.csv")
X, y = d[F.FEATS].to_numpy(float), d["Nano size"].to_numpy(float)
out, w, means, chosen = F.evaluate(X, y, F.DEFAULT, repeats=3, k=10, nested=True, n_jobs=1)
pd.DataFrame([{"model": m, **v} for m, v in out.items()]).to_csv("../results/G_nested_original.csv", index=False)
pd.Series(chosen).value_counts().rename_axis("configuration").reset_index(name="times_selected").to_csv(
    "../results/G_nested_original_selected.csv", index=False)
import numpy as np
w = pd.DataFrame(w)[F.BASE]
pd.DataFrame({"base model": F.BASE, "mean": w.mean().values, "std": w.std().values}).to_csv("../results/G_nested_original_weights.csv", index=False)
print(pd.read_csv("../results/G_nested_original.csv").round(3).to_string())
