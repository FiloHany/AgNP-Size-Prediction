"""
STEP 2a - Main (unbiased) result: nested cross-validation.
Outer loop: 10-fold CV repeated 3 times (30 outer folds). Inside every outer training fold,
the 72 candidate configurations of the framework are compared by an inner 3-fold CV (MAE)
and the best one is refitted on the outer training fold and tested on the outer test fold.
Outputs: ../results/N_nested_performance.csv, N_selected_configs.csv, N_weights.csv, N_oof.csv
(about 60 min on 2 CPU cores)
"""
import os
import time
import pandas as pd
import framework_v3 as F

R = "../results"
d = pd.read_csv("../data/arc_data_v3_30.csv")
X, y = d[F.FEATS].to_numpy(float), d["Nano size"].to_numpy(float)
t0 = time.time()
out, w, oof, chosen = F.evaluate(X, y, cfg={}, repeats=int(os.environ.get("NREP", 3)), k=10, nested=True)
pd.DataFrame([{"model": m, **v} for m, v in out.items()]).to_csv(f"{R}/N_nested_performance.csv", index=False)
pd.Series(chosen).value_counts().rename_axis("configuration").reset_index(name="times_selected").to_csv(
    f"{R}/N_selected_configs.csv", index=False)
pd.DataFrame(w).to_csv(f"{R}/N_weights_per_fold.csv", index=False)
pd.DataFrame({"Ref": d.Ref, "measured_nm": y} | oof).to_csv(f"{R}/N_oof_predictions.csv", index=False)
print(f"nested done in {(time.time() - t0) / 60:.1f} min")
print(pd.DataFrame([{"model": m, **v} for m, v in out.items()]).round(3).to_string(index=False))
print(pd.Series(chosen).value_counts().to_string())
