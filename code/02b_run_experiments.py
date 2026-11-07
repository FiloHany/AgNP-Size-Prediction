"""
STEP 2b - All supporting tables (run after 02a_nested.py). Each stage is skipped if its output exists.
The "final configuration" = the configuration selected most often by the nested CV (02a).

  D   development table: effect of each change, one at a time      -> D_development.csv
  T   Tables 5/6: final configuration with each augmentation        -> T5_T6_by_augmentation.csv
  A   ablation around the final configuration                       -> A_ablation.csv
  W   learned ensemble weights                                      -> W_ensemble_weights.csv
  L   leave-one-study-out                                           -> L_leave_one_study_out.csv
  Y   y-randomization                                               -> Y_y_randomization.csv
  C   learning curve                                                -> C_learning_curve.csv
  I   permutation importance                                        -> I_feature_importance.csv
  M   imputer masking test                                          -> M_masking_test.csv
  O   original protocol on the authors' synthetic sheets            -> O_original_protocol.csv
  H   hyper-parameter / configuration table                         -> H_configuration.csv
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import framework_v3 as F

R = "../results"
REP = int(os.environ.get("REPEATS", 10))
d = pd.read_csv("../data/arc_data_v3_30.csv")
X, y, STUDY = d[F.FEATS].to_numpy(float), d["Nano size"].to_numpy(float), d.study.to_numpy()
T0 = time.time()
V2 = dict(imputer="Autoencoder", aug="noise", mult=1.0, physics=False, log_target=False, indicators=False)

sel = pd.read_csv(f"{R}/N_selected_configs.csv")
by_name = {F.cfg_name(c): c for c in F.GRID}
FINAL = by_name[sel.iloc[0].configuration]
print("final configuration:", sel.iloc[0].configuration, f"(selected in {sel.iloc[0].times_selected} of {sel.times_selected.sum()} outer folds)")


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def done(f):
    return os.path.exists(f"{R}/{f}")


def stacked(out):
    return out["Stacked ensemble (proposed)"]


def with_aug(c, a):
    return c | dict(aug=a, mult=0.0 if a == "none" else 1.0)


final_eval = {}


def get_final():
    if "x" not in final_eval:
        final_eval["x"] = F.evaluate(X, y, FINAL, repeats=REP, k=10)
    return final_eval["x"]


# ---------------------------------------------------------------- D development
if not done("D_development.csv"):
    v2 = pd.read_csv("../data/v2_reference_result.csv").iloc[0]
    rows = [{"step": "Package v2: 33 records, 5-fold CV, autoencoder + noise (1x) + stacking", "R2": v2.R2, "R2_sd": v2.R2_sd,
             "MAE": v2.MAE, "MAE_sd": v2.MAE_sd, "RMSE": v2.RMSE, "RMSE_sd": v2.RMSE_sd}]
    for name, cfg, k in [("30 records (Refs 15, 17 removed; Ref 18 merged), 5-fold CV", V2, 5),
                         ("30 records, 10-fold CV  [baseline for the rows below]", V2, 10),
                         ("  + physics features", V2 | dict(physics=True), 10),
                         ("  + missing-value indicators", V2 | dict(indicators=True), 10),
                         ("  + log-transformed target", V2 | dict(log_target=True), 10),
                         ("  no augmentation", with_aug(V2, "none"), 10),
                         ("  Gaussian copula instead of noise", with_aug(V2, "copula"), 10),
                         ("  mean imputation instead of autoencoder", V2 | dict(imputer="Mean"), 10),
                         ("  KNN imputation instead of autoencoder", V2 | dict(imputer="KNN"), 10)]:
        out = F.evaluate(X, y, cfg, repeats=REP, k=k)[0]
        rows.append({"step": name, **stacked(out)})
        log(f"D {name}: R2 = {stacked(out)['R2']:.3f}")
    rows.append({"step": f"Final configuration (most selected by nested CV): {F.cfg_name(FINAL)}", **stacked(get_final()[0])})
    nested = pd.read_csv(f"{R}/N_nested_performance.csv").set_index("model").loc["Stacked ensemble (proposed)"]
    rows.append({"step": "Nested CV, selection inside each training fold (unbiased main result)", **nested.to_dict()})
    pd.DataFrame(rows).to_csv(f"{R}/D_development.csv", index=False)

# ---------------------------------------------------------------- T Tables 5/6
if not done("T5_T6_by_augmentation.csv"):
    rows = []
    for a, label in [("copula", "Gaussian copula"), ("noise", "Noise augmentation"), ("none", "No augmentation")]:
        c = with_aug(FINAL, a)
        out = get_final()[0] if c == FINAL else F.evaluate(X, y, c, repeats=REP, k=10)[0]
        rows += [{"augmentation": label, "model": m, **v} for m, v in out.items()]
        log(f"T {label}: stacked R2 = {stacked(out)['R2']:.3f}")
    pd.DataFrame(rows).to_csv(f"{R}/T5_T6_by_augmentation.csv", index=False)

# ---------------------------------------------------------------- A ablation
if not done("A_ablation.csv"):
    out, w, oof, _ = get_final()
    rows = [{"variant": "Final framework (stacked ensemble)", **out["Stacked ensemble (proposed)"]},
            {"variant": "Simple average instead of stacking", **out["Average ensemble"]}]
    rows += [{"variant": f"Single model: {b}", **out[b]} for b in F.BASE]
    alts = [("Without physics features", FINAL | dict(physics=not FINAL["physics"])) if FINAL["physics"] else None,
            ("With physics features", FINAL | dict(physics=True)) if not FINAL["physics"] else None,
            ("Autoencoder imputation", FINAL | dict(imputer="Autoencoder")) if FINAL["imputer"] != "Autoencoder" else None,
            ("Mean imputation", FINAL | dict(imputer="Mean")) if FINAL["imputer"] != "Mean" else None,
            ("KNN imputation", FINAL | dict(imputer="KNN")) if FINAL["imputer"] != "KNN" else None,
            ("Log-transformed target", FINAL | dict(log_target=True)) if not FINAL["log_target"] else ("Without log target", FINAL | dict(log_target=False)),
            ("With missing indicators", FINAL | dict(indicators=True)) if not FINAL["indicators"] else ("Without missing indicators", FINAL | dict(indicators=False))]
    for item in alts:
        if item is None:
            continue
        name, c = item
        o = F.evaluate(X, y, c, repeats=REP, k=10)[0]
        rows.append({"variant": name, **stacked(o)})
        log(f"A {name}: R2 = {stacked(o)['R2']:.3f}")
    pd.DataFrame(rows).to_csv(f"{R}/A_ablation.csv", index=False)
    pd.DataFrame({"Ref": d.Ref, "measured_nm": y} | oof).to_csv(f"{R}/F_oof_predictions_final.csv", index=False)

# ---------------------------------------------------------------- W weights
if not done("W_ensemble_weights.csv"):
    w = pd.DataFrame(get_final()[1])
    t = w[F.BASE].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "base model"})
    t.to_csv(f"{R}/W_ensemble_weights.csv", index=False)
    nw = pd.read_csv(f"{R}/N_weights_per_fold.csv")[F.BASE].agg(["mean", "std"]).T
    nw.reset_index().rename(columns={"index": "base model"}).to_csv(f"{R}/W_ensemble_weights_nested.csv", index=False)

# ---------------------------------------------------------------- L leave-one-study-out
if not done("L_leave_one_study_out.csv"):
    out, oof = F.evaluate_groups(X, y, STUDY, FINAL)
    pd.DataFrame([{"model": m, **v} for m, v in out.items()]).to_csv(f"{R}/L_leave_one_study_out.csv", index=False)
    pd.DataFrame({"Ref": d.Ref, "measured_nm": y, "predicted_nm": oof["Stacked ensemble (proposed)"]}).to_csv(
        f"{R}/L_oof_predictions.csv", index=False)
    log(f"L leave-one-study-out: stacked R2 = {stacked(out)['R2']:.3f}")

# ---------------------------------------------------------------- Y y-randomization
if not done("Y_y_randomization.csv"):
    rng = np.random.default_rng(F.SEED)
    perm = [stacked(F.evaluate(X, rng.permutation(y), FINAL, repeats=1, k=10)[0])["R2"]
            for _ in range(int(os.environ.get("NPERM", 30)))]
    pd.DataFrame({"R2_shuffled_target": perm}).to_csv(f"{R}/Y_y_randomization.csv", index=False)
    log(f"Y y-randomization: mean {np.mean(perm):.3f} max {np.max(perm):.3f}")

# ---------------------------------------------------------------- C learning curve
if not done("C_learning_curve.csv"):
    rows = []
    for n in [10, 15, 20, 25, 30]:
        for r in range(1 if n == 30 else 4):
            idx = np.arange(30) if n == 30 else np.sort(np.random.default_rng(F.SEED + r).choice(30, n, replace=False))
            o = F.evaluate(X[idx], y[idx], FINAL, repeats=3 if n == 30 else 1, k=min(10, n))[0]
            rows.append({"n_records": n, "draw": r, "R2": stacked(o)["R2"], "MAE": stacked(o)["MAE"]})
        log(f"C n={n}")
    pd.DataFrame(rows).to_csv(f"{R}/C_learning_curve.csv", index=False)

# ---------------------------------------------------------------- I importance
if not done("I_feature_importance.csv"):
    def fold_imp(r, tr, te):
        fw = F.fit_framework(X[tr], y[tr], FINAL, seed=F.SEED + r)
        base = np.mean(np.abs(fw.predict(X[te]) - y[te]))
        rng = np.random.default_rng(r)
        res = []
        for j in range(X.shape[1]):
            s = []
            for _ in range(10):
                Xp = X[te].copy()
                Xp[:, j] = X[rng.integers(0, len(X), len(te)), j]   # draw from the full column distribution
                s.append(np.mean(np.abs(fw.predict(Xp) - y[te])) - base)
            res.append(np.mean(s))
        return res
    tasks = [(r, tr, te) for r in range(5) for tr, te in F.outer_folds(30, 5, F.SEED + r)]
    imps = np.array(Parallel(n_jobs=2)(delayed(fold_imp)(*t) for t in tasks))
    pd.DataFrame({"feature": F.FEATS, "MAE_increase_nm": imps.mean(0), "sd": imps.std(0)}).sort_values(
        "MAE_increase_nm", ascending=False).to_csv(f"{R}/I_feature_importance.csv", index=False)
    log("I done")

# ---------------------------------------------------------------- M masking test
if not done("M_masking_test.csv"):
    obs = ~np.isnan(X)
    sd = np.nanstd(X, 0) + 1e-12
    recs = []
    for r in range(20):
        rng = np.random.default_rng(F.SEED + r)
        M = obs & (rng.random(X.shape) < 0.15)
        for j in range(X.shape[1]):
            if (obs[:, j] & ~M[:, j]).sum() < 3:
                M[:, j] = False
        Xm = X.copy(); Xm[M] = np.nan
        for name in ["Autoencoder", "KNN", "Mean", "Median", "MICE"]:
            E = (F.make_imputer(name).fit_transform(Xm) - X) / sd
            recs.append({"imputer": name, "feature": "ALL", "nrmse": np.sqrt(np.mean(E[M] ** 2))})
            for j, f in enumerate(F.FEATS):
                if M[:, j].any():
                    recs.append({"imputer": name, "feature": f, "nrmse": np.sqrt(np.mean(E[M[:, j], j] ** 2))})
    t = pd.DataFrame(recs).groupby(["imputer", "feature"]).nrmse.mean().unstack()
    t = t[["ALL"] + [f for f in F.FEATS if f in t.columns]].sort_values("ALL")
    t.round(3).to_csv(f"{R}/M_masking_test.csv")
    log("M done")

# ---------------------------------------------------------------- O original protocol
if not done("O_original_protocol.csv"):
    def rf():
        return RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=2)

    def lg():
        return lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.01, max_depth=7, num_leaves=31, subsample=0.31,
                                 reg_alpha=1.18e-7, reg_lambda=1e-9, random_state=42, verbose=-1, n_jobs=2)

    def dn():
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64, 32), learning_rate_init=1e-3,
                             batch_size=32, max_iter=200, early_stopping=True, validation_fraction=0.2,
                             n_iter_no_change=15, random_state=42))
    rows = []
    for sheet, label in [("GaussianCopula", "Gaussian copula"), ("Noise_Augmentation", "Noise augmentation")]:
        s = pd.read_excel("../data/synthetic_datasets_original.xlsx", sheet)
        Xa, ya = s.iloc[:, 1:].to_numpy(float), s.iloc[:, 0].to_numpy(float)
        res = {m: [] for m in ["Random Forest", "LightGBM", "DNN", "Average ensemble (original)"]}
        for i in range(10):
            Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.2, random_state=i)
            P = {n: f().fit(Xtr, ytr).predict(Xte) for n, f in [("Random Forest", rf), ("LightGBM", lg), ("DNN", dn)]}
            P["Average ensemble (original)"] = np.mean(list(P.values()), 0)
            for n, p in P.items():
                res[n].append(F.metrics(yte, p))
        for n, a in res.items():
            a = np.array(a)
            rows.append({"augmentation": label, "model": n, "R2": a[:, 0].mean(), "R2_sd": a[:, 0].std(),
                         "MAE": a[:, 1].mean(), "MAE_sd": a[:, 1].std(), "RMSE": a[:, 2].mean(), "RMSE_sd": a[:, 2].std()})
        log(f"O {label}")
    pd.DataFrame(rows).to_csv(f"{R}/O_original_protocol.csv", index=False)

# ---------------------------------------------------------------- H configuration
if not done("H_configuration.csv"):
    c = FINAL
    hp = [
        ("Data", "30 records from 14 studies (Refs 15 and 17 removed: no current/voltage; Ref 18 merged with Ref 1: identical record); 10 inputs; target = mean size (nm)"),
        ("Evaluation (main result)", "Nested cross-validation: outer 10-fold CV repeated 3 times; inside each outer training fold, 72 framework configurations compared by inner 3-fold CV (MAE); best one refitted and tested on the outer fold"),
        ("Evaluation (supporting tables)", f"Final configuration fixed, 10-fold CV repeated {REP} times; mean ± SD over repeats"),
        ("Configuration grid (nested)", "imputer {Autoencoder, Mean, KNN} × augmentation {none, noise 1×, Gaussian copula 1×} × physics features {yes, no} × log-transformed target {yes, no} × missing indicators {yes, no}"),
        ("Final configuration (most selected)", F.cfg_name(c)),
        ("Autoencoder imputer", "Fully connected 10-8-3-8-10 (ReLU), MSE loss, L2 alpha = 0.01, standardized inputs; missing cells initialised at 0 and refilled iteratively (5 passes); training fold only; target not used"),
        ("KNN / Mean imputer", "k = 3 neighbours on available inputs / column mean of the training fold"),
        ("Missing indicators", "0/1 columns for yield, zeta potential, temperature, beaker volume and electrode gap (1 = not reported)"),
        ("Physics features", "log(V × I) power; log(V / gap) electric field; log(I / (π (d1/2)²)) current density at the anode; computed after imputation"),
        ("Noise augmentation", "Bootstrap of training-fold rows + Gaussian noise, SD = 5% of each column's SD, inputs and target; 1 synthetic row per real row"),
        ("Gaussian copula augmentation", "Empirical marginals + Gaussian copula fitted on [inputs, target] of the training fold; 1 synthetic row per real row"),
        ("Random Forest", "n_estimators = 200, max_depth = 20, min_samples_split = 5, random_state = 42"),
        ("LightGBM", "n_estimators = 400, learning_rate = 0.03, max_depth = 3, num_leaves = 7, min_child_samples = 3, subsample = 0.8, colsample_bytree = 0.8"),
        ("DNN", "Dense 128-64-32 (ReLU), Adam 1e-3, batch 32, ≤ 200 epochs, early stopping (patience 15, 20% validation); inputs and target standardized"),
        ("Stacked ensemble", "Learned weighted average (non-negative least squares, weights sum to 1) on inner 3-fold out-of-fold predictions; base predictions limited to the training size range"),
    ]
    pd.DataFrame(hp, columns=["component", "configuration"]).to_csv(f"{R}/H_configuration.csv", index=False)

log("ALL DONE")
