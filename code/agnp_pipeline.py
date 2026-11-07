"""
Leakage-free evaluation pipeline for the AgNP size-prediction paper (revision).

Answers Reviewer 4 point by point:
  R4.1/R4.2  model complexity vs n=33     -> model comparison incl. simple baselines (Exp 3)
  R4.3       leakage / memorization       -> leaky vs clean protocol (Exp 0), y-randomization (Exp 4)
  R4.4       1,000 bootstrap+noise copies -> augmentation ablation (Exp 2)
  R4.6       autoencoder on ~76% missing  -> imputation ablation incl. drop-feature option (Exp 1)
  R1         novelty / insight            -> permutation importance (Exp 6), learning curve (Exp 5)

RULE: everything that learns from data (imputer, scaler, augmentation, model) is fitted
INSIDE each CV fold on the training part only. Test folds contain only real, original samples.

Usage:
    python agnp_pipeline.py --data your_data.csv --target size_nm
    python agnp_pipeline.py --simulate            # dry run on simulated data
"""
import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
SEED = 42


# ---------------------------------------------------------------- components
class AutoencoderImputer(BaseEstimator, TransformerMixin):
    """Small denoising autoencoder imputer (the paper's approach), fitted on train fold only."""

    def __init__(self, hidden=(8, 3, 8), n_iter=5, random_state=SEED):
        self.hidden, self.n_iter, self.random_state = hidden, n_iter, random_state

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        with np.errstate(all="ignore"):
            self.mean_ = np.nan_to_num(np.nanmean(X, 0))
            self.std_ = np.nan_to_num(np.nanstd(X, 0), nan=1.0) + 1e-9
        Z = (X - self.mean_) / self.std_
        mask = np.isnan(Z)
        Zf = np.where(mask, 0.0, Z)
        self.ae_ = MLPRegressor(hidden_layer_sizes=self.hidden, max_iter=800,
                                random_state=self.random_state, alpha=1e-2)
        for _ in range(self.n_iter):
            self.ae_.fit(Zf, Zf)
            Zf = np.where(mask, self.ae_.predict(Zf), Z)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        Z = (X - self.mean_) / self.std_
        mask = np.isnan(Z)
        Zf = np.where(mask, 0.0, Z)
        for _ in range(self.n_iter):
            Zf = np.where(mask, self.ae_.predict(Zf), Z)
        return Zf * self.std_ + self.mean_


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop given column indices (e.g. temperature) and optionally add a 'was reported' flag."""

    def __init__(self, cols=(), add_flag=True):
        self.cols, self.add_flag = cols, add_flag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        keep = [i for i in range(X.shape[1]) if i not in self.cols]
        out = X[:, keep]
        if self.add_flag and len(self.cols):
            flags = (~np.isnan(X[:, list(self.cols)])).astype(float)
            out = np.hstack([out, flags])
        return out


class NoiseAugmentedRegressor(BaseEstimator, RegressorMixin):
    """Bootstrap + Gaussian noise augmentation applied to the TRAINING fold only.
    Placed after the scaler, so noise_frac is a fraction of each feature's std."""

    def __init__(self, estimator=None, n_synthetic=0, noise_frac=0.05, random_state=SEED):
        self.estimator, self.n_synthetic = estimator, n_synthetic
        self.noise_frac, self.random_state = noise_frac, random_state

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        if self.n_synthetic > 0:
            rng = np.random.default_rng(self.random_state)
            idx = rng.integers(0, len(X), self.n_synthetic)
            Xs = X[idx] + rng.normal(0, self.noise_frac, (self.n_synthetic, X.shape[1]))
            ys = y[idx] + rng.normal(0, self.noise_frac * y.std(), self.n_synthetic)
            X, y = np.vstack([X, Xs]), np.concatenate([y, ys])
        self.est_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return self.est_.predict(X)


# ---------------------------------------------------------------- factories
def make_imputer(name, temp_idx):
    return {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "knn": KNNImputer(n_neighbors=3),
        "mice": IterativeImputer(max_iter=20, random_state=SEED),
        "autoencoder": AutoencoderImputer(),
        "drop_temp+flag": Pipeline([("drop", DropColumns(cols=(temp_idx,))),
                                    ("imp", KNNImputer(n_neighbors=3))]),
    }[name]


def make_model(name):
    gp_kernel = ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel()
    base = {
        "mean_baseline": DummyRegressor(),
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "gpr": GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True,
                                        n_restarts_optimizer=3, random_state=SEED),
        "svr": SVR(C=10, epsilon=0.1),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                               random_state=SEED),
        "gbr": GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                         random_state=SEED),
        "mlp_dnn": MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=3000,
                                random_state=SEED),
    }
    if name == "stacked":
        return StackingRegressor(
            estimators=[("rf", RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                                     random_state=SEED, n_jobs=-1)),
                        ("gbr", base["gbr"]), ("svr", base["svr"])],
            final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 13)), cv=3)
    return base[name]


def build(imputer, model, n_synthetic=0, noise_frac=0.05, temp_idx=0):
    return Pipeline([
        ("impute", make_imputer(imputer, temp_idx)),
        ("scale", StandardScaler()),
        ("model", NoiseAugmentedRegressor(make_model(model), n_synthetic, noise_frac)),
    ])


# ---------------------------------------------------------------- evaluation
def repeated_cv(pipe, X, y, n_splits=5, n_repeats=20, y_override=None):
    """Repeated K-fold. For each repeat, collect out-of-fold predictions on real samples
    and compute metrics on the full set of 33 OOF predictions (stable for tiny folds)."""
    yv = y if y_override is None else y_override
    rows = []
    for r in range(n_repeats):
        oof = np.zeros(len(yv))
        for tr, te in KFold(n_splits, shuffle=True, random_state=SEED + r).split(X):
            m = clone(pipe).fit(X[tr], yv[tr])
            oof[te] = m.predict(X[te])
        rows.append({"R2": r2_score(yv, oof), "MAE": mean_absolute_error(yv, oof),
                     "RMSE": mean_squared_error(yv, oof) ** 0.5})
    df = pd.DataFrame(rows)
    return {f"{k}_mean": df[k].mean() for k in df} | {f"{k}_std": df[k].std() for k in df}


def leaky_protocol(X, y, temp_idx, n_synthetic=1000, noise_frac=0.05):
    """Reproduces the suspected original protocol: impute on ALL data, augment from ALL data,
    THEN split. Synthetic near-copies of a sample end up in both train and test."""
    Xi = AutoencoderImputer().fit_transform(X)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(Xi), n_synthetic)
    sd = Xi.std(0)
    Xa = np.vstack([Xi, Xi[idx] + rng.normal(0, noise_frac, (n_synthetic, Xi.shape[1])) * sd])
    ya = np.concatenate([y, y[idx] + rng.normal(0, noise_frac * y.std(), n_synthetic)])
    scores = []
    for r in range(20):
        Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.2, random_state=SEED + r)
        sc = StandardScaler().fit(Xtr)
        m = clone(make_model("stacked")).fit(sc.transform(Xtr), ytr)
        scores.append(r2_score(yte, m.predict(sc.transform(Xte))))
    return {"R2_mean": float(np.mean(scores)), "R2_std": float(np.std(scores))}


# ---------------------------------------------------------------- data
def simulate(n=33, seed=SEED):
    """Synthetic stand-in shaped like the paper's data (only for testing the code)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "temperature_C": rng.uniform(20, 100, n),
        "AgNO3_mM": rng.uniform(0.5, 10, n),
        "reducer_mM": rng.uniform(0.5, 20, n),
        "pH": rng.uniform(4, 11, n),
        "time_min": rng.uniform(5, 180, n),
        "stabilizer_pct": rng.uniform(0, 2, n),
    })
    df["size_nm"] = (10 + 1.8 * df.AgNO3_mM - 0.6 * df.reducer_mM + 2.5 * (df.pH - 7) ** 2 / 4
                     + 0.05 * df.time_min - 4 * df.stabilizer_pct + 0.08 * df.temperature_C
                     + rng.normal(0, 3, n))
    miss_t = rng.choice(n, 25, replace=False)              # ~76% missing temperature
    df.loc[miss_t, "temperature_C"] = np.nan
    for c in ["pH", "time_min"]:
        df.loc[rng.choice(n, 3, replace=False), c] = np.nan
    return df


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--target", default="size_nm")
    ap.add_argument("--temp-col", default="temperature_C")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    df = simulate() if a.simulate else pd.read_csv(a.data)
    os.makedirs(a.out, exist_ok=True)
    feats = [c for c in df.columns if c != a.target]
    X, y = df[feats].to_numpy(float), df[a.target].to_numpy(float)
    t = feats.index(a.temp_col) if a.temp_col in feats else 0
    R = a.repeats
    print(f"n={len(df)}  features={feats}")
    print("missing per feature:", df[feats].isna().sum().to_dict())

    summary = {}

    # Exp 0: leaky vs clean ------------------------------------------------
    leaky = leaky_protocol(X, y, t)
    clean = repeated_cv(build("autoencoder", "stacked", 1000, temp_idx=t), X, y, n_repeats=R)
    exp0 = pd.DataFrame([{"protocol": "leaky (impute+augment before split)", **leaky},
                         {"protocol": "clean (all steps inside CV folds)",
                          "R2_mean": clean["R2_mean"], "R2_std": clean["R2_std"]}])
    exp0.to_csv(f"{a.out}/exp0_leaky_vs_clean.csv", index=False)
    print("\nExp0 leaky vs clean\n", exp0.round(3).to_string(index=False))

    # Exp 1: imputation ablation (model fixed = GPR, no augmentation) -----
    rows = []
    for imp in ["mean", "median", "knn", "mice", "autoencoder", "drop_temp+flag"]:
        rows.append({"imputer": imp, **repeated_cv(build(imp, "gpr", temp_idx=t), X, y, n_repeats=R)})
    exp1 = pd.DataFrame(rows)
    exp1.to_csv(f"{a.out}/exp1_imputation.csv", index=False)
    print("\nExp1 imputation\n", exp1.round(3).to_string(index=False))
    best_imp = exp1.loc[exp1.R2_mean.idxmax(), "imputer"]

    # Exp 2: augmentation ablation (inside folds) -------------------------
    rows = []
    for n_syn in [0, 26, 78, 1000]:     # none, 1x, 3x train size, paper's 1000
        for model in ["gpr", "stacked"]:
            rows.append({"n_synthetic": n_syn, "model": model,
                         **repeated_cv(build(best_imp, model, n_syn, temp_idx=t), X, y, n_repeats=R)})
    exp2 = pd.DataFrame(rows)
    exp2.to_csv(f"{a.out}/exp2_augmentation.csv", index=False)
    print("\nExp2 augmentation\n", exp2.round(3).to_string(index=False))

    # Exp 3: model comparison ---------------------------------------------
    models = ["mean_baseline", "ridge", "gpr", "svr", "random_forest", "gbr", "stacked", "mlp_dnn"]
    rows = [{"model": m, **repeated_cv(build(best_imp, m, temp_idx=t), X, y, n_repeats=R)} for m in models]
    exp3 = pd.DataFrame(rows).sort_values("R2_mean", ascending=False)
    exp3.to_csv(f"{a.out}/exp3_models.csv", index=False)
    print("\nExp3 models\n", exp3.round(3).to_string(index=False))
    best_model = exp3.iloc[0]["model"]
    best_pipe = build(best_imp, best_model, temp_idx=t)

    # Exp 4: y-randomization ----------------------------------------------
    rng = np.random.default_rng(SEED)
    perm_r2 = [repeated_cv(best_pipe, X, y, n_repeats=3, y_override=rng.permutation(y))["R2_mean"]
               for _ in range(30)]
    real_r2 = exp3.iloc[0]["R2_mean"]
    p_val = (np.sum(np.array(perm_r2) >= real_r2) + 1) / (len(perm_r2) + 1)
    pd.DataFrame({"permuted_R2": perm_r2}).to_csv(f"{a.out}/exp4_y_randomization.csv", index=False)
    print(f"\nExp4 y-randomization: real R2={real_r2:.3f}  permuted mean={np.mean(perm_r2):.3f}  p={p_val:.3f}")

    # Exp 5: learning curve -----------------------------------------------
    lc = []
    for frac in [0.4, 0.55, 0.7, 0.85, 1.0]:
        sc = []
        for r in range(10):
            idx = np.random.default_rng(SEED + r).choice(len(y), int(round(frac * len(y))), replace=False)
            sc.append(repeated_cv(best_pipe, X[idx], y[idx], n_repeats=1)["R2_mean"])
        lc.append({"n_train_pool": len(idx), "R2_mean": np.mean(sc), "R2_std": np.std(sc)})
    exp5 = pd.DataFrame(lc)
    exp5.to_csv(f"{a.out}/exp5_learning_curve.csv", index=False)
    print("\nExp5 learning curve\n", exp5.round(3).to_string(index=False))

    # Exp 6: permutation importance (on held-out folds) -------------------
    imps = []
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        m = clone(best_pipe).fit(X[tr], y[tr])
        imps.append(permutation_importance(m, X[te], y[te], n_repeats=20,
                                           random_state=SEED).importances_mean)
    exp6 = pd.DataFrame({"feature": feats, "importance_mean": np.mean(imps, 0),
                         "importance_std": np.std(imps, 0)}).sort_values("importance_mean", ascending=False)
    exp6.to_csv(f"{a.out}/exp6_importance.csv", index=False)
    print("\nExp6 permutation importance\n", exp6.round(3).to_string(index=False))

    # Figures ------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].bar(["Leaky", "Clean"], exp0.R2_mean, yerr=exp0.R2_std, color=["#c0504d", "#4f81bd"])
    ax[0, 0].set_title("Exp 0: same model, two protocols"); ax[0, 0].set_ylabel("R²")
    e3 = exp3.sort_values("R2_mean")
    ax[0, 1].barh(e3.model, e3.R2_mean, xerr=e3.R2_std, color="#4f81bd")
    ax[0, 1].set_title("Exp 3: models (repeated 5-fold CV)"); ax[0, 1].set_xlabel("R²")
    ax[1, 0].hist(perm_r2, bins=12, color="#999"); ax[1, 0].axvline(real_r2, color="#c0504d", lw=2)
    ax[1, 0].set_title(f"Exp 4: y-randomization (p={p_val:.3f})"); ax[1, 0].set_xlabel("R²")
    ax[1, 1].errorbar(exp5.n_train_pool, exp5.R2_mean, yerr=exp5.R2_std, marker="o")
    ax[1, 1].set_title("Exp 5: learning curve"); ax[1, 1].set_xlabel("samples"); ax[1, 1].set_ylabel("R²")
    fig.tight_layout(); fig.savefig(f"{a.out}/summary_figure.png", dpi=150)

    summary = {"best_imputer": best_imp, "best_model": best_model, "best_R2": real_r2,
               "leaky_R2": leaky["R2_mean"], "clean_R2_same_setup": clean["R2_mean"],
               "y_random_p": p_val}
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2, default=float)
    print("\nSUMMARY", json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
