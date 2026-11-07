"""
Version 3 of the paper's framework (leakage-free, improved).

Per training fold (nothing is ever fitted on test records):
  1. imputation of missing inputs           (Autoencoder / Mean / KNN)
  2. missing-value indicators               (1 = the value was not reported)
  3. data augmentation                      (none / noise / Gaussian copula), training rows only
  4. physics-based features                 power P = V*I, field E = V/gap, current density J = I/area
  5. base models on log(size)               Random Forest, LightGBM, DNN
  6. stacked ensemble                       learned weighted average (weights >= 0, sum = 1)
The imputer, the augmentation, and whether to use physics features, a log target and
missing indicators (72 combinations) are selected by NESTED cross-validation
(an inner CV inside each outer training fold), so the choice never sees the test records.
"""
import itertools

import lightgbm as lgb
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import nnls
from scipy.stats import norm
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from agnp_pipeline import AutoencoderImputer

SEED = 42
N_JOBS = 2
BASE = ["Random Forest", "LightGBM", "DNN"]
FEATS = ["Current", "Voltage", "Yield", "Zeta potential", "pH", "Temperature", "Volume of beaker",
         "Electrode gap", "Electrode dimension 1", "Electrode dimension 2"]
IND_COLS = ["Yield", "Zeta potential", "Temperature", "Volume of beaker", "Electrode gap"]
PHYS = ["log Power (V*I)", "log Field (V/gap)", "log Current density (I/area)"]
MODEL_NAMES = BASE + ["Average ensemble", "Stacked ensemble (proposed)"]


# ----------------------------------------------------------------- components
def make_imputer(name):
    return {
        "Autoencoder": lambda: AutoencoderImputer(),
        "KNN": lambda: KNNImputer(n_neighbors=3, keep_empty_features=True),
        "Mean": lambda: SimpleImputer(strategy="mean", keep_empty_features=True),
        "Median": lambda: SimpleImputer(strategy="median", keep_empty_features=True),
        "MICE": lambda: IterativeImputer(max_iter=20, random_state=SEED, keep_empty_features=True),
    }[name]()


def make_base(seed=SEED, log_target=True, rf_max_features=1.0):
    log_t = dict(func=np.log, inverse_func=np.exp, check_inverse=False)
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5,
                               max_features=rf_max_features, random_state=seed, n_jobs=1)
    lg = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=3,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                           n_jobs=1, verbose=-1)
    dnn = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), learning_rate_init=1e-3, batch_size=32, max_iter=200,
        early_stopping=True, validation_fraction=0.2, n_iter_no_change=15, random_state=seed))
    if log_target:
        ytf = Pipeline([("log", FunctionTransformer(**log_t)), ("std", StandardScaler())])
        return {"Random Forest": TransformedTargetRegressor(rf, **log_t),
                "LightGBM": TransformedTargetRegressor(lg, **log_t),
                "DNN": TransformedTargetRegressor(dnn, transformer=ytf)}
    return {"Random Forest": rf, "LightGBM": lg,
            "DNN": TransformedTargetRegressor(dnn, transformer=StandardScaler())}


def physics(X):
    """X: imputed base inputs in FEATS order -> 3 log-scaled physics features."""
    I, V = X[:, FEATS.index("Current")], X[:, FEATS.index("Voltage")]
    gap = np.clip(X[:, FEATS.index("Electrode gap")], 0.01, None)
    d1 = np.clip(X[:, FEATS.index("Electrode dimension 1")], 0.1, None)
    I, V = np.clip(I, 1e-3, None), np.clip(V, 1e-3, None)
    return np.column_stack([np.log(V * I), np.log(V / gap), np.log(I / (np.pi * (d1 / 2) ** 2))])


def gaussian_copula_sample(Z, n, rng):
    m = Z.shape[0]
    ranks = np.argsort(np.argsort(Z, 0), 0) + 1
    G = norm.ppf(ranks / (m + 1))
    C = np.corrcoef(G.T)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    U = norm.cdf(rng.multivariate_normal(np.zeros(Z.shape[1]), C + 1e-6 * np.eye(Z.shape[1]), n, check_valid="ignore"))
    return np.column_stack([np.quantile(Z[:, j], U[:, j]) for j in range(Z.shape[1])])


def augment(X, y, n_ind, method, mult, rng, noise_frac=0.05):
    """Synthetic rows from training rows only. The last n_ind columns are 0/1 indicators."""
    n = int(round(mult * len(y)))
    if method == "none" or n == 0:
        return X, y
    if method == "noise":
        i = rng.integers(0, len(X), n)
        Xs = X[i].copy()
        k = X.shape[1] - n_ind
        Xs[:, :k] += rng.normal(0, 1, (n, k)) * noise_frac * (X[:, :k].std(0) + 1e-12)
        ys = y[i] + rng.normal(0, 1, n) * noise_frac * y.std()
    else:
        S = gaussian_copula_sample(np.column_stack([X, y]), n, rng)
        Xs, ys = S[:, :-1], S[:, -1]
        if n_ind:
            Xs[:, -n_ind:] = np.round(Xs[:, -n_ind:])
    return np.vstack([X, Xs]), np.concatenate([y, np.clip(ys, 0.5, None)])


# ------------------------------------------------------------------ framework
DEFAULT = dict(imputer="Autoencoder", aug="noise", mult=1.0, log_target=False, physics=False,
               indicators=False, rf_max_features=1.0)


class Framework:
    def __init__(self, cfg, imp, ind_idx, models, w, lo, hi):
        self.cfg, self.imp, self.ind_idx, self.models, self.w, self.lo, self.hi = cfg, imp, ind_idx, models, w, lo, hi

    def design(self, X):
        Xi = self.imp.transform(X)
        parts = [Xi]
        if self.cfg["indicators"]:
            parts.append(np.isnan(X[:, self.ind_idx]).astype(float))
        return np.column_stack(parts)

    def predict_all(self, X):
        Z = add_physics(self.design(X), self.cfg)
        P = np.column_stack([np.clip(self.models[n].predict(Z), self.lo, self.hi) for n in BASE])
        out = {n: P[:, j] for j, n in enumerate(BASE)}
        out["Average ensemble"] = P.mean(1)
        out["Stacked ensemble (proposed)"] = P @ self.w
        return out

    def predict(self, X):
        return self.predict_all(X)["Stacked ensemble (proposed)"]

    @property
    def weights(self):
        return dict(zip(BASE, self.w))


def add_physics(Z, cfg):
    if not cfg["physics"]:
        return Z
    return np.column_stack([Z, physics(Z[:, :len(FEATS)])])


def _fit_bases(Z, y, n_ind, cfg, rng, seed):
    Za, ya = augment(Z, y, n_ind, cfg["aug"], cfg["mult"], rng)
    Za = add_physics(Za, cfg)
    return {n: m.fit(Za, ya) for n, m in make_base(seed, cfg["log_target"], cfg["rf_max_features"]).items()}


def fit_framework(Xtr, ytr, cfg=None, seed=SEED, inner_k=3):
    cfg = DEFAULT | (cfg or {})
    rng = np.random.default_rng(seed)
    ind_idx = [FEATS.index(c) for c in IND_COLS] if cfg["indicators"] else []
    imp = make_imputer(cfg["imputer"]).fit(Xtr)
    proto = Framework(cfg, imp, ind_idx, None, None, None, None)
    Z = proto.design(Xtr)
    n_ind = len(ind_idx)
    lo, hi = ytr.min(), ytr.max()
    oof = np.zeros((len(ytr), len(BASE)))
    for itr, iva in KFold(inner_k, shuffle=True, random_state=seed).split(Z):
        ms = _fit_bases(Z[itr], ytr[itr], n_ind, cfg, rng, seed)
        Zv = add_physics(Z[iva], cfg)
        for j, n in enumerate(BASE):
            oof[iva, j] = np.clip(ms[n].predict(Zv), lo, hi)
    w, _ = nnls(oof, ytr)
    w = w / w.sum() if w.sum() > 0 else np.full(len(BASE), 1 / len(BASE))
    models = _fit_bases(Z, ytr, n_ind, cfg, rng, seed)
    return Framework(cfg, imp, ind_idx, models, w, lo, hi)


# ------------------------------------------------------- nested selection
GRID = [dict(imputer=i, aug=a, mult=(0.0 if a == "none" else 1.0), physics=p, log_target=lg, indicators=ind)
        for i, a, p, lg, ind in itertools.product(["Autoencoder", "Mean", "KNN"], ["none", "noise", "copula"],
                                                  [True, False], [True, False], [True, False])]


def cfg_name(c):
    return (f"imputer={c['imputer']} | aug={c['aug']} | physics={'yes' if c['physics'] else 'no'} | "
            f"log-target={'yes' if c['log_target'] else 'no'} | indicators={'yes' if c['indicators'] else 'no'}")


def select_config(Xtr, ytr, seed, inner_k=3, grid=GRID):
    """Choose the configuration with the lowest inner-CV MAE (training records only)."""
    best, best_mae, scores = None, np.inf, {}
    folds = list(KFold(inner_k, shuffle=True, random_state=seed + 1000).split(Xtr))
    for c in grid:
        p = np.zeros(len(ytr))
        for itr, iva in folds:
            p[iva] = fit_framework(Xtr[itr], ytr[itr], c, seed=seed).predict(Xtr[iva])
        mae = mean_absolute_error(ytr, p)
        scores[cfg_name(c)] = mae
        if mae < best_mae:
            best, best_mae = c, mae
    return best, scores


# ---------------------------------------------------------------- evaluation
def metrics(y, p):
    return r2_score(y, p), mean_absolute_error(y, p), mean_squared_error(y, p) ** 0.5


def outer_folds(n, k, seed):
    return list(KFold(k, shuffle=True, random_state=seed).split(np.arange(n)))


def _fold(X, y, tr, te, cfg, seed, nested):
    chosen = None
    if nested:
        chosen, _ = select_config(X[tr], y[tr], seed)
        cfg = (cfg or {}) | chosen
    fw = fit_framework(X[tr], y[tr], cfg, seed=seed)
    return te, fw.predict_all(X[te]), fw.weights, (cfg_name(chosen) if chosen else None)


def summarise(y, oofs, repeats):
    out = {}
    for m in MODEL_NAMES:
        a = np.array([metrics(y, oofs[m][r]) for r in range(repeats)])
        out[m] = dict(R2=a[:, 0].mean(), R2_sd=a[:, 0].std(), MAE=a[:, 1].mean(), MAE_sd=a[:, 1].std(),
                      RMSE=a[:, 2].mean(), RMSE_sd=a[:, 2].std())
    return out


def evaluate(X, y, cfg=None, repeats=10, k=10, nested=False, n_jobs=N_JOBS):
    """Repeated k-fold CV; metrics on pooled out-of-fold predictions of each repeat (mean ± SD)."""
    tasks = [(r, tr, te) for r in range(repeats) for tr, te in outer_folds(len(y), k, SEED + r)]
    res = Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, tr, te, cfg, SEED + r, nested) for r, tr, te in tasks)
    oofs = {m: np.zeros((repeats, len(y))) for m in MODEL_NAMES}
    weights, chosen = [], []
    for (r, _, _), (te, preds, w, ch) in zip(tasks, res):
        for m in MODEL_NAMES:
            oofs[m][r, te] = preds[m]
        weights.append(w | {"repeat": r})
        chosen.append(ch)
    return summarise(y, oofs, repeats), weights, {m: oofs[m].mean(0) for m in MODEL_NAMES}, chosen


def evaluate_groups(X, y, groups, cfg=None, n_jobs=N_JOBS):
    ug = np.unique(groups)
    tasks = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in ug]
    res = Parallel(n_jobs=n_jobs)(delayed(_fold)(X, y, tr, te, cfg, SEED, False) for tr, te in tasks)
    oof = {m: np.zeros(len(y)) for m in MODEL_NAMES}
    for te, preds, _, _ in res:
        for m in MODEL_NAMES:
            oof[m][te] = preds[m]
    return {m: dict(zip(["R2", "MAE", "RMSE"], metrics(y, oof[m]))) for m in MODEL_NAMES}, oof
