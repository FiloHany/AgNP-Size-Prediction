"""
STEP 3 - All figures (300 dpi) from ../results -> ../figures
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

R, FIG = "../results", "../figures"
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#b7b6b0", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True})
ORDER = ["Random Forest", "LightGBM", "DNN", "Average ensemble", "Stacked ensemble (proposed)"]
SHORT = ["RF", "LightGBM", "DNN", "Average\nensemble", "Stacked\nensemble\n(proposed)"]


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=300)
    plt.close(fig)
    print("saved", name)


def box(ax, x, y, w, h, text, fc, ec=MUTED, bold=False, fs=8.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3))


# ---------------------------------------------------------------- Fig 1 workflow
fig, ax = plt.subplots(figsize=(10.5, 7.2))
ax.set_xlim(0, 10.5); ax.set_ylim(-0.4, 7.6); ax.axis("off"); ax.grid(False)
box(ax, 5.25, 7.15, 6.2, 0.6, "Raw records: 30 syntheses from 14 published studies\n(10 inputs, missing values left empty)", "#eef3fa", bold=True)
arrow(ax, 5.25, 6.85, 5.25, 6.5)
box(ax, 5.25, 6.18, 6.2, 0.6, "Split FIRST: 10-fold cross-validation, repeated\n(test fold = real records only)", "#fff3e8", ORANGE, bold=True)
arrow(ax, 4.0, 5.88, 2.75, 5.5); arrow(ax, 6.5, 5.88, 7.9, 5.5)
ax.text(2.75, 5.3, "TRAINING FOLD (real records)", ha="center", fontsize=9, weight="bold", color=BLUE)
ax.text(7.9, 5.3, "TEST FOLD (real records only)", ha="center", fontsize=9, weight="bold", color=ORANGE)
H = 0.62
steps = [("0  Nested selection: inner CV chooses imputer,\naugmentation, physics features, target scale, indicators", 4.75, "#efedff", "#7a6ad8"),
         ("1  Imputation (+ missing-value indicators)", 3.95, "#e7f1fb", BLUE),
         ("2  Augmentation from training rows only (if selected)", 3.15, "#e7f1fb", BLUE),
         ("3  Physics features: power V·I, field V/gap,\ncurrent density I/area", 2.35, "#e7f1fb", BLUE),
         ("4  Base models: RF, LightGBM, DNN", 1.55, "#e7f1fb", BLUE),
         ("5  Stacked ensemble (weights learned from\ninner out-of-fold predictions)", 0.75, "#e7f1fb", BLUE)]
for i, (t, yy, fc, ec) in enumerate(steps):
    box(ax, 2.75, yy, 4.6, H, t, fc, ec)
    if i:
        arrow(ax, 2.75, steps[i - 1][1] - H / 2, 2.75, yy + H / 2, BLUE)
box(ax, 7.9, 3.95, 3.6, H, "Impute with the imputer\nfitted on the training fold", "#f9ece7", ORANGE)
box(ax, 7.9, 1.55, 3.6, H, "Predict size with the\nfitted framework", "#f9ece7", ORANGE)
box(ax, 7.9, 0.75, 3.6, H, "R², MAE, RMSE on real records\n(mean ± SD over repeats)", "#e5f4e4", AQUA, bold=True)
arrow(ax, 7.9, 3.95 - H / 2, 7.9, 1.55 + H / 2, ORANGE); arrow(ax, 7.9, 1.55 - H / 2, 7.9, 0.75 + H / 2, ORANGE)
arrow(ax, 5.05, 3.95, 6.1, 3.95); ax.text(5.57, 4.05, "fitted imputer", ha="center", fontsize=7.5, color=MUTED)
arrow(ax, 5.05, 0.75, 6.1, 1.42); ax.text(5.5, 1.22, "fitted models", ha="center", fontsize=7.5, color=MUTED, rotation=30)
ax.text(5.25, -0.2, "Synthetic rows (if any) are used only for training; every test record is a real record that the imputer,\n"
        "the augmentation, the selection step and the models never saw.", ha="center", fontsize=8.5, color=INK, style="italic")
save(fig, "fig01_workflow")

# ---------------------------------------------------------------- Fig 2 missing values
m = pd.read_csv("../data/missing_values_v3.csv", index_col=0).iloc[:, 0].drop("Nano size", errors="ignore").sort_values()
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(m.index, m.values, color=BLUE, height=0.6)
for i, v in enumerate(m.values):
    ax.annotate(f"{v} ({v / 30:.0%})", (v + 0.3, i), va="center", fontsize=8.5, color=INK)
ax.set_xlim(0, 30); ax.set_xlabel("missing values (of 30 records)")
ax.set_title("Missing values per input", loc="left")
save(fig, "fig02_missing_values")

# ---------------------------------------------------------------- Fig 3 by augmentation (Tables 5/6)
t = pd.read_csv(f"{R}/T5_T6_by_augmentation.csv")
augs = ["Gaussian copula", "Noise augmentation", "No augmentation"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
lo = max(-1, min(0, t.R2.min() - 0.1))
for ax, aug in zip(axes, augs):
    s = t[t.augmentation == aug].set_index("model").loc[ORDER]
    vals = s.R2.clip(lower=lo)
    cols = [ORANGE if "proposed" in n else BLUE for n in s.index]
    ax.bar(range(5), vals, color=cols, width=0.62)
    ax.errorbar(range(5), vals, yerr=s.R2_sd, fmt="none", ecolor=INK, capsize=3, lw=1)
    for i, (v, raw, sd) in enumerate(zip(vals, s.R2, s.R2_sd)):
        ax.annotate(f"{raw:.2f}", (i, max(v, 0) + sd + 0.03), ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(5), SHORT); ax.set_ylim(lo, 1); ax.set_title(aug, loc="left")
axes[0].set_ylabel("R² on real held-out records")
fig.suptitle("Performance on real held-out records by augmentation strategy (mean ± SD)", x=0.01, ha="left", fontsize=10)
save(fig, "fig03_performance_by_augmentation")

# ---------------------------------------------------------------- Fig 4 synthetic vs real
o = pd.read_csv(f"{R}/O_original_protocol.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
for ax, aug in zip(axes, ["Gaussian copula", "Noise augmentation"]):
    a = o[o.augmentation == aug].set_index("model"); b = t[t.augmentation == aug].set_index("model")
    names = ["Random Forest", "LightGBM", "DNN"]
    sv = [a.loc[n, "R2"] for n in names] + [a.loc["Average ensemble (original)", "R2"]]
    rv = [b.loc[n, "R2"] for n in names] + [b.loc["Stacked ensemble (proposed)", "R2"]]
    x = np.arange(4); w = 0.38
    ax.bar(x - w / 2 - 0.01, sv, w, color=GREY, label="Synthetic test rows (original protocol)")
    ax.bar(x + w / 2 + 0.01, np.clip(rv, 0, None), w, color=BLUE, label="Real held-out records (revised framework)")
    for i, (s_, r_) in enumerate(zip(sv, rv)):
        ax.annotate(f"{s_:.2f}", (i - w / 2, s_ + 0.02), ha="center", fontsize=8, color=INK)
        ax.annotate(f"{r_:.2f}", (i + w / 2, max(r_, 0) + 0.02), ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 1.12); ax.set_xticks(x, ["RF", "LightGBM", "DNN", "Ensemble"]); ax.set_title(aug, loc="left")
axes[0].set_ylabel("R²")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=2)
fig.suptitle("Synthetic test rows (original) vs real held-out records (revised)", x=0.01, ha="left", fontsize=10)
save(fig, "fig04_synthetic_vs_real_test", rect=(0, 0.08, 1, 1))

# ---------------------------------------------------------------- Fig 5 development
dv = pd.read_csv(f"{R}/D_development.csv")
fig, ax = plt.subplots(figsize=(10, 0.48 * len(dv) + 1.3))
yy = np.arange(len(dv))[::-1]
cols = [GREY if i == 0 else (ORANGE if s.startswith("Nested") else (AQUA if s.startswith("Final") else BLUE))
        for i, s in enumerate(dv.step)]
ax.barh(yy, dv.R2, xerr=dv.R2_sd, color=cols, height=0.6, capsize=2)
for yv, (r, sd) in zip(yy, zip(dv.R2, dv.R2_sd)):
    ax.annotate(f"{r:.2f} ± {sd:.2f}", (1.01, yv), xycoords=("axes fraction", "data"), va="center", fontsize=8, color=INK)
def short(s):
    if s.startswith("Final configuration"):
        return "Final configuration (most selected in nested CV): mean imputation,\nno augmentation, physics features, stacked ensemble"
    return s
ax.set_yticks(yy, [short(s) for s in dv.step], fontsize=8)
ax.set_xlim(0, 1); ax.set_xlabel("R² of the stacked ensemble on real held-out records")
fig.suptitle("Effect of each revision step (grey: previous version; aqua: final settings; orange: nested CV, main result)", x=0.01, ha="left", fontsize=9.5)
save(fig, "fig05_development_steps")

# ---------------------------------------------------------------- Fig 6 ablation
ab = pd.read_csv(f"{R}/A_ablation.csv").sort_values("R2")
ab["variant"] = ab.variant.replace({"Final framework (stacked ensemble)": "Best configuration (stacked ensemble)",
                                    "Single model: Random Forest": "Random Forest only", "Single model: LightGBM": "LightGBM only",
                                    "Single model: DNN": "DNN only", "Without physics features": "Without physics-informed descriptors",
                                    "Log-transformed target": "Logarithmic target", "With missing indicators": "With missing-value indicators"})
lo = max(-1, min(0, ab.R2.min() - 0.1))
fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(ab) + 1.2))
ax.barh(ab.variant, ab.R2.clip(lower=lo), xerr=ab.R2_sd, color=[ORANGE if v.startswith("Best") else BLUE for v in ab.variant],
        height=0.6, capsize=2)
for i, (r, s) in enumerate(zip(ab.R2, ab.R2_sd)):
    ax.annotate(f"{r:.2f} ± {s:.2f}", (1.02, i), xycoords=("axes fraction", "data"), va="center", fontsize=8, color=INK)
ax.axvline(0, color=INK, lw=0.8); ax.set_xlim(lo, 1)
ax.set_xlabel("R² on real held-out records (mean ± SD)"); ax.set_title("Ablation around the best configuration", loc="left")
save(fig, "fig06_ablation")

# ---------------------------------------------------------------- Fig 7 parity
oof = pd.read_csv(f"{R}/N_oof_predictions.csv")
nested = pd.read_csv(f"{R}/N_nested_performance.csv").set_index("model").loc["Stacked ensemble (proposed)"]
fig, ax = plt.subplots(figsize=(4.8, 4.5))
ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--")
ax.scatter(oof.measured_nm, oof["Stacked ensemble (proposed)"], s=42, color=BLUE, edgecolor="white", lw=1, zorder=3)
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xlabel("Measured size (nm)"); ax.set_ylabel("Predicted size (nm), held-out")
ax.set_title(f"Nested CV: R² = {nested.R2:.2f} ± {nested.R2_sd:.2f}, MAE = {nested.MAE:.1f} nm", loc="left", fontsize=9.5)
save(fig, "fig07_parity")

# ---------------------------------------------------------------- Fig 8 learning curve
lc = pd.read_csv(f"{R}/C_learning_curve.csv").groupby("n_records").R2.agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.errorbar(lc.n_records, lc["mean"], yerr=lc["std"].fillna(0), marker="o", ms=6, lw=2, color=BLUE, capsize=3)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xlabel("number of real records used"); ax.set_ylabel("R² (stacked ensemble)")
ax.set_title("Learning curve over the number of real records", loc="left")
save(fig, "fig08_learning_curve")

# ---------------------------------------------------------------- Fig 9 y-randomization
yr = pd.read_csv(f"{R}/Y_y_randomization.csv").iloc[:, 0]
final = pd.read_csv(f"{R}/A_ablation.csv").set_index("variant").loc["Final framework (stacked ensemble)"]
p = (np.sum(yr >= final.R2) + 1) / (len(yr) + 1)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(yr, bins=12, color=GREY, edgecolor="white"); ax.axvline(final.R2, color=ORANGE, lw=2)
ax.annotate(f"real targets\nR² = {final.R2:.2f}", (final.R2, ax.get_ylim()[1] * 0.8), xytext=(-6, 0),
            textcoords="offset points", ha="right", fontsize=9)
ax.set_xlabel("R² with shuffled targets"); ax.set_ylabel("count")
ax.set_title(f"y-randomization ({len(yr)} permutations), p = {p:.3f}", loc="left")
save(fig, "fig09_y_randomization")

# ---------------------------------------------------------------- Fig 10 importance
im = pd.read_csv(f"{R}/I_feature_importance.csv").sort_values("MAE_increase_nm")
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.barh(im.feature, im.MAE_increase_nm, xerr=im.sd, color=BLUE, height=0.6, capsize=2)
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("increase in MAE when the input is permuted (nm)")
ax.set_title("Permutation importance (best configuration)", loc="left")
save(fig, "fig10_feature_importance")

# ---------------------------------------------------------------- Fig S1 masking
ms = pd.read_csv(f"{R}/M_masking_test.csv", index_col=0)
cols = [c for c in ms.columns if c != "ALL"]
fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(len(cols)); w = 0.16
pal = {"Autoencoder": ORANGE, "KNN": BLUE, "Mean": GREY, "Median": "#8a8983", "MICE": AQUA}
for i, imp in enumerate(ms.index):
    ax.bar(x + (i - 2) * w, ms.loc[imp, cols].clip(upper=3), w, color=pal.get(imp, BLUE), label=f"{imp} (overall {ms.loc[imp, 'ALL']:.2f})")
    for j, c in enumerate(cols):
        if ms.loc[imp, c] > 3:
            ax.annotate(f"{ms.loc[imp, c]:.1f} (cut)", (x[j] + (i - 2) * w, 3.02), ha="center", fontsize=7.5, color=INK)
ax.set_ylim(0, 3.3)
ax.axhline(1, color=INK, lw=0.8, ls="--")
ax.set_xticks(x, cols, rotation=20, ha="right"); ax.set_ylabel("RMSE on hidden values (SD units)")
ax.set_title("Imputation accuracy on artificially hidden values (1.0 = guessing the column mean)", loc="left")
ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.38))
save(fig, "figS1_masking_test")
