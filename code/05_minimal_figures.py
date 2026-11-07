"""
STEP 5 - Figures for the minimally revised manuscript (framework exactly as in the original paper):
figM_workflow, figM_learning_curve, figM_augmentation, figM_S2_original_vs_corrected
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
ORDER = ["Random Forest", "LightGBM", "DNN", "Stacked ensemble (proposed)"]
SHORT = ["RF", "LightGBM", "DNN", "Proposed Stacked\nEnsemble"]


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}.png", dpi=300); plt.close(fig); print("saved", name)


def box(ax, x, y, w, h, text, fc, ec=MUTED, bold=False, fs=8.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3))


# workflow
fig, ax = plt.subplots(figsize=(10.5, 6.6))
ax.set_xlim(0, 10.5); ax.set_ylim(-0.4, 7.0); ax.axis("off"); ax.grid(False)
box(ax, 5.25, 6.55, 6.6, 0.6, "AgNPs synthesis and characterization  +  raw dataset (33 records)\ncleaning and standardization → 30 records from 14 studies", "#eef3fa", bold=True)
arrow(ax, 5.25, 6.25, 5.25, 5.92)
box(ax, 5.25, 5.6, 6.2, 0.6, "Split first: 10-fold cross-validation, repeated 10 times\n(test fold = real records only)", "#fff3e8", ORANGE, bold=True)
arrow(ax, 4.0, 5.3, 2.75, 4.92); arrow(ax, 6.5, 5.3, 7.9, 4.92)
ax.text(2.75, 4.72, "TRAINING FOLD", ha="center", fontsize=9, weight="bold", color=BLUE)
ax.text(7.9, 4.72, "TEST FOLD (real records)", ha="center", fontsize=9, weight="bold", color=ORANGE)
H = 0.62
steps = [("1  Autoencoder imputation\n(fitted on the training fold)", 3.95),
         ("2  Augmentation: Noise Augmentation or\nGaussian Copula (training records only)", 3.15),
         ("3  Base models: RF, LightGBM, DNN", 2.35),
         ("4  Proposed Stacked Ensemble (weights\noptimized on inner CV predictions)", 1.55)]
for i, (t, yy) in enumerate(steps):
    box(ax, 2.75, yy, 4.6, H, t, "#e7f1fb", BLUE)
    if i:
        arrow(ax, 2.75, steps[i - 1][1] - H / 2, 2.75, yy + H / 2, BLUE)
box(ax, 7.9, 3.95, 3.6, H, "Impute with the fitted\nautoencoder", "#f9ece7", ORANGE)
box(ax, 7.9, 2.35, 3.6, H, "Predict AgNP size with\nthe fitted models", "#f9ece7", ORANGE)
box(ax, 7.9, 1.55, 3.6, H, "R², MAE, RMSE on real records\n(mean ± SD over repeats)", "#e5f4e4", AQUA, bold=True)
arrow(ax, 7.9, 3.95 - H / 2, 7.9, 2.35 + H / 2, ORANGE); arrow(ax, 7.9, 2.35 - H / 2, 7.9, 1.55 + H / 2, ORANGE)
arrow(ax, 5.05, 3.95, 6.1, 3.95); ax.text(5.57, 4.05, "fitted imputer", ha="center", fontsize=7.5, color=MUTED)
arrow(ax, 5.05, 1.55, 6.1, 2.22); ax.text(5.5, 2.02, "fitted models", ha="center", fontsize=7.5, color=MUTED, rotation=30)
box(ax, 5.25, 0.55, 8.6, 0.55, "Nested cross-validation: imputation and augmentation method selected inside each training fold", "#efedff", "#7a6ad8", fs=8.4)
ax.text(5.25, -0.2, "Synthetic records are used only for training; every test record is a real record never seen during training.",
        ha="center", fontsize=8.5, color=INK, style="italic")
save(fig, "figM_workflow")

# learning curve
lc = pd.read_csv(f"{R}/E_learning_curve_cv.csv").rename(columns={"n_train": "n_records"})
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for a, lab, c in [("copula", "Gaussian Copula", BLUE), ("noise", "Noise Augmentation", ORANGE), ("none", "No augmentation", GREY)]:
    g = lc[lc.augmentation == a].groupby("n_records").R2.agg(["mean", "std"]).reset_index()
    ax.errorbar(g.n_records, g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=3, lw=1.8, color=c, label=lab)
ax.set_xlabel("Number of real training records (test folds fixed)"); ax.set_ylabel("R² on real held-out records")
ax.set_title("Proposed Stacked Ensemble: learning curve", loc="left", fontsize=10)
ax.set_xticks(sorted(lc.n_records.unique())); ax.legend(frameon=False, fontsize=8.5)
save(fig, "figM_learning_curve")

# augmentation comparison
e = pd.read_csv(f"{R}/E_original_framework.csv")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax, met, lab in [(axes[0], "R2", "R²"), (axes[1], "MAE", "MAE (nm)")]:
    x = np.arange(len(ORDER)); w = 0.26
    for i, (a, nm, c) in enumerate([("copula", "Gaussian Copula", BLUE), ("noise", "Noise Augmentation", ORANGE), ("none", "No augmentation", GREY)]):
        s = e[e.augmentation == a].set_index("model").loc[ORDER]
        ax.bar(x + (i - 1) * w, s[met], w, yerr=s[met + "_sd"], capsize=2, color=c, label=nm)
    ax.set_xticks(x, SHORT, fontsize=8.5); ax.set_ylabel(lab)
    if met == "R2":
        ax.set_ylim(0, 1)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, loc="lower center", ncol=3)
fig.suptitle("Performance on real held-out records (10-fold CV repeated 10 times; error bars: SD)", x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{FIG}/figM_augmentation.png", dpi=300); plt.close(fig); print("saved figM_augmentation")

# S2 original protocol vs corrected
o = pd.read_csv(f"{R}/O_original_protocol.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, (a_o, a_e, title) in zip(axes, [("Gaussian copula", "copula", "Gaussian Copula"), ("Noise augmentation", "noise", "Noise Augmentation")]):
    so = o[o.augmentation == a_o].set_index("model")
    se = e[e.augmentation == a_e].set_index("model")
    labs = ["RF", "LightGBM", "DNN"]
    x = np.arange(3)
    vo = [so.loc[m].R2 for m in ["Random Forest", "LightGBM", "DNN"]]
    ve = [se.loc[m].R2 for m in ["Random Forest", "LightGBM", "DNN"]]
    ax.bar(x - 0.2, vo, 0.4, color=GREY, label="Augmentation before split (synthetic test records)")
    ax.bar(x + 0.2, ve, 0.4, color=BLUE, label="Augmentation within training folds (real test records)")
    for xi, v in zip(x - 0.2, vo): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + 0.2, ve): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, labs); ax.set_title(title, fontsize=10); ax.set_ylim(0, 1.1)
axes[0].set_ylabel("R²")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=8.5)
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(f"{FIG}/figM_S2_original_vs_corrected.png", dpi=300); plt.close(fig)
print("saved figM_S2")

# parity (graphical abstract)
oo = pd.read_csv(f"{R}/E_oof.csv")
g = pd.read_csv(f"{R}/G_nested_original.csv").set_index("model").loc["Stacked ensemble (proposed)"]
fig, ax = plt.subplots(figsize=(4.8, 4.4))
lim = [0, max(oo.measured_nm.max(), oo.pred_copula.max()) * 1.08]
ax.plot(lim, lim, color=GREY, lw=1, ls="--")
ax.scatter(oo.measured_nm, oo.pred_copula, s=34, color=BLUE, edgecolor="white", lw=0.6)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("Measured size (nm)"); ax.set_ylabel("Predicted size (nm)")
ax.set_title("Proposed Stacked Ensemble (Gaussian Copula)", loc="left", fontsize=9.5)
save(fig, "figM_parity")
