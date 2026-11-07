"""
STEP 4 - Paper-ready tables (mean ± SD) in one Excel workbook: ../results/Paper_Tables.xlsx
"""
import numpy as np
import pandas as pd

R = "../results"
ORDER = ["Random Forest", "LightGBM", "DNN", "Average ensemble", "Stacked ensemble (proposed)"]


def pm(m, s, nd=2):
    return f"{m:.{nd}f}" if pd.isna(s) else f"{m:.{nd}f} ± {s:.{nd}f}"


def perf(df, keys):
    out = df[keys].copy()
    if "MAE_sd" in df:
        out["MAE (nm)"] = [pm(a, b) for a, b in zip(df.MAE, df.MAE_sd)]
        out["RMSE (nm)"] = [pm(a, b) for a, b in zip(df.RMSE, df.RMSE_sd)]
        out["R²"] = [pm(a, b, 3) for a, b in zip(df.R2, df.R2_sd)]
    else:
        out["MAE (nm)"] = df.MAE.round(2); out["RMSE (nm)"] = df.RMSE.round(2); out["R²"] = df.R2.round(3)
    return out


S = {}
n = pd.read_csv(f"{R}/N_nested_performance.csv").set_index("model").loc[ORDER].reset_index()
S["Main_Nested_CV_result"] = perf(n, ["model"])
sel = pd.read_csv(f"{R}/N_selected_configs.csv")
sel["share"] = (sel.times_selected / sel.times_selected.sum()).map(lambda v: f"{v:.0%}")
S["Nested_selected_configs"] = sel
t = pd.read_csv(f"{R}/T5_T6_by_augmentation.csv")
for aug, name in [("Gaussian copula", "Table5_Copula"), ("Noise augmentation", "Table6_Noise"), ("No augmentation", "Table_No_augmentation")]:
    S[name] = perf(t[t.augmentation == aug].set_index("model").loc[ORDER].reset_index(), ["model"])
ti = pd.read_csv(f"{R}/T_framework_as_titled.csv")
for fw in ti.framework.unique():
    S["Titled_" + ("Copula" if "copula" in fw else "Noise")] = perf(ti[ti.framework == fw].set_index("model").loc[ORDER].reset_index(), ["model"])
S["Development_steps"] = perf(pd.read_csv(f"{R}/D_development.csv"), ["step"])
S["Ablation"] = perf(pd.read_csv(f"{R}/A_ablation.csv"), ["variant"])
w = pd.read_csv(f"{R}/W_ensemble_weights.csv")
wn = pd.read_csv(f"{R}/W_ensemble_weights_nested.csv")
S["Ensemble_weights"] = pd.DataFrame({"base model": w["base model"],
                                      "weight, final configuration": [pm(a, b) for a, b in zip(w["mean"], w["std"])],
                                      "weight, nested CV": [pm(a, b) for a, b in zip(wn["mean"], wn["std"])]})
lo = pd.read_csv(f"{R}/L_leave_one_study_out.csv").set_index("model").loc[ORDER].reset_index()
S["Leave_one_study_out"] = perf(lo, ["model"])
o = pd.read_csv(f"{R}/O_original_protocol.csv")
S["Original_protocol_synthetic"] = perf(o, ["augmentation", "model"])
yr = pd.read_csv(f"{R}/Y_y_randomization.csv").iloc[:, 0]
fin = pd.read_csv(f"{R}/A_ablation.csv").set_index("variant").loc["Final framework (stacked ensemble)"]
S["y_randomization"] = pd.DataFrame({"quantity": ["R² with real targets", "R² shuffled, mean", "R² shuffled, max", "permutations", "p-value"],
                                     "value": [f"{fin.R2:.3f}", f"{yr.mean():.3f}", f"{yr.max():.3f}", len(yr),
                                               f"{(np.sum(yr >= fin.R2) + 1) / (len(yr) + 1):.3f}"]})
lc = pd.read_csv(f"{R}/C_learning_curve.csv").groupby("n_records").R2.agg(["mean", "std"]).reset_index()
S["Learning_curve"] = pd.DataFrame({"records": lc.n_records, "R²": [pm(a, b, 3) for a, b in zip(lc["mean"], lc["std"])]})
fi = pd.read_csv(f"{R}/I_feature_importance.csv")
S["Feature_importance"] = pd.DataFrame({"input": fi.feature, "MAE increase (nm)": [pm(a, b) for a, b in zip(fi.MAE_increase_nm, fi.sd)]})
S["Masking_test"] = pd.read_csv(f"{R}/M_masking_test.csv", index_col=0).round(2).reset_index().rename(columns={"ALL": "Overall"})
S["Configuration"] = pd.read_csv(f"{R}/H_configuration.csv")
S["Data_record_changes"] = pd.read_csv("../data/v3_record_changes.csv")
pn = pd.read_csv("../data/parsing_notes.csv")
S["Data_parsing_notes"] = pn[~pn.Ref.astype(str).isin(["15", "1,18"])]  # Ref 15 removed, Ref 18 merged in version 3

with pd.ExcelWriter(f"{R}/Paper_Tables.xlsx", engine="openpyxl") as xw:
    for name, df in S.items():
        df.to_excel(xw, sheet_name=name[:31], index=False)
        ws = xw.sheets[name[:31]]
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(12, width + 2), 95)
for name, df in S.items():
    print(f"\n== {name}\n{df.to_string(index=False)}")
