# Minimal revision of the manuscript (November 2026)

The minimally revised manuscript keeps the original text and framework exactly as written
(autoencoder imputation + noise augmentation / Gaussian copula + RF, LightGBM, DNN and the Proposed Stacked Ensemble;
no physics descriptors). Its numbers come from these files:

| Manuscript item | Script | Result file |
|---|---|---|
| Tables 5 and 6, Fig. 7, Discussion numbers | 02e_original_framework.py | results/E_original_framework.csv, E_weights.csv |
| Table 4 (nested CV, 9 configurations: imputer x augmentation) | 02f_nested_original.py | results/G_nested_original*.csv |
| Fig. 6 (learning curve, fixed real test folds) | 02g_learning_curve_cv.py | results/E_learning_curve_cv.csv |
| Table 3 (synthetic-data quality inside training folds, DCR) | 02d_synthetic_quality.py | results/S_synthetic_quality.csv |
| Table S3 (leave-one-study-out) | 02e_original_framework.py | results/E_loso_copula.csv |
| Figs. 2, 6, 7, S2 and the graphical abstract plot | 05_minimal_figures.py | figures/figM_*.png |

The other result files (physics descriptors, 72-configuration nested CV, ablation, etc.) belong to the extended analysis
described in [REVISION_REPORT.md](REVISION_REPORT.md) and are not used in the minimally revised manuscript.
