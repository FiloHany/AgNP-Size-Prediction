# AgNP Size Prediction

**Manuscript:** *A Machine Learning Framework for Silver Nanoparticle Size Prediction: Integrating Autoencoder Imputation, Noise Augmentation, and Stacked Ensemble Learning*
**Prepared by:** Eng. Filopater Hany (AI part) · November 2026
**Purpose of this package:** the same scope, title and framework as the submitted paper and as package v2, now with the improvements discussed after v2: a cleaner dataset, 10-fold CV, physics-based input features, nested selection of the framework settings, and the analyses the reviewers requested. There is **no data leakage**: every test record is a real record that the imputer, the augmentation, the selection step and the models never saw.

---

## 1. Summary in one page

**What stays the same**

- The same literature data source (arc-discharge syntheses of AgNPs), the same 10 inputs, target = particle size (nm).
- The same framework as in the title: **imputation → augmentation (noise or Gaussian copula) → stacked ensemble of Random Forest, LightGBM and DNN.**
- The same type of tables (Tables 5 and 6: RF, LightGBM, DNN and ensemble; MAE, RMSE, R²) and the same figure types.

**What is new compared with package v2**

| | v2 (sent before) | v3 (this package) |
|---|---|---|
| Records | 33 | 30 (2 unusable records removed, 1 duplicate merged) |
| Cross-validation | 5-fold × 10 repeats | 10-fold × 10 repeats; nested 10-fold × 3 for the main result |
| Inputs | 10 reported inputs | 10 inputs + 3 physics features (power, electric field, current density) |
| Framework settings | fixed by us | chosen by the data inside each training fold (nested CV; 72 candidates) |
| **Main result, stacked ensemble** | R² = 0.55 ± 0.13, MAE = 13.6 nm | **R² = 0.61 ± 0.08, MAE = 12.4 nm** (nested CV, fully unbiased) |
| Framework as titled (autoencoder + copula + stacking) | R² = 0.55 | **R² = 0.67 ± 0.08, MAE = 11.9 nm** |
| Best configuration (mean imputation, no augmentation) | R² = 0.66 | **R² = 0.73 ± 0.04, MAE = 10.5 nm** |

**How to read the three numbers**

- **0.61 (nested CV)** is the strictest number: the configuration (imputer, augmentation, features, target scale) is chosen using the training fold only, then tested once on the held-out fold. This removes the "we tried many settings and reported the best" bias. **This should be the headline result.**
- **0.67** is the framework exactly as named in the title, with physics features added, at fixed settings. This replaces Table 5 (copula). The noise version (0.63) replaces Table 6.
- **0.73** is the single configuration that nested CV picked most often. It is reported as the best configuration, and is slightly optimistic because it was chosen after looking at the data (Section 6).

**What the doctor needs to know** (details in Section 6)

1. The log-transformed target we planned **made results worse** (0.64 → 0.57) and was not used. Missing-value indicators had no effect. Physics features help a little in the best configuration (0.70 → 0.73).
2. On real records, augmentation and the autoencoder **still do not increase accuracy** (no augmentation 0.73 vs copula 0.71 vs noise 0.63; the autoencoder 0.72 vs mean imputation 0.73). They are kept as in the title and reported honestly.
3. Random Forest alone is as good as the stacked ensemble (0.76 vs 0.73), and the learned stacking weights confirm this (RF ≈ 0.7). The DNN is weak on 30 records (R² ≈ 0.3).
4. The dataset change (Refs 15 and 17 removed, Ref 18 merged with Ref 1) must be stated in the Methods and agreed by the doctors.

---

## 2. Why the original 0.99 cannot be used (unchanged from v2)

In the original code, 1,000 augmented rows were generated from **all** records and **then** split at random into train and test. Each test row was a slightly noisy copy of a training row (nearest-neighbour distance 0.14 SD units vs 4.1 for random pairs; 100% of test rows had a near-copy in training). Reproducing that protocol here gives the same R² ≈ 0.99 for noise augmentation (Fig. 4, sheet *Original_protocol_synthetic*). In this package the order is corrected: **split first, then impute, augment, select and train on the training part only; test on real records only** (Fig. 1).

---

## 3. Every change from the submitted paper to v3, with reasons and reviewer mapping

| # | Change | Why | Reviewer |
|--|--------------------|--------------|-----|
| 1 | Split into folds **before** imputation and augmentation; test on real records only | Removes the data leakage | R3.1, R4.3, R4.4 |
| 2 | Repeated 10-fold CV (10 repeats), results as mean ± SD | Stable estimates with SDs; 10-fold trains on 27 of 30 records instead of 24 | R3.3 |
| 3 | **Nested CV** for the main result: inside each training fold, 72 framework configurations are compared by inner 3-fold CV and the best is applied to the held-out fold | Choosing settings on the same data used to report results would bias the result upward | R3.1, R3.3, R4.4 |
| 4 | Dataset: Refs 15 and 17 removed, Ref 18 merged with Ref 1 (30 records, 14 studies), decided before any model was run | Refs 15/17 report no current or voltage (all process inputs would be invented); Ref 18 is the same experiment as Ref 1 | R4.3, R3.6 |
| 5 | Imputer fitted inside each training fold, target never used; autoencoder, mean and KNN compared; masking test added | Validates the imputation and shows how well hidden values are recovered | R3.5, R4.6 |
| 6 | Augmentation generated from the training fold only (1 synthetic row per real row); none / noise / copula compared | Synthetic rows never reach the test set; a non-augmented baseline is reported | R3.2, R4.4 |
| 7 | Stacked ensemble with **learned** weights (non-negative, summing to 1) from inner out-of-fold predictions; weights reported | The original "ensemble" was a plain average | R3.4 |
| 8 | Physics-based features: log(V·I) power, log(V/gap) electric field, log(I/electrode area) current density | Arc physics: particle size depends on energy input and current density; helps the models on few records | R3.6, R4.2 |
| 9 | Missing-value indicators (0/1 columns) evaluated | Tells the model which values were imputed; no effect, kept as an option only | R3.5, R4.6 |
| 10 | Log-transformed target evaluated | Sizes are skewed (5–92 nm); it **reduced** accuracy and was not selected | R3.3 |
| 11 | DNN: same architecture (128-64-32), target standardized | Without it the DNN is unstable on few records | R4.2 |
| 12 | LightGBM: small-data leaf settings (min_child_samples = 3) | The default (20) cannot split folds of ~27 records | R3.7 |
| 13 | Base-model predictions limited to the size range seen in training | Sizes cannot be negative or far outside the observed range | — |
| 14 | Complete configuration table for all components | Inconsistent hyper-parameters were flagged | R3.7 |
| 15 | Learning curve over the number of **real records** | The former plots (trees/epochs) were read as sample-size curves | R3.8 |
| 16 | Ablation table and "development steps" table (each change in turn) | Shows the contribution of every component; non-augmented baseline requested | R3.2 |
| 17 | Leave-one-study-out, y-randomization, permutation importance | Generalization to a new laboratory, proof that the model learns real structure, interpretability | R3.6, R4.6, R4.2 |
| 18 | Corrected flowchart (Fig. 1) and figure numbering | The flowchart contradicted the text | R3.1, R3.10, R4.5 |

---

## 4. Results to put in the paper

All values are on **real held-out records**, as mean ± SD over repeats. Formatted versions are in `results/Paper_Tables.xlsx`, one sheet per table (sheet name in brackets).

### 4.1 Main result – nested cross-validation [Main_Nested_CV_result]

Outer 10-fold CV × 3 repeats; inside each outer training fold the configuration is chosen from 72 candidates by inner 3-fold CV.

| Model | MAE (nm) | RMSE (nm) | R² |
|---|---|---|---|
| Random Forest | 11.57 ± 0.38 | 18.80 ± 1.18 | 0.658 ± 0.043 |
| LightGBM | 11.17 ± 0.49 | 19.41 ± 1.02 | 0.637 ± 0.039 |
| DNN | 18.71 ± 1.95 | 28.73 ± 0.54 | 0.206 ± 0.030 |
| Average ensemble | 12.58 ± 0.33 | 20.48 ± 1.02 | 0.596 ± 0.041 |
| **Stacked ensemble (proposed)** | **12.43 ± 0.98** | **20.14 ± 2.07** | **0.606 ± 0.083** |

Which configurations were selected [Nested_selected_configs]: physics features in 25 of 30 folds; mean imputation in 19, autoencoder in 6, KNN in 5; no augmentation in 12, copula in 9, noise in 9; log target in 13; missing indicators in 10. The most frequent single configuration (mean imputation, no augmentation, physics features, no log target, no indicators) was chosen in 4 of 30 folds; many configurations perform almost equally well.

*Suggested text:* "To avoid optimistic bias from configuration selection, the complete framework was evaluated by nested cross-validation: within each outer training fold, the imputer, augmentation strategy, feature set and target scale were selected by inner cross-validation, and the selected framework was tested once on the outer fold. The stacked ensemble achieved R² = 0.61 ± 0.08 and MAE = 12.4 ± 1.0 nm on real held-out records."

### 4.2 Replacement for Tables 5 and 6 – framework as titled [Titled_Copula, Titled_Noise]

Autoencoder imputation + augmentation + physics features + stacking; 10-fold CV × 10 repeats.

**Table 5 – Gaussian copula**

| Model | MAE (nm) | RMSE (nm) | R² |
|---|---|---|---|
| Random Forest | 11.67 ± 1.31 | 17.88 ± 1.72 | 0.690 ± 0.061 |
| LightGBM | 11.31 ± 1.93 | 17.80 ± 2.44 | 0.689 ± 0.087 |
| DNN | 17.83 ± 3.94 | 25.84 ± 4.14 | 0.341 ± 0.201 |
| Average ensemble | 12.28 ± 1.59 | 18.50 ± 1.46 | 0.669 ± 0.052 |
| **Stacked ensemble (proposed)** | **11.88 ± 1.69** | **18.34 ± 2.21** | **0.672 ± 0.080** |

**Table 6 – Noise augmentation**

| Model | MAE (nm) | RMSE (nm) | R² |
|---|---|---|---|
| Random Forest | 11.80 ± 1.26 | 19.31 ± 2.15 | 0.637 ± 0.077 |
| LightGBM | 10.78 ± 0.90 | 18.45 ± 1.73 | 0.670 ± 0.061 |
| DNN | 15.06 ± 2.75 | 24.56 ± 3.88 | 0.405 ± 0.189 |
| Average ensemble | 11.78 ± 1.05 | 19.33 ± 1.67 | 0.638 ± 0.062 |
| **Stacked ensemble (proposed)** | **11.87 ± 1.23** | **19.52 ± 2.39** | **0.628 ± 0.085** |

### 4.3 Best configuration and augmentation comparison [Table5_Copula, Table6_Noise, Table_No_augmentation]

Mean imputation + physics features + stacking; 10-fold CV × 10 repeats (Fig. 3).

| Stacked ensemble | MAE (nm) | RMSE (nm) | R² |
|---|---|---|---|
| No augmentation (best configuration) | 10.49 ± 0.70 | 16.73 ± 1.11 | 0.729 ± 0.037 |
| Gaussian copula | 10.88 ± 1.06 | 17.22 ± 1.52 | 0.712 ± 0.052 |
| Noise augmentation | 11.73 ± 0.66 | 19.56 ± 1.32 | 0.630 ± 0.048 |

The full per-model versions of these three tables are in the workbook.

### 4.4 Development steps – the effect of each change [Development_steps] (Fig. 5)

| Step | R² (stacked) |
|---|---|
| Package v2: 33 records, 5-fold, autoencoder + noise + stacking | 0.553 ± 0.125 |
| 30 records, 5-fold | 0.660 ± 0.089 |
| 30 records, 10-fold (baseline for the rows below) | 0.639 ± 0.084 |
| + physics features | 0.628 ± 0.085 |
| + missing-value indicators | 0.642 ± 0.078 |
| + log-transformed target | 0.570 ± 0.101 |
| no augmentation | 0.722 ± 0.049 |
| Gaussian copula instead of noise | 0.661 ± 0.066 |
| mean imputation instead of autoencoder | 0.653 ± 0.047 |
| KNN imputation instead of autoencoder | 0.619 ± 0.077 |
| Best configuration (most selected in nested CV) | 0.729 ± 0.037 |
| **Nested CV (main result)** | **0.606 ± 0.083** |

Each "+" row changes **one** setting relative to the 10-fold baseline. This table is useful for the response letter: it shows exactly what each suggestion did, including the ones that did not help.

### 4.5 Ablation around the best configuration [Ablation] (Fig. 6) – answers R3.2

| Variant | R² |
|---|---|
| Final framework (stacked ensemble) | 0.729 ± 0.037 |
| Simple average instead of stacking | 0.699 ± 0.044 |
| Single model: Random Forest | 0.761 ± 0.037 |
| Single model: LightGBM | 0.695 ± 0.039 |
| Single model: DNN | 0.326 ± 0.214 |
| Without physics features | 0.699 ± 0.040 |
| Autoencoder imputation | 0.715 ± 0.048 |
| KNN imputation | 0.707 ± 0.040 |
| Log-transformed target | 0.616 ± 0.040 |
| With missing indicators | 0.728 ± 0.051 |

### 4.6 Learned ensemble weights [Ensemble_weights] – answers R3.4

| Base model | Weight, best configuration | Weight, nested CV |
|---|---|---|
| Random Forest | 0.72 ± 0.34 | 0.62 ± 0.35 |
| LightGBM | 0.23 ± 0.33 | 0.24 ± 0.32 |
| DNN | 0.05 ± 0.09 | 0.14 ± 0.16 |

### 4.7 Original protocol vs corrected protocol [Original_protocol_synthetic] (Fig. 4) – answers R3.1

The authors' original augmented sheets with a random 80/20 split of synthetic rows: noise augmentation R² = 0.993 (ensemble), Gaussian copula 0.681. These are reported only as a clearly labelled supplementary comparison.

### 4.8 Robustness

- **Leave-one-study-out** [Leave_one_study_out]: every study held out in turn, as a new laboratory. Stacked ensemble R² = 0.60, MAE = 16.5 nm; Random Forest R² = 0.68. Answers R3.6 / R4.6.
- **y-randomization** [y_randomization] (Fig. 9): 30 runs with shuffled sizes give mean R² = −0.24 (max 0.08) vs 0.73 with the real sizes; p = 0.032 (the smallest possible p with 30 permutations). The model learns real structure.
- **Learning curve** [Learning_curve] (Fig. 8): R² ≈ 0.64–0.71 from 10 to 25 records, 0.76 with 30 records; the small subsets have large SDs. More records are expected to help.
- **Permutation importance** [Feature_importance] (Fig. 10): voltage (MAE +11.2 nm when shuffled) and current (+9.8 nm) dominate; electrode gap is third; yield, zeta potential, beaker volume and temperature contribute nothing measurable. This also answers the concern about post-synthesis inputs (yield, zeta): the model does not rely on them.
- **Masking test** [Masking_test] (Fig. S1): with 15% of known values hidden, every simple imputer recovers them about as well as the autoencoder (RMSE 0.90–0.98 SD units; 1.0 = guessing the mean). MICE is unstable (2.2). Answers R3.5.

---

## 5. Which file replaces which item in the manuscript

| Manuscript item | Replace with |
|---|---|
| Fig. 1 flowchart | `figures/fig01_workflow.png` |
| Missing-data description | `figures/fig02_missing_values.png`, `data/missing_values_v3.csv` |
| Table 5 (copula) | sheet `Titled_Copula` (Section 4.2) |
| Table 6 (noise) | sheet `Titled_Noise` (Section 4.2) |
| New main-result table / abstract numbers | sheet `Main_Nested_CV_result` (Section 4.1); parity plot `fig07_parity.png` |
| Figures comparing models (bar charts) | `fig03_performance_by_augmentation.png` |
| New: development steps | sheet `Development_steps`, `fig05_development_steps.png` |
| New: ablation | sheet `Ablation`, `fig06_ablation.png` |
| New: ensemble weights | sheet `Ensemble_weights` |
| Learning-curve figures | `fig08_learning_curve.png` |
| Hyper-parameter table | sheet `Configuration` |
| Supplementary | `fig04_synthetic_vs_real_test.png`, `fig09_y_randomization.png`, `fig10_feature_importance.png`, `figS1_masking_test.png`; sheets `Original_protocol_synthetic`, `Leave_one_study_out`, `y_randomization`, `Masking_test`, `Data_record_changes`, `Data_parsing_notes` |

**Numbers for the abstract (suggested):** "…evaluated by nested cross-validation on 30 literature records, the stacked ensemble predicted particle size of unseen syntheses with R² = 0.61 ± 0.08 and MAE = 12.4 nm; voltage and current were the dominant predictors."

---

## 6. Points the doctor must be aware of

1. **Dataset change (needs the doctors' agreement).** Ref 15 (no current or voltage; the size cell became an Excel date) and Ref 17 (vacuum process; no current, voltage, pH, temperature or electrode data) are removed. Ref 18 has identical conditions and size to Ref 1, so it is merged. The decision was made before running any model, for data-quality reasons. If the doctors prefer to keep 33 records, package v2 applies.
2. **Which number is the "real" one.** 0.61 (nested CV) is the fully unbiased estimate. 0.73 was chosen after seeing results, so it is somewhat optimistic; the nested CV shows that no single configuration is clearly best (the top one was selected in only 4 of 30 folds). Present 0.73 as the best configuration, not as the main result.
3. **Suggestions that did not help are reported, not hidden:** the log target (worse), missing indicators (no change), augmentation (no gain on real records), autoencoder vs mean imputation (no gain). Reviewers asked for exactly these comparisons; hiding them would be a new problem.
4. **Expected reviewer question:** "Why use augmentation, an autoencoder and a stacked ensemble if simpler options are as good?" Suggested answer: the framework is designed for growing literature datasets with heavy missingness; on the current 30 records the components are neutral or slightly negative, which the ablation documents, and the copula version is within one SD of the best configuration (0.71 vs 0.73).
5. **Ref 8 dominance:** 10 of the 30 records come from one study (Ref 8). The leave-one-study-out analysis (R² = 0.60) shows the result does not depend on it.
6. **Data assumptions to state in the Methods:** ranges ("20–30 nm", "0.8 to 1.6 A", "32.5 and 63 °C") use the lower bound, as in the original submission; current is in amperes; zeta potential is used as an absolute value.
7. **The DNN** uses scikit-learn's MLPRegressor with the original architecture (128-64-32). It is weak on 30 records; the stacking assigns it a small weight automatically.
8. **Still needed from the chemistry side:** TEM statistics (number of particles measured, size histogram, mean ± SD), requested by Reviewer 3, point 9.

---

## 7. Folder and file guide

```
AgNP_revision_v3_improved/
├── README.md                      repository overview
├── docs/REVISION_REPORT.md/.docx  this guide (same content)
├── run_all.sh                     runs all steps in order
├── requirements.txt               Python packages
├── data/
│   ├── arc_data_raw.xlsx                  raw literature table as received (33 records)
│   ├── synthetic_datasets_original.xlsx   the authors' original augmented sheets (comparison only)
│   ├── arc_data_paper33.csv               33-record table (paper conventions), input of step 1b
│   ├── arc_data_v3_30.csv                 30-record modelling table used in v3
│   ├── v3_record_changes.csv              which records were removed/merged and why
│   ├── parsing_notes.csv                  every non-trivial conversion (ranges, Excel date)
│   ├── missing_values_v3.csv              missing counts (Fig. 2)
│   ├── missing_values_per_column.csv      missing counts for the 33-record table
│   └── v2_reference_result.csv            the v2 headline result (first row of the development table)
├── code/
│   ├── 01_prepare_data.py           raw Excel -> arc_data_paper33.csv
│   ├── 01b_clean_v3.py              33 -> 30 records (Refs 15, 17 removed; Ref 18 merged)
│   ├── framework_v3.py              the framework: imputers, augmentation, physics features, models, stacking, CV, nested selection
│   ├── agnp_pipeline.py             autoencoder imputer used by framework_v3.py
│   ├── 02a_nested.py                main result: nested CV (~80 min on 2 cores)
│   ├── 02b_run_experiments.py       all other experiments (~25 min)
│   ├── 02c_framework_as_titled.py   Tables 5/6 for the framework as titled (~2 min)
│   ├── 03_make_figures.py           all figures (300 dpi)
│   └── 04_format_tables.py          all tables -> results/Paper_Tables.xlsx
├── results/
│   ├── Paper_Tables.xlsx            ALL TABLES, formatted (one sheet per table)
│   ├── N_*.csv                      nested CV: performance, selected configurations, weights, predictions
│   ├── T_framework_as_titled.csv    Tables 5/6, framework as titled
│   ├── T5_T6_by_augmentation.csv    best configuration with copula / noise / none
│   ├── D_development.csv            development steps
│   ├── A_ablation.csv               ablation;  F_oof_predictions_final.csv  predictions
│   ├── W_ensemble_weights*.csv      learned stacking weights
│   ├── L_*.csv                      leave-one-study-out
│   ├── Y_y_randomization.csv, C_learning_curve.csv, I_feature_importance.csv, M_masking_test.csv
│   ├── O_original_protocol.csv      original protocol on the authors' sheets
│   ├── H_configuration.csv          configuration of every component
│   └── *_log.txt                    console output of each run
└── figures/
    ├── fig01_workflow.png                    corrected flowchart (with nested selection)
    ├── fig02_missing_values.png              missing values per input
    ├── fig03_performance_by_augmentation.png copula / noise / none, all models
    ├── fig04_synthetic_vs_real_test.png      original vs corrected protocol
    ├── fig05_development_steps.png           effect of each change
    ├── fig06_ablation.png                    ablation
    ├── fig07_parity.png                      measured vs predicted (nested CV)
    ├── fig08_learning_curve.png              learning curve over real records
    ├── fig09_y_randomization.png             y-randomization
    ├── fig10_feature_importance.png          permutation importance
    └── figS1_masking_test.png                imputer masking test
```

---

## 8. How to reproduce everything

```bash
pip install -r requirements.txt
bash run_all.sh
```

- Total running time is about 2 hours on a 2-core machine (the nested CV takes most of it). Random seeds are fixed, so the numbers are reproducible; for example, the autoencoder + noise + physics result (0.628) came out identical in two independent scripts.
- Finished stages are skipped: to recompute one table, delete its CSV in `results/` and rerun the script.
- Environment variables: `REPEATS` (CV repeats, default 10), `NPERM` (y-randomization permutations, default 30).

---

## 9. Glossary

- **Data leakage:** information from the test data reaching the training process, which makes results look better than they are.
- **10-fold cross-validation:** split the records into 10 parts, train on 9, test on the 10th, and rotate; repeating with different splits gives a mean and an SD.
- **Nested cross-validation:** a second, inner cross-validation inside each training part chooses the settings, so the outer test part is never used for any decision.
- **Leave-one-study-out:** all records from one published study are held out together, simulating a new laboratory.
- **Physics features:** inputs computed from the reported ones: power (V × I), electric field (V / gap), current density (I / electrode area).
- **Missing-value indicator:** a 0/1 column telling the model that a value was imputed.
- **R²:** 1 = perfect, 0 = no better than predicting the average size. **MAE / RMSE:** average error in nm.
- **Stacked ensemble:** a learned weighted average of the base models.
- **Ablation:** removing or changing one component at a time to measure its contribution.
- **Masking test:** hiding known values and checking how well an imputer recovers them.
- **y-randomization:** retraining with shuffled sizes; a real model should fail when the link between inputs and sizes is destroyed.
