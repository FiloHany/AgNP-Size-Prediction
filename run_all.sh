#!/usr/bin/env bash
# Runs the full v3 analysis from the raw Excel file to the paper tables and figures (~2 hours on 2 cores).
set -e
cd "$(dirname "$0")/code"
python 01_prepare_data.py ../data/arc_data_raw.xlsx ../data
python 01b_clean_v3.py
python -u 02a_nested.py | tee ../results/nested_log.txt
python -u 02b_run_experiments.py | tee ../results/run_log.txt
python -u 02c_framework_as_titled.py | tee ../results/titled_log.txt
python 02d_synthetic_quality.py | tee ../results/synthetic_quality_log.txt
python -u 02e_original_framework.py | tee ../results/original_framework_log.txt
python -u 02f_nested_original.py | tee ../results/nested_original_log.txt
python -u 02g_learning_curve_cv.py | tee ../results/lc_cv_log.txt
python 03_make_figures.py
python 05_minimal_figures.py
python 04_format_tables.py > /dev/null
echo "Done: see ../results/Paper_Tables.xlsx and ../figures/"
