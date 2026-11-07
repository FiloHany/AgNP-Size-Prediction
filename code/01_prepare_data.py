"""
STEP 1 - Build the modelling table from the raw literature data, following the
conventions of the original submission so that the dataset (33 records, 10 inputs)
stays the same as in the paper.

Conventions (same as the original manuscript):
  * ranges ("20nm-30nm", "0.8 to 1.6", "32.5 and 63 C") -> LOWER bound
  * Ref 15 size cell was converted by Excel into the date 2021-08-15; it is read as the
    range "8-15 nm" and, by the same convention, 8 nm is used  [to be verified by the authors]
  * zeta potential used as magnitude (positive), as in the original data
  * units: current A, voltage V, temperature C, beaker volume mL, gap mm, electrode dims mm
Records kept: all 33 (incl. the vacuum-process record, Ref 17, whose missing inputs are
imputed inside the cross-validation folds).
Refs 1 and 18 report the same experiment: both rows are kept but receive the SAME group id,
so they are always in the same CV fold (prevents pseudoreplication leakage).

Inputs (10, as in the paper): Current, Voltage, Yield, Zeta potential, pH, Temperature,
Volume of beaker, Electrode gap, Electrode dimension 1, Electrode dimension 2.
Target: Nano size (nm).

Usage: python 01_prepare_data.py ../data/arc_data_raw.xlsx ../data
"""
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "../data/arc_data_raw.xlsx"
out = sys.argv[2] if len(sys.argv) > 2 else "../data"
raw = pd.read_excel(src)
raw.columns = [c.strip() for c in raw.columns]


def nums(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, (int, float, np.number)):
        return [abs(float(v))]
    s = str(v).replace("−", "-").replace("–", "-")
    return [abs(float(x.replace(" ", ""))) for x in re.findall(r"-?\s?\d+(?:\.\d+)?", s)]


def first(v):
    n = nums(v)
    return n[0] if n else np.nan


def gap_mm(v):
    n = nums(v)
    if not n:
        return np.nan
    return n[0] / 1000 if "micro" in str(v).lower() else n[0]


def vol_ml(v):
    n = nums(v)
    if not n:
        return np.nan
    s = str(v).lower()
    return n[0] * 1000 if re.search(r"\d\s*l\b", s) and "ml" not in s else n[0]


def dims(v):
    n = nums(v)
    if not n:
        return np.nan, np.nan
    return n[0], (n[1] if len(n) > 1 else n[0])


rows, notes = [], []
for i, r in raw.iterrows():
    s = r["Nano size"]
    if isinstance(s, (datetime, pd.Timestamp)):
        size = 8.0
        notes.append((i + 2, r["Ref"], "Nano size", str(s), 8.0,
                      "Excel date; read as range 8-15 nm, lower bound used (verify with Ref 15)"))
    else:
        size = first(s)
        if len(nums(s)) > 1:
            notes.append((i + 2, r["Ref"], "Nano size", str(s), size, "range -> lower bound"))
    for col, key in [("Current", "Current"), ("Temprature", "Temperature")]:
        if len(nums(r[col])) > 1:
            notes.append((i + 2, r["Ref"], key, str(r[col]), first(r[col]), "range -> lower bound"))
    d1, d2 = dims(r["Electrode dimension"])
    rows.append({
        "Ref": int(r["Ref"]),
        "Nano size": size,
        "Current": first(r["Current"]),
        "Voltage": first(r["Volage"]),
        "Yield": first(r["Yield"]),
        "Zeta potential": first(r["Zeta potentail"]),
        "pH": first(r["PH"]),
        "Temperature": first(r["Temprature"]),
        "Volume of beaker": vol_ml(r["Volume of beaker"]),
        "Electrode gap": gap_mm(r["gap"]),
        "Electrode dimension 1": d1,
        "Electrode dimension 2": d2,
    })
df = pd.DataFrame(rows)

# group id: identical records (same inputs and size) share a group
key = [c for c in df.columns if c != "Ref"]
sig = df[key].apply(lambda r: "|".join(str(x) for x in r.values), axis=1)
df["group"] = df.groupby(sig, sort=False).ngroup()
dups = df[df.duplicated("group", keep=False)]
for g, d in dups.groupby("group"):
    notes.append((None, ",".join(map(str, d.Ref)), "all", "identical records", g,
                  "kept, same CV group (always in the same fold)"))
# study id for leave-one-study-out (duplicate studies merged)
df["study"] = df.groupby("group")["Ref"].transform("min")

df.to_csv(f"{out}/arc_data_paper33.csv", index=False)
pd.DataFrame(notes, columns=["excel_row", "Ref", "column", "raw_value", "used_value", "rule"]).to_csv(
    f"{out}/parsing_notes.csv", index=False)
miss = df.drop(columns=["Ref", "group", "study"]).isna().sum()
miss.to_csv(f"{out}/missing_values_per_column.csv", header=["missing"])
print(df.drop(columns=["group"]).to_string())
print("\nrecords:", len(df), "| CV groups:", df.group.nunique(), "| studies:", df.study.nunique())
print("missing:", miss[miss > 0].to_dict())
