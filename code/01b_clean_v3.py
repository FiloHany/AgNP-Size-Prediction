"""
STEP 1b - Data refinement for version 3 (decided BEFORE running any model):
  * remove Ref 15 and Ref 17: neither reports current or voltage (all their process
    inputs would be invented by the imputer); Ref 15's size is also unknown (Excel date)
  * merge Refs 1 and 18: identical conditions and size (the same experiment reported twice);
    Ref 18 is removed and Ref 1 kept
Result: 30 records from 14 independent studies.
Usage: python 01b_clean_v3.py
"""
import pandas as pd

d = pd.read_csv("../data/arc_data_paper33.csv")
log = [("Ref 15", "removed", "no current or voltage reported; size cell corrupted into a date"),
       ("Ref 17", "removed", "vacuum process (10^-3 Pa); no current, voltage, pH, temperature or electrode data"),
       ("Ref 18", "removed (merged with Ref 1)", "identical conditions and size to Ref 1 (duplicate record)")]
v3 = d[~d.Ref.isin([15, 17, 18])].reset_index(drop=True)
v3["group"] = range(len(v3))       # every record is now unique
v3["study"] = v3["Ref"]
v3.to_csv("../data/arc_data_v3_30.csv", index=False)
pd.DataFrame(log, columns=["record", "action", "reason"]).to_csv("../data/v3_record_changes.csv", index=False)
m = v3.drop(columns=["Ref", "group", "study"]).isna().sum()
m.to_csv("../data/missing_values_v3.csv", header=["missing"])
print(f"records: {len(v3)} | studies: {v3.study.nunique()}")
print("missing:", m[m > 0].to_dict())
