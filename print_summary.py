import pandas as pd
import json
from pathlib import Path

study_dir = sorted(Path("studies").glob("tenor_vs_tomchuk_full_*"))[-1]
df = pd.read_csv(study_dir / "summary_results.csv")

for exp in ["exp1_p_sweep", "exp2_flux", "exp3_smearing", "exp4_distribution", "exp5_anisotropy", "exp7_p_synchrotron", "exp8_p_home", "exp9_flux_synchrotron"]:
    s = df[df["experiment"] == exp]
    t = s["tenor_abs_err_p"].median()
    o = s["tomchuk_abs_err_p_pdi"].median()
    won = "TENOR" if t < o*0.9 else ("Tomchuk" if o < t*0.9 else "Tie")
    print(f"| {exp:21} | {t:7.3f} | {o:7.3f} | {won:7} |")
