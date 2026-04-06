import json
import numpy as np
from pathlib import Path

study_dir = sorted(Path("studies").glob("tenor_vs_tomchuk_fast_*"))[-1]
files = list((study_dir / "logs").glob("case_*.json"))
if not files:
    print("No cases finished yet.")
else:
    err_t = []
    err_o = []
    
    # Store by experiment
    data_by_exp = {}

    for f in files:
        data = json.loads(f.read_text())
        exp = data.get("experiment", "unknown")
        
        t_err = data.get("analysis", {}).get("tenor_abs_err_p", np.nan)
        o_err = data.get("analysis", {}).get("tomchuk_abs_err_p_pdi", np.nan)

        err_t.append(t_err)
        err_o.append(o_err)

        if exp not in data_by_exp:
            data_by_exp[exp] = {"t": [], "o": []}
        data_by_exp[exp]["t"].append(t_err)
        data_by_exp[exp]["o"].append(o_err)

    t_med = np.nanmedian(err_t)
    o_med = np.nanmedian(err_o)
    print(f"Finished {len(files)} cases.")
    print(f"Overall TENOR median err: {t_med:.4f}")
    print(f"Overall Tomchuk median err: {o_med:.4f}")
    print("-" * 40)
    for exp, vals in data_by_exp.items():
        t = np.nanmedian(vals["t"])
        o = np.nanmedian(vals["o"])
        print(f"[{exp}] TENOR: {t:.4f} | Tomchuk: {o:.4f}")
