import pandas as pd
from tenor_tomchuk_comparison_study import run_single_case

results = []
p_vals = [0.05, 0.3, 0.6]
qrg_limits = [1.0, 0.85, 0.8, 0.7, 0.6, 0.5]

print("Running qRg=limit evaluation...")
for p in p_vals:
    for qlimit in qrg_limits:
        case = dict(
            experiment="test_qrg",
            replicate=1,
            seed=42 + int(p*100) + int(qlimit*100),
            p_val=p,
            flux_exp=8,
            smearing_x=3.0,
            smearing_y=3.0,
            dist_type="Lognormal",
            case_id=len(results),
            # Overrides for TENOR
            tenor_qrg_limit=qlimit,
        )
        row = run_single_case(case)
        results.append({
            "p_val": p,
            "qrg_limit": qlimit,
            "tenor_abs_err_p": row.get("tenor_abs_err_p", float("nan")),
            "tenor_rel_err_rg": row.get("tenor_rel_err_rg", float("nan")),
            "tomchuk_abs_err_p": row.get("tomchuk_abs_err_p_pdi", float("nan")),
        })
        print(f"p={p:.2f}, qRg={qlimit:.2f} -> abs_err_p: {results[-1]['tenor_abs_err_p']:.4f}, rel_err_rg: {results[-1]['tenor_rel_err_rg']:.4f}")

df = pd.DataFrame(results)
print("\n=== SUMMARY of absolute error in p ===")
pivot_p = df.pivot(index="p_val", columns="qrg_limit", values="tenor_abs_err_p")
print(pivot_p.to_string())

print("\n=== SUMMARY of relative error in Rg ===")
pivot_rg = df.pivot(index="p_val", columns="qrg_limit", values="tenor_rel_err_rg")
print(pivot_rg.to_string())
