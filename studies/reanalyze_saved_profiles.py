from __future__ import annotations

from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_utils import build_reconstruction_quality_summary, perform_saxs_analysis


REL_TOL = 0.2
SUMMARY_NAME = "benchmark_summary.csv"
AGG_NAME = "aggregate_by_condition.csv"
BEST_WORST_NAME = "best_worst_cases.csv"


def build_sanity_from_theory(row: pd.Series) -> dict:
    comparisons = {
        "Rg": ("Rg_extracted", "Rg_theory"),
        "G": ("G_extracted", "G_theory"),
        "B": ("B_extracted", "B_theory"),
        "Q": ("Q_extracted", "Q_theory"),
        "lc": ("lc_extracted", "lc_theory"),
        "PDI": ("PDI_extracted", "PDI_theory"),
        "PDI2": ("PDI2_extracted", "PDI2_theory"),
        "mean_radius_pdi": ("mean_radius_rec_pdi", "mean_radius_theory"),
        "mean_radius_pdi2": ("mean_radius_rec_pdi2", "mean_radius_theory"),
        "p_rec_pdi": ("p_rec_pdi", "p_input"),
        "p_rec_pdi2": ("p_rec_pdi2", "p_input"),
    }
    rel_errs = {}
    failures = []
    for label, (obs_key, exp_key) in comparisons.items():
        obs = float(row.get(obs_key, 0.0))
        exp = float(row.get(exp_key, 0.0))
        if exp != 0:
            rel_err = (obs - exp) / exp
        else:
            rel_err = 0.0 if obs == 0 else np.inf
        rel_errs[label] = rel_err
        if not np.isfinite(rel_err) or abs(rel_err) > REL_TOL:
            failures.append(label)

    suggestions = []
    if any(key in failures for key in ("Rg", "G")):
        suggestions.append("Low-q extraction looks unstable: inspect the Guinier window, reduce smearing, and keep enough low-q points below qRg about 1.")
    if "B" in failures:
        suggestions.append("High-q Porod extraction looks unstable: extend q_max, reduce smearing, and check whether the fitted unified model is capturing the Porod tail.")
    if any(key in failures for key in ("Q", "lc", "PDI2", "p_rec_pdi2")):
        suggestions.append("Invariant-based extraction is drifting: prefer the unified-fit analytic Q/lc path, extend the measured q-range, and inspect any residual tail-correction dependence.")
    if any(key in failures for key in ("PDI", "p_rec_pdi")):
        suggestions.append("PDI-based recovery is drifting: inspect the consistency of G, Rg, and B extraction against the simulated curve and compare unified-fit versus hybrid estimates.")
    if any(key in failures for key in ("mean_radius_pdi", "mean_radius_pdi2")):
        suggestions.append("Recovered mean size is drifting: inspect moment-to-size conversion and the chosen distribution family.")
    if not suggestions:
        suggestions.append("All current sanity checks passed within tolerance.")

    out = {
        "sanity_pass": len(failures) == 0,
        "sanity_failures": ",".join(failures) if failures else "none",
        "sanity_suggestions": " | ".join(suggestions),
    }
    for label, rel_err in rel_errs.items():
        out[f"Sanity_RelErr_{label}"] = rel_err
    return out


def reanalyze_row(row: pd.Series) -> dict:
    profile = pd.read_csv(row["profile_csv_path"], usecols=["q", "I_normalized"])
    q_vals = profile["q"].to_numpy(dtype=float)
    i_vals = profile["I_normalized"].to_numpy(dtype=float)
    analysis = perform_saxs_analysis(
        q_vals,
        i_vals,
        row["distribution"],
        float(row["mean_rg_input"]),
        "Sphere",
        "Tomchuk",
        float(row["mean_rg_input"]) * (1.0 + 8.0 * float(row["p_input"])),
    )
    recon = build_reconstruction_quality_summary(analysis)

    updated = {
        "tomchuk_extraction": analysis.get("tomchuk_extraction", "none"),
        "Rg_extracted": float(analysis.get("Rg", 0.0)),
        "G_extracted": float(analysis.get("G", 0.0)),
        "B_extracted": float(analysis.get("B", 0.0)),
        "Q_extracted": float(analysis.get("Q", 0.0)),
        "lc_extracted": float(analysis.get("lc", 0.0)),
        "PDI_extracted": float(analysis.get("PDI", 0.0)),
        "PDI2_extracted": float(analysis.get("PDI2", 0.0)),
        "Rg_guinier": float(analysis.get("Rg_guinier", 0.0)),
        "G_guinier": float(analysis.get("G_guinier", 0.0)),
        "p_rec_pdi": float(analysis.get("p_rec_pdi", 0.0)),
        "p_rec_pdi2": float(analysis.get("p_rec_pdi2", 0.0)),
        "mean_radius_rec_pdi": float(analysis.get("mean_r_rec_pdi", 0.0)),
        "mean_radius_rec_pdi2": float(analysis.get("mean_r_rec_pdi2", 0.0)),
        "rrms_pdi": float(analysis.get("rrms_pdi", 0.0)),
        "rrms_pdi2": float(analysis.get("rrms_pdi2", 0.0)),
        "rrms_primary": float(analysis.get("rrms", 0.0)),
        "rrms_quality_pdi": recon["quality_pdi"],
        "rrms_quality_pdi2": recon["quality_pdi2"],
        "best_reconstruction_variant": recon["best_variant"],
        "best_reconstruction_rrms": recon["best_rrms"],
        "rel_err_Rg": 0.0,
        "rel_err_G": 0.0,
        "rel_err_B": 0.0,
        "rel_err_Q": 0.0,
        "rel_err_lc": 0.0,
        "rel_err_PDI": 0.0,
        "rel_err_PDI2": 0.0,
        "abs_err_p_pdi": abs(float(analysis.get("p_rec_pdi", 0.0)) - float(row["p_input"])),
        "abs_err_p_pdi2": abs(float(analysis.get("p_rec_pdi2", 0.0)) - float(row["p_input"])),
    }

    for label, obs_key, theory_key in (
        ("Rg", "Rg_extracted", "Rg_theory"),
        ("G", "G_extracted", "G_theory"),
        ("B", "B_extracted", "B_theory"),
        ("Q", "Q_extracted", "Q_theory"),
        ("lc", "lc_extracted", "lc_theory"),
        ("PDI", "PDI_extracted", "PDI_theory"),
        ("PDI2", "PDI2_extracted", "PDI2_theory"),
    ):
        theory = float(row[theory_key])
        obs = float(updated[obs_key])
        updated[f"rel_err_{label}"] = (obs - theory) / theory if theory != 0 else 0.0

    updated.update(build_sanity_from_theory(pd.Series({**row.to_dict(), **updated})))
    return updated


def rebuild_aggregate(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df.groupby(["distribution", "p_input", "flux_exp", "smearing"], as_index=False).agg(
        mean_abs_err_pdi=("abs_err_p_pdi", "mean"),
        std_abs_err_pdi=("abs_err_p_pdi", "std"),
        mean_abs_err_pdi2=("abs_err_p_pdi2", "mean"),
        std_abs_err_pdi2=("abs_err_p_pdi2", "std"),
        mean_rrms_pdi=("rrms_pdi", "mean"),
        mean_rrms_pdi2=("rrms_pdi2", "mean"),
        mean_rel_err_Rg=("rel_err_Rg", "mean"),
        mean_rel_err_G=("rel_err_G", "mean"),
        mean_rel_err_B=("rel_err_B", "mean"),
        mean_rel_err_Q=("rel_err_Q", "mean"),
        mean_rel_err_lc=("rel_err_lc", "mean"),
        mean_rel_err_PDI=("rel_err_PDI", "mean"),
        mean_rel_err_PDI2=("rel_err_PDI2", "mean"),
        sanity_pass_rate=("sanity_pass", "mean"),
    )


def rebuild_best_worst(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for distribution, group in summary_df.groupby("distribution"):
        score = group["abs_err_p_pdi2"] + group["abs_err_p_pdi"] + group["best_reconstruction_rrms"]
        best_idx = score.idxmin()
        worst_idx = score.idxmax()
        best_row = summary_df.loc[best_idx].copy()
        best_row["category"] = "best"
        worst_row = summary_df.loc[worst_idx].copy()
        worst_row["category"] = "worst"
        rows.extend([best_row, worst_row])
    return pd.DataFrame(rows)


def main() -> None:
    tables_dir = Path("studies/tomchuk_benchmark_20260329/tables")
    summary_path = tables_dir / SUMMARY_NAME
    aggregate_path = tables_dir / AGG_NAME
    best_worst_path = tables_dir / BEST_WORST_NAME

    summary_df = pd.read_csv(summary_path)
    original_summary = summary_df.copy()

    row_dicts = [row._asdict() for row in summary_df.itertuples(index=False)]
    max_workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        updates = list(executor.map(reanalyze_row, (pd.Series(row) for row in row_dicts), chunksize=50))
    updates_df = pd.DataFrame(updates)
    for col in updates_df.columns:
        summary_df[col] = updates_df[col]

    backup_path = tables_dir / "benchmark_summary_before_guinier_refresh.csv"
    if not backup_path.exists():
        original_summary.to_csv(backup_path, index=False)

    summary_df.to_csv(summary_path, index=False)
    rebuild_aggregate(summary_df).to_csv(aggregate_path, index=False)
    rebuild_best_worst(summary_df).to_csv(best_worst_path, index=False)

    changed = (summary_df["Rg_extracted"] - original_summary["Rg_extracted"]).abs() > 1e-12
    changed_guinier = (summary_df["Rg_guinier"] - original_summary.get("Rg_guinier", 0)).abs() > 1e-12 if "Rg_guinier" in summary_df.columns else pd.Series(dtype=bool)
    print(f"Reanalyzed {len(summary_df)} rows")
    print(f"Rows with changed selected Rg: {int(changed.sum())}")
    if "Rg_guinier" in summary_df.columns:
        print(f"Rows with nonzero Guinier Rg after refresh: {int((summary_df['Rg_guinier'] > 0).sum())}")


if __name__ == "__main__":
    main()
