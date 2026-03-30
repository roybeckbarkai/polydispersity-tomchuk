import argparse
import csv
import math
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis_utils import (
    build_reconstruction_quality_summary,
    build_sanity_summary_row,
    calculate_sphere_input_theoretical_parameters,
    normalize_simulated_sphere_intensity,
    perform_saxs_analysis,
)
from sim_utils import run_simulation_core


DISTRIBUTIONS = ["Gaussian", "Lognormal", "Schulz", "Boltzmann", "Triangular", "Uniform"]
P_VALUES = [round(x, 1) for x in np.arange(0.1, 1.01, 0.1)]
FLUX_EXPS = [5, 6, 7, 8, 9]
SMEARINGS = list(range(1, 11))
REPLICATES = 5
MEAN_RG = 2.0
PIXELS = 1024
N_BINS = 1024
Q_MIN = 0.0
Q_MAX = 10.0 / MEAN_RG
BINNING_MODE = "Logarithmic"
MODE = "Sphere"
METHOD = "Tomchuk"


def ensure_dirs(base_dir):
    dirs = {
        "base": base_dir,
        "data": base_dir / "data",
        "tiff": base_dir / "data" / "detector_tiff",
        "csv": base_dir / "data" / "profiles_csv",
        "tables": base_dir / "tables",
        "figures": base_dir / "figures",
        "reports": base_dir / "reports",
        "logs": base_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_run_id(dist_type, p_val, flux_exp, smearing, replicate):
    return f"{dist_type}_p{p_val:.1f}_f{flux_exp}_s{smearing}_r{replicate}"


def save_tiff(path, array_2d):
    img = Image.fromarray(np.asarray(array_2d, dtype=np.float32))
    img.save(path, compression="tiff_lzw")


def run_one(task):
    out_tiff = Path(task["out_tiff"])
    out_csv = Path(task["out_csv"])

    params = {
        "mean_rg": task["mean_rg"],
        "p_val": task["p_val"],
        "dist_type": task["dist_type"],
        "mode": MODE,
        "pixels": PIXELS,
        "q_min": Q_MIN,
        "q_max": Q_MAX,
        "n_bins": N_BINS,
        "smearing": task["smearing"],
        "flux": 10 ** task["flux_exp"],
        "noise": True,
        "binning_mode": BINNING_MODE,
    }

    np.random.seed(task["seed"])
    q_sim, i_sim_raw, i_2d_final, r_vals, pdf_vals = run_simulation_core(params)
    i_sim_norm, normalization_scale = normalize_simulated_sphere_intensity(q_sim, i_sim_raw, r_vals, pdf_vals)
    analysis_res = perform_saxs_analysis(q_sim, i_sim_norm, task["dist_type"], task["mean_rg"], MODE, METHOD, task["mean_rg"] * (1 + 8 * task["p_val"]))
    analysis_res["simulation_normalization_scale"] = normalization_scale

    theory = calculate_sphere_input_theoretical_parameters(task["mean_rg"], task["p_val"], task["dist_type"])
    sanity = build_sanity_summary_row(q_sim, i_sim_norm, r_vals, pdf_vals, analysis_res)
    reconstruction = build_reconstruction_quality_summary(analysis_res)

    save_tiff(out_tiff, i_2d_final)
    one_d_df = pd.DataFrame({
        "q": q_sim,
        "I_raw": i_sim_raw,
        "I_normalized": i_sim_norm,
    })
    if len(analysis_res.get("I_fit_unified", [])) == len(q_sim):
        one_d_df["I_fit_unified"] = analysis_res["I_fit_unified"]
    if len(analysis_res.get("I_fit_pdi", [])) == len(q_sim):
        one_d_df["I_fit_pdi"] = analysis_res["I_fit_pdi"]
    if len(analysis_res.get("I_fit_pdi2", [])) == len(q_sim):
        one_d_df["I_fit_pdi2"] = analysis_res["I_fit_pdi2"]
    one_d_df.to_csv(out_csv, index=False)

    row = {
        "run_id": task["run_id"],
        "distribution": task["dist_type"],
        "mean_rg_input": task["mean_rg"],
        "p_input": task["p_val"],
        "flux_exp": task["flux_exp"],
        "flux": 10 ** task["flux_exp"],
        "smearing": task["smearing"],
        "replicate": task["replicate"],
        "seed": task["seed"],
        "pixels": PIXELS,
        "n_bins": N_BINS,
        "q_min": Q_MIN,
        "q_max": Q_MAX,
        "binning_mode": BINNING_MODE,
        "noise": True,
        "normalization_scale": normalization_scale,
        "tomchuk_extraction": analysis_res.get("tomchuk_extraction", "none"),
        "tiff_path": str(out_tiff),
        "profile_csv_path": str(out_csv),
        "Rg_extracted": analysis_res.get("Rg", 0.0),
        "G_extracted": analysis_res.get("G", 0.0),
        "B_extracted": analysis_res.get("B", 0.0),
        "Q_extracted": analysis_res.get("Q", 0.0),
        "lc_extracted": analysis_res.get("lc", 0.0),
        "PDI_extracted": analysis_res.get("PDI", 0.0),
        "PDI2_extracted": analysis_res.get("PDI2", 0.0),
        "p_rec_pdi": analysis_res.get("p_rec_pdi", 0.0),
        "p_rec_pdi2": analysis_res.get("p_rec_pdi2", 0.0),
        "mean_radius_rec_pdi": analysis_res.get("mean_r_rec_pdi", 0.0),
        "mean_radius_rec_pdi2": analysis_res.get("mean_r_rec_pdi2", 0.0),
        "rrms_pdi": analysis_res.get("rrms_pdi", 0.0),
        "rrms_pdi2": analysis_res.get("rrms_pdi2", 0.0),
        "rrms_primary": analysis_res.get("rrms", 0.0),
        "rrms_quality_pdi": reconstruction["quality_pdi"],
        "rrms_quality_pdi2": reconstruction["quality_pdi2"],
        "best_reconstruction_variant": reconstruction["best_variant"],
        "best_reconstruction_rrms": reconstruction["best_rrms"],
        "Rg_theory": theory.get("Rg", 0.0),
        "G_theory": theory.get("G", 0.0),
        "B_theory": theory.get("B", 0.0),
        "Q_theory": theory.get("Q", 0.0),
        "lc_theory": theory.get("lc", 0.0),
        "PDI_theory": theory.get("PDI", 0.0),
        "PDI2_theory": theory.get("PDI2", 0.0),
        "mean_radius_theory": theory.get("mean_radius", 0.0),
        "rel_err_Rg": ((analysis_res.get("Rg", 0.0) - theory.get("Rg", 0.0)) / theory.get("Rg", 1.0)) if theory.get("Rg", 0.0) else 0.0,
        "rel_err_G": ((analysis_res.get("G", 0.0) - theory.get("G", 0.0)) / theory.get("G", 1.0)) if theory.get("G", 0.0) else 0.0,
        "rel_err_B": ((analysis_res.get("B", 0.0) - theory.get("B", 0.0)) / theory.get("B", 1.0)) if theory.get("B", 0.0) else 0.0,
        "rel_err_Q": ((analysis_res.get("Q", 0.0) - theory.get("Q", 0.0)) / theory.get("Q", 1.0)) if theory.get("Q", 0.0) else 0.0,
        "rel_err_lc": ((analysis_res.get("lc", 0.0) - theory.get("lc", 0.0)) / theory.get("lc", 1.0)) if theory.get("lc", 0.0) else 0.0,
        "rel_err_PDI": ((analysis_res.get("PDI", 0.0) - theory.get("PDI", 0.0)) / theory.get("PDI", 1.0)) if theory.get("PDI", 0.0) else 0.0,
        "rel_err_PDI2": ((analysis_res.get("PDI2", 0.0) - theory.get("PDI2", 0.0)) / theory.get("PDI2", 1.0)) if theory.get("PDI2", 0.0) else 0.0,
        "abs_err_p_pdi": abs(analysis_res.get("p_rec_pdi", 0.0) - task["p_val"]),
        "abs_err_p_pdi2": abs(analysis_res.get("p_rec_pdi2", 0.0) - task["p_val"]),
        "sanity_pass": sanity.get("Sanity_Pass", False),
        "sanity_failures": sanity.get("Sanity_Failures", "none"),
        "sanity_suggestions": sanity.get("Sanity_Suggestions", ""),
    }
    for key, value in sanity.items():
        if key.startswith("Sanity_RelErr_"):
            row[key] = value
    return row


def append_row(csv_path, row, fieldnames):
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def aggregate_results(summary_csv, tables_dir):
    df = pd.read_csv(summary_csv)
    aggregate = (
        df.groupby(["distribution", "p_input", "flux_exp", "smearing"], as_index=False)
        .agg(
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
    )
    aggregate.to_csv(tables_dir / "aggregate_by_condition.csv", index=False)

    best_worst = []
    for dist_type, group in df.groupby("distribution"):
        best_idx = (group["abs_err_p_pdi"] + group["abs_err_p_pdi2"]).idxmin()
        worst_idx = (group["abs_err_p_pdi"] + group["abs_err_p_pdi2"]).idxmax()
        best_row = group.loc[best_idx].to_dict()
        worst_row = group.loc[worst_idx].to_dict()
        best_row["category"] = "best"
        worst_row["category"] = "worst"
        best_worst.extend([best_row, worst_row])
    pd.DataFrame(best_worst).to_csv(tables_dir / "best_worst_cases.csv", index=False)
    return df, aggregate


def plot_heatmaps(aggregate, figures_dir):
    fluxes = sorted(aggregate["flux_exp"].unique())
    smearings = sorted(aggregate["smearing"].unique())
    for metric, out_name, title_prefix in [
        ("mean_abs_err_pdi", "heatmaps_abs_err_pdi.png", "Mean |p_rec - p_input| from PDI"),
        ("mean_abs_err_pdi2", "heatmaps_abs_err_pdi2.png", "Mean |p_rec - p_input| from PDI2"),
    ]:
        fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
        axes = axes.ravel()
        vmax = float(aggregate[metric].max())
        for ax, dist_type in zip(axes, DISTRIBUTIONS):
            group = aggregate.loc[aggregate["distribution"] == dist_type]
            pivot = group.groupby(["smearing", "flux_exp"])[metric].mean().unstack()
            pivot = pivot.reindex(index=smearings, columns=fluxes)
            im = ax.imshow(pivot.values, aspect="auto", origin="lower", vmin=0, vmax=vmax, cmap="viridis")
            ax.set_title(dist_type)
            ax.set_xticks(range(len(fluxes)))
            ax.set_xticklabels(fluxes)
            ax.set_yticks(range(len(smearings)))
            ax.set_yticklabels(smearings)
            ax.set_xlabel("Forward-count exponent")
            ax.set_ylabel("Smearing")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.85, label=metric)
        fig.suptitle(title_prefix)
        fig.savefig(figures_dir / out_name, dpi=200)
        plt.close(fig)


def plot_p_curves(aggregate, figures_dir):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    axes = axes.ravel()
    for ax, dist_type in zip(axes, DISTRIBUTIONS):
        group = aggregate.loc[aggregate["distribution"] == dist_type].copy()
        # choose best flux/smearing pair for the distribution
        best_pair = (
            group.groupby(["flux_exp", "smearing"], as_index=False)[["mean_abs_err_pdi", "mean_abs_err_pdi2"]]
            .mean()
            .assign(score=lambda df: df["mean_abs_err_pdi"] + df["mean_abs_err_pdi2"])
            .sort_values("score")
            .iloc[0]
        )
        subset = group.loc[(group["flux_exp"] == best_pair["flux_exp"]) & (group["smearing"] == best_pair["smearing"])].sort_values("p_input")
        ax.plot(subset["p_input"], subset["mean_abs_err_pdi"], marker="o", label="PDI")
        ax.plot(subset["p_input"], subset["mean_abs_err_pdi2"], marker="s", label="PDI2")
        ax.set_title(f"{dist_type} (forward=1e{int(best_pair['flux_exp'])}, smear={int(best_pair['smearing'])})")
        ax.set_xlabel("Input p")
        ax.set_ylabel("Mean absolute p error")
        ax.legend()
    fig.savefig(figures_dir / "p_error_curves_best_conditions.png", dpi=200)
    plt.close(fig)


def plot_representative_fits(summary_df, figures_dir):
    ordered = summary_df.assign(score=summary_df["abs_err_p_pdi"] + summary_df["abs_err_p_pdi2"]).sort_values("score")
    best_cases = ordered.groupby("distribution", as_index=False).first()
    worst_cases = ordered.groupby("distribution", as_index=False).last()
    for label, cases in [("best", best_cases), ("worst", worst_cases)]:
        fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
        axes = axes.ravel()
        for ax, (_, row) in zip(axes, cases.iterrows()):
            profile = pd.read_csv(row["profile_csv_path"])
            ax.loglog(profile["q"], profile["I_normalized"], "k.", markersize=2, label="Simulated")
            if "I_fit_pdi" in profile:
                ax.loglog(profile["q"], profile["I_fit_pdi"], "-", linewidth=1, label="PDI fit")
            if "I_fit_pdi2" in profile:
                ax.loglog(profile["q"], profile["I_fit_pdi2"], "-", linewidth=1, label="PDI2 fit")
            ax.set_title(f"{row['distribution']} p={row['p_input']:.1f} forward=1e{int(row['flux_exp'])} smear={int(row['smearing'])}")
            ax.set_xlabel("q")
            ax.set_ylabel("I(q)")
            ax.legend(fontsize=7)
        fig.savefig(figures_dir / f"representative_{label}_fits.png", dpi=200)
        plt.close(fig)


def write_latex_report(base_dir, aggregate_df, summary_df):
    reports_dir = base_dir / "reports"
    figures_dir = base_dir / "figures"
    tables_dir = base_dir / "tables"

    dist_lines = []
    for dist_type in DISTRIBUTIONS:
        group = aggregate_df.loc[aggregate_df["distribution"] == dist_type]
        mean_pdi = group["mean_abs_err_pdi"].mean()
        mean_pdi2 = group["mean_abs_err_pdi2"].mean()
        best_row = group.assign(score=group["mean_abs_err_pdi"] + group["mean_abs_err_pdi2"]).sort_values("score").iloc[0]
        worst_row = group.assign(score=group["mean_abs_err_pdi"] + group["mean_abs_err_pdi2"]).sort_values("score").iloc[-1]
        better = "PDI2" if mean_pdi2 < mean_pdi else "PDI"
        dist_lines.append(
            f"\\paragraph{{{dist_type}}} "
            f"The mean absolute recovery error was {mean_pdi:.3f} for PDI and {mean_pdi2:.3f} for PDI2, so {better} was the better index on average. "
            f"The best averaged condition occurred at forward-count exponent {int(best_row['flux_exp'])} and smearing {int(best_row['smearing'])}, "
            f"while the worst averaged condition occurred at forward-count exponent {int(worst_row['flux_exp'])} and smearing {int(worst_row['smearing'])}."
        )

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{float}}
\usepackage{{hyperref}}
\title{{Appendix: Benchmark Study of Tomchuk Polydispersity Analysis in the Polydispersity App}}
\date{{{datetime.now().strftime("%Y-%m-%d")}}}
\begin{{document}}
\maketitle

\section*{{A. Study Design}}
This appendix reports a reproducible benchmark study of the app's Tomchuk analysis over six supported distribution families: Gaussian, Lognormal, Schulz, Boltzmann, Triangular, and Uniform.
The study used $R_g = 2$ nm, detector size $1024 \times 1024$, 1D bin count 1024, logarithmic binning, and $q_{{\max}} = {Q_MAX:.1f}$ nm$^{{-1}}$.
The input polydispersity range was $p = 0.1$ to $1.0$ in steps of 0.1. Forward-pixel photon-count exponents ranged from $10^5$ to $10^9$, smearing ranged from 1 to 10 pixels, and each condition was repeated five times with distinct random seeds.

\section*{{B. Workflow Used In This App}}
For each synthetic run, the app simulated a 2D detector pattern and corresponding 1D radial profile, normalized the 1D sphere data onto the input-theory amplitude scale, extracted Tomchuk quantities using the current unified-fit-first pipeline, reconstructed SAXS curves from the recovered PDI and PDI2 solutions, and evaluated reconstruction quality by relative RMS error.

\section*{{C. Deviations From The Original Tomchuk Paper}}
The present implementation is not a verbatim reproduction of the original paper. The main deviations are:
\begin{{itemize}}
\item The analysis is performed with a declared input distribution family rather than inferring the family from a PDI--PDI2 classification map.
\item The code uses a unified Beaucage-style fit to estimate $G$, $R_g$, and $B$ when possible, with fallback logic retained internally.
\item $Q$ and $l_c$ are obtained from analytic expressions tied to the fitted unified model; the historical code path also included explicit tail corrections to finite-$q$ integrals.
\item Simulated 1D data are normalized before analysis so amplitude-carrying extracted quantities can be compared directly to input-based theory.
\item Reconstruction quality is reported by relative RMS error instead of count-weighted $\chi^2$ in order to remain comparable across forward-count levels.
\end{{itemize}}

\section*{{D. Benchmark Matrix}}
The full run-by-run results are stored in \texttt{{tables/benchmark\_summary.csv}}, and the aggregated condition table is stored in \texttt{{tables/aggregate\_by\_condition.csv}}.
Every run also has a saved 2D detector TIFF and 1D CSV profile in the \texttt{{data/}} subdirectories.

\section*{{E. Representative Figures}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/heatmaps_abs_err_pdi.png}}
\caption{{Mean absolute recovery error in $p$ from PDI, averaged over replicates and input $p$.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/heatmaps_abs_err_pdi2.png}}
\caption{{Mean absolute recovery error in $p$ from PDI2, averaged over replicates and input $p$.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/p_error_curves_best_conditions.png}}
\caption{{Mean absolute recovery error versus input $p$ for the best averaged forward-count/smearing condition of each distribution family.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/representative_best_fits.png}}
\caption{{Representative best-fit reconstructed SAXS curves.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/representative_worst_fits.png}}
\caption{{Representative worst-case reconstructed SAXS curves.}}
\end{{figure}}

\section*{{F. Distribution-Specific Findings}}
{"".join(dist_lines)}

\section*{{G. Overall Findings}}
Across the full benchmark, performance depended strongly on both the distribution family and the input polydispersity. In general, increasing smearing degraded recovery, while increasing forward-pixel photon count helped most strongly when the run was still noise-limited.
However, once the forward-pixel photon count was high enough, the remaining error was dominated by model and inversion bias rather than by shot noise. PDI2 often outperformed PDI for broader distributions, while low-polydispersity cases remained challenging because the indices approach unity and become inversion-sensitive.

\section*{{H. Limitations}}
The main limitations identified in this benchmark are:
\begin{{itemize}}
\item The benchmark still assumes a declared distribution family rather than solving the unknown-family problem posed in the original Tomchuk framework.
\item The extraction remains sensitive to finite-$q$ coverage, detector discretization, and smearing, even after amplitude normalization.
\item Low-polydispersity cases are intrinsically hard because both PDI and PDI2 approach values close to 1, making the inverse mapping unstable.
\item Several families, especially Triangular and Uniform, can show strong bias or collapse in one of the two recovered indices.
\item The saved detector images are physically useful for inspection, but the simulation still omits effects such as background subtraction errors, detector masks, and calibration uncertainty.
\end{{itemize}}

\section*{{I. Reproducibility}}
The entire study was generated from a script in \texttt{{studies/run\_tomchuk\_benchmark.py}}. All outputs were written under the study directory, including a manifest of the parameter grid, TIFF detector images, 1D CSV files, summary tables, and the present \LaTeX{{}} appendix.

\end{{document}}
"""
    tex_path = reports_dir / "tomchuk_benchmark_appendix.tex"
    tex_path.write_text(tex)
    return tex_path


def compile_latex(tex_path):
    try:
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_path.name],
                cwd=tex_path.parent,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return True, None
    except Exception as exc:
        return False, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Run the Tomchuk benchmark study and generate an appendix report.")
    parser.add_argument("--output-dir", default=str(Path("studies") / f"tomchuk_benchmark_{datetime.now().strftime('%Y%m%d')}"))
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--compile-pdf", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    dirs = ensure_dirs(base_dir)
    manifest_path = dirs["tables"] / "manifest.csv"
    summary_csv = dirs["tables"] / "benchmark_summary.csv"

    tasks = []
    for dist_type in DISTRIBUTIONS:
        for p_val in P_VALUES:
            for flux_exp in FLUX_EXPS:
                for smearing in SMEARINGS:
                    for replicate in range(1, REPLICATES + 1):
                        run_id = build_run_id(dist_type, p_val, flux_exp, smearing, replicate)
                        out_tiff = dirs["tiff"] / f"{run_id}.tiff"
                        out_csv = dirs["csv"] / f"{run_id}.csv"
                        seed = (
                            DISTRIBUTIONS.index(dist_type) * 100000
                            + int(round(p_val * 10)) * 10000
                            + flux_exp * 1000
                            + smearing * 10
                            + replicate
                        )
                        tasks.append({
                            "run_id": run_id,
                            "dist_type": dist_type,
                            "mean_rg": MEAN_RG,
                            "p_val": p_val,
                            "flux_exp": flux_exp,
                            "smearing": smearing,
                            "replicate": replicate,
                            "seed": seed,
                            "out_tiff": str(out_tiff),
                            "out_csv": str(out_csv),
                        })

    manifest_df = pd.DataFrame(tasks)
    manifest_df.to_csv(manifest_path, index=False)

    completed = set()
    if summary_csv.exists():
        try:
            completed = set(pd.read_csv(summary_csv, usecols=["run_id"])["run_id"].tolist())
        except Exception:
            completed = set()
    pending_tasks = [task for task in tasks if task["run_id"] not in completed]
    print(f"Total tasks: {len(tasks)} | Completed: {len(completed)} | Pending: {len(pending_tasks)}")

    fieldnames = None
    if pending_tasks:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(run_one, task) for task in pending_tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                if fieldnames is None:
                    fieldnames = list(row.keys())
                append_row(summary_csv, row, fieldnames)
                if idx % 100 == 0 or idx == len(pending_tasks):
                    print(f"Completed {idx}/{len(pending_tasks)} pending runs")

    summary_df, aggregate_df = aggregate_results(summary_csv, dirs["tables"])
    plot_heatmaps(aggregate_df, dirs["figures"])
    plot_p_curves(aggregate_df, dirs["figures"])
    plot_representative_fits(summary_df, dirs["figures"])
    tex_path = write_latex_report(base_dir, aggregate_df, summary_df)

    compiled_ok = False
    compile_error = None
    if args.compile_pdf:
        compiled_ok, compile_error = compile_latex(tex_path)

    report_lines = [
        f"Study directory: {base_dir}",
        f"Summary CSV: {summary_csv}",
        f"Aggregate CSV: {dirs['tables'] / 'aggregate_by_condition.csv'}",
        f"LaTeX appendix: {tex_path}",
        f"PDF compiled: {compiled_ok}",
    ]
    if compile_error:
        report_lines.append(f"PDF compile error: {compile_error}")
    (dirs["logs"] / "run_summary.txt").write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
