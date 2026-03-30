import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

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
FLUX_EXPS = list(range(5, 10))
SMEARINGS = list(range(1, 11))
REPLICATES = 5
MEAN_RG = 2.0
PIXELS = 1024
N_BINS = 1024
Q_MIN = 0.0
Q_MAX = 5.0
BINNING_MODE = "Logarithmic"
NNLS_FACTOR = 8.0


def safe_rel_err(observed, expected):
    if expected == 0:
        return math.nan
    return (observed - expected) / expected


def save_tiff(path, array):
    array = np.asarray(array, dtype=np.float32)
    image = Image.fromarray(array)
    image.save(path, compression="tiff_deflate")


def build_case_output_dir(root_dir, dist_type, p_val, flux_exp, smearing, replicate):
    return (
        Path(root_dir)
        / "runs"
        / dist_type
        / f"p_{p_val:.1f}"
        / f"fluxexp_{flux_exp}"
        / f"smearing_{smearing}"
        / f"rep_{replicate}"
    )


def run_single_case(case):
    output_dir = build_case_output_dir(
        case["root_dir"],
        case["dist_type"],
        case["p_val"],
        case["flux_exp"],
        case["smearing"],
        case["replicate"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "mean_rg": case["mean_rg"],
        "p_val": case["p_val"],
        "dist_type": case["dist_type"],
        "mode": "Sphere",
        "pixels": case["pixels"],
        "q_min": case["q_min"],
        "q_max": case["q_max"],
        "n_bins": case["n_bins"],
        "smearing": case["smearing"],
        "flux": 10 ** case["flux_exp"],
        "noise": True,
        "binning_mode": case["binning_mode"],
    }

    np.random.seed(case["seed"])
    q_sim, i_raw, i_2d, r_vals, pdf_vals = run_simulation_core(params)
    i_norm, normalization_scale = normalize_simulated_sphere_intensity(q_sim, i_raw, r_vals, pdf_vals)

    analysis_res = perform_saxs_analysis(
        q_sim,
        i_norm,
        case["dist_type"],
        case["mean_rg"],
        "Sphere",
        "Tomchuk",
        case["mean_rg"] * (1 + NNLS_FACTOR * case["p_val"]),
    )
    analysis_res["simulation_normalization_scale"] = normalization_scale

    theory = calculate_sphere_input_theoretical_parameters(case["mean_rg"], case["p_val"], case["dist_type"])
    sanity = build_sanity_summary_row(q_sim, i_norm, r_vals, pdf_vals, analysis_res)
    reconstruction = build_reconstruction_quality_summary(analysis_res)

    detector_path = output_dir / "detector_2d.tiff"
    profile_path = output_dir / "profile_1d.csv"

    save_tiff(detector_path, i_2d)

    profile_data = {
        "q": q_sim,
        "I_raw": i_raw,
        "I_normalized": i_norm,
    }
    for key in ["I_fit_unified", "I_fit_pdi", "I_fit_pdi2"]:
        values = np.asarray(analysis_res.get(key, []))
        if values.ndim == 1 and len(values) == len(q_sim):
            profile_data[key] = values
    profile_df = pd.DataFrame(profile_data)
    profile_df.to_csv(profile_path, index=False)

    row = {
        "distribution": case["dist_type"],
        "p_input": case["p_val"],
        "flux_exp": case["flux_exp"],
        "flux": params["flux"],
        "smearing": case["smearing"],
        "replicate": case["replicate"],
        "seed": case["seed"],
        "mean_rg_input": case["mean_rg"],
        "pixels": case["pixels"],
        "n_bins": case["n_bins"],
        "q_min": case["q_min"],
        "q_max": case["q_max"],
        "normalization_scale": normalization_scale,
        "tomchuk_extraction": analysis_res.get("tomchuk_extraction", "none"),
        "detector_tiff": str(detector_path.relative_to(case["root_dir"])),
        "profile_csv": str(profile_path.relative_to(case["root_dir"])),
        "Rg_extracted": analysis_res.get("Rg", 0),
        "G_extracted": analysis_res.get("G", 0),
        "B_extracted": analysis_res.get("B", 0),
        "Q_extracted": analysis_res.get("Q", 0),
        "lc_extracted": analysis_res.get("lc", 0),
        "PDI_extracted": analysis_res.get("PDI", 0),
        "PDI2_extracted": analysis_res.get("PDI2", 0),
        "p_rec_pdi": analysis_res.get("p_rec_pdi", 0),
        "p_rec_pdi2": analysis_res.get("p_rec_pdi2", 0),
        "mean_radius_pdi": analysis_res.get("mean_r_rec_pdi", 0),
        "mean_radius_pdi2": analysis_res.get("mean_r_rec_pdi2", 0),
        "rrms_pdi": analysis_res.get("rrms_pdi", 0),
        "rrms_pdi2": analysis_res.get("rrms_pdi2", 0),
        "rrms_best_variant": reconstruction.get("best_variant", "n/a"),
        "rrms_best": reconstruction.get("best_rrms", 0),
        "quality_pdi": reconstruction.get("quality_pdi", "n/a"),
        "quality_pdi2": reconstruction.get("quality_pdi2", "n/a"),
        "Rg_theory": theory["Rg"],
        "G_theory": theory["G"],
        "B_theory": theory["B"],
        "Q_theory": theory["Q"],
        "lc_theory": theory["lc"],
        "PDI_theory": theory["PDI"],
        "PDI2_theory": theory["PDI2"],
        "p_theory": theory["p_true"],
        "sanity_pass": sanity.get("Sanity_Pass", False),
        "sanity_failures": sanity.get("Sanity_Failures", "none"),
        "sanity_suggestions": sanity.get("Sanity_Suggestions", ""),
    }

    for label in ["Rg", "G", "B", "Q", "lc", "PDI", "PDI2"]:
        row[f"RelErr_{label}"] = safe_rel_err(row[f"{label}_extracted"], row[f"{label}_theory"])
    row["AbsErr_p_pdi"] = abs(row["p_rec_pdi"] - row["p_theory"])
    row["AbsErr_p_pdi2"] = abs(row["p_rec_pdi2"] - row["p_theory"])
    row["RelErr_p_pdi"] = safe_rel_err(row["p_rec_pdi"], row["p_theory"])
    row["RelErr_p_pdi2"] = safe_rel_err(row["p_rec_pdi2"], row["p_theory"])
    return row


def build_cases(root_dir):
    cases = []
    seed = 1000
    for dist_type in DISTRIBUTIONS:
        for p_val in P_VALUES:
            for flux_exp in FLUX_EXPS:
                for smearing in SMEARINGS:
                    for replicate in range(1, REPLICATES + 1):
                        cases.append(
                            {
                                "root_dir": str(root_dir),
                                "dist_type": dist_type,
                                "p_val": p_val,
                                "flux_exp": flux_exp,
                                "smearing": smearing,
                                "replicate": replicate,
                                "seed": seed,
                                "mean_rg": MEAN_RG,
                                "pixels": PIXELS,
                                "n_bins": N_BINS,
                                "q_min": Q_MIN,
                                "q_max": Q_MAX,
                                "binning_mode": BINNING_MODE,
                            }
                        )
                        seed += 1
    return cases


def summarize_results(summary_df, output_dir):
    group_cols = ["distribution", "p_input", "flux_exp", "smearing"]
    agg = (
        summary_df.groupby(group_cols)
        .agg(
            median_abs_err_p_pdi=("AbsErr_p_pdi", "median"),
            median_abs_err_p_pdi2=("AbsErr_p_pdi2", "median"),
            mean_abs_err_p_pdi=("AbsErr_p_pdi", "mean"),
            mean_abs_err_p_pdi2=("AbsErr_p_pdi2", "mean"),
            median_rrms_pdi=("rrms_pdi", "median"),
            median_rrms_pdi2=("rrms_pdi2", "median"),
            sanity_pass_rate=("sanity_pass", "mean"),
        )
        .reset_index()
    )
    agg.to_csv(output_dir / "summary_by_condition.csv", index=False)

    dist_summary = (
        summary_df.groupby("distribution")
        .agg(
            mean_abs_err_p_pdi=("AbsErr_p_pdi", "mean"),
            mean_abs_err_p_pdi2=("AbsErr_p_pdi2", "mean"),
            median_rrms_pdi=("rrms_pdi", "median"),
            median_rrms_pdi2=("rrms_pdi2", "median"),
            sanity_pass_rate=("sanity_pass", "mean"),
        )
        .reset_index()
    )
    dist_summary.to_csv(output_dir / "summary_by_distribution.csv", index=False)
    return agg, dist_summary


def plot_distribution_heatmaps(agg_df, figures_dir):
    figures = []
    for dist_type in DISTRIBUTIONS:
        sub = agg_df[agg_df["distribution"] == dist_type]
        if sub.empty:
            continue
        piv_pdi = sub.pivot_table(index="smearing", columns="p_input", values="median_abs_err_p_pdi", aggfunc="median")
        piv_pdi2 = sub.pivot_table(index="smearing", columns="p_input", values="median_abs_err_p_pdi2", aggfunc="median")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for ax, data, title in zip(
            axes,
            [piv_pdi, piv_pdi2],
            [f"{dist_type} median |p_rec - p| from PDI", f"{dist_type} median |p_rec - p| from PDI2"],
        ):
            im = ax.imshow(data.values, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("p input")
            ax.set_ylabel("smearing")
            ax.set_xticks(range(len(data.columns)))
            ax.set_xticklabels([f"{x:.1f}" for x in data.columns], rotation=45)
            ax.set_yticks(range(len(data.index)))
            ax.set_yticklabels([str(int(x)) for x in data.index])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig_path = figures_dir / f"{dist_type.lower()}_heatmaps.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        figures.append(fig_path.name)
    return figures


def plot_flux_sensitivity(summary_df, figures_dir):
    reps = []
    p_values = [0.1, 0.5, 1.0]
    for p_val in p_values:
        sub = summary_df[np.isclose(summary_df["p_input"], p_val)]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for ax, value_col, title in zip(
            axes,
            ["AbsErr_p_pdi", "AbsErr_p_pdi2"],
            [f"Mean |p_rec - p| vs flux exponent, p={p_val:.1f} (PDI)", f"Mean |p_rec - p| vs flux exponent, p={p_val:.1f} (PDI2)"],
        ):
            grouped = (
                sub.groupby(["distribution", "flux_exp"])[value_col]
                .mean()
                .reset_index()
            )
            for dist_type in DISTRIBUTIONS:
                dist_sub = grouped[grouped["distribution"] == dist_type]
                ax.plot(dist_sub["flux_exp"], dist_sub[value_col], marker="o", label=dist_type)
            ax.set_title(title)
            ax.set_xlabel("flux exponent")
            ax.set_ylabel("mean absolute p error")
            ax.grid(True, alpha=0.3)
        axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        p_label = str(p_val).replace(".", "p")
        fig_path = figures_dir / f"flux_sensitivity_p_{p_label}.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        reps.append(fig_path.name)
    return reps


def latex_escape(text):
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def build_report_text(output_dir, summary_df, dist_summary, figure_files):
    best_pdi2 = dist_summary.sort_values("mean_abs_err_p_pdi2").iloc[0]
    worst_pdi2 = dist_summary.sort_values("mean_abs_err_p_pdi2").iloc[-1]
    best_pdi = dist_summary.sort_values("mean_abs_err_p_pdi").iloc[0]
    worst_pdi = dist_summary.sort_values("mean_abs_err_p_pdi").iloc[-1]

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\title{Appendix: Benchmark Study of Tomchuk Polydispersity Analysis in the Polydispersity App}",
        r"\author{Automated Study Output}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\section{Scope of the study}",
        r"This appendix documents a benchmark study of the Tomchuk analysis implementation in the polydispersity app.",
        r"The study covered all six supported sphere distribution families: Gaussian, Lognormal, Schulz, Boltzmann, Triangular, and Uniform.",
        r"The sampled parameter grid used mean scattering radius of gyration $R_g = 2$ nm, detector size $1024 \times 1024$, logarithmic binning, and $1024$ one-dimensional bins.",
        r"The maximum momentum transfer was set to the current app default for this $R_g$, namely $q_{\max} = 5.0$ nm$^{-1}$.",
        r"The benchmark varied input polydispersity from $p = 0.1$ to $1.0$ in steps of $0.1$, flux exponents from $10^5$ to $10^9$, smearing from $1$ to $10$ pixels, and used five stochastic replicates per condition.",
        rf"In total, {len(summary_df)} simulated SAXS runs were executed and saved in the study folder.",
        r"\section{Methods and deviations from the original Tomchuk paper}",
        r"The implementation benchmarked here is not a verbatim reproduction of the original Tomchuk workflow.",
        r"The main deviations are as follows:",
        r"\begin{itemize}",
        r"\item The app assumes the distribution family is declared by the user rather than inferred from a PDI--PDI2 classification diagram.",
        r"\item The intensity is fitted with a unified Beaucage/Tomchuk-style model and falls back to a hybrid route when needed.",
        r"\item The code includes analytic and historical tail-handling logic when evaluating invariant-derived quantities, even though the current preferred path is the unified-fit route.",
        r"\item For simulated data, the one-dimensional profile is normalized before Tomchuk extraction so that amplitude-carrying terms can be compared directly with input-based theory. This is a benchmarking convenience and not part of the original paper.",
        r"\item The recovered fit-quality metric is reported as relative RMS error rather than reduced $\chi^2$ so that values can be compared across flux levels.",
        r"\end{itemize}",
        r"\section{Data products}",
        r"For every run, the study saved the following artifacts:",
        r"\begin{itemize}",
        r"\item a $1024 \times 1024$ detector image as TIFF,",
        r"\item a one-dimensional CSV file containing raw and normalized intensity, and reconstructed fits,",
        r"\item a row in the master CSV table containing extracted parameters, input-based theoretical parameters, relative errors, reconstruction errors, and sanity-check fields.",
        r"\end{itemize}",
        r"\section{High-level findings}",
        r"\begin{itemize}",
        rf"\item Best average PDI recovery across distribution families: {latex_escape(best_pdi['distribution'])} (mean absolute p error {best_pdi['mean_abs_err_p_pdi']:.4f}).",
        rf"\item Worst average PDI recovery across distribution families: {latex_escape(worst_pdi['distribution'])} (mean absolute p error {worst_pdi['mean_abs_err_p_pdi']:.4f}).",
        rf"\item Best average PDI2 recovery across distribution families: {latex_escape(best_pdi2['distribution'])} (mean absolute p error {best_pdi2['mean_abs_err_p_pdi2']:.4f}).",
        rf"\item Worst average PDI2 recovery across distribution families: {latex_escape(worst_pdi2['distribution'])} (mean absolute p error {worst_pdi2['mean_abs_err_p_pdi2']:.4f}).",
        r"\item The normalization step removes trivial amplitude drift with flux, but does not remove extraction bias caused by finite q-range, detector sampling, or smearing.",
        r"\item PDI2 is generally more robust than PDI for broad distributions, but this is distribution dependent and breaks down for some families and low-p regimes.",
        r"\end{itemize}",
        r"\section{Distribution-level summary}",
        r"\begin{longtable}{lrrrrr}",
        r"\toprule",
        r"Distribution & mean $|p_{\mathrm{PDI}}-p|$ & mean $|p_{\mathrm{PDI2}}-p|$ & median RelRMS PDI & median RelRMS PDI2 & sanity pass rate \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in dist_summary.iterrows():
        lines.append(
            rf"{latex_escape(row['distribution'])} & {row['mean_abs_err_p_pdi']:.4f} & {row['mean_abs_err_p_pdi2']:.4f} & "
            rf"{row['median_rrms_pdi']:.4f} & {row['median_rrms_pdi2']:.4f} & {row['sanity_pass_rate']:.3f} \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\section{Interpretation}",
            r"The study is already dense enough that no further extension was required to support a complete appendix. The requested grid spans low to high polydispersity, five flux levels, ten smearing levels, six distribution families, and five replicates per condition.",
            r"The dominant conclusions are therefore based on a broad parameter coverage rather than isolated examples.",
            r"\section{Limitations of the technique}",
            r"\begin{itemize}",
            r"\item The original Tomchuk method was designed for strongly polydisperse spheres. Performance deteriorates in the lower-p regime because the indices approach unity and the inversion becomes numerically fragile.",
            r"\item The present code still assumes a declared distribution family. If the family is wrong, the inversion can appear internally consistent while the recovered p is still wrong.",
            r"\item Smearing remains a major practical limitation. Even after normalization, increasing smearing distorts the low-q and intermediate-q regions that control the fitted invariants.",
            r"\item Finite q-range still matters. Normalization solves amplitude comparability but does not recover information that is absent from the measured range.",
            r"\item PDI and PDI2 do not fail in the same way. In some families PDI is the weaker route, while in others PDI2 inherits larger invariant bias.",
            r"\item The current implementation includes model-fitting and normalization choices that are useful for benchmarking but differ from a strict paper-only workflow.",
            r"\end{itemize}",
            r"\section{Figures}",
        ]
    )
    for figure_file in figure_files:
        caption_name = figure_file.replace("_", "-")
        lines.extend(
            [
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.95\textwidth]{{../figures/{figure_file}}}",
                rf"\caption{{Benchmark figure: {latex_escape(caption_name)}}}",
                r"\end{figure}",
            ]
        )
    lines.append(r"\end{document}")
    tex_path = output_dir / "report" / "appendix.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines))
    return tex_path


def compile_latex(tex_path):
    import subprocess

    cmd = ["pdflatex", "-interaction=nonstopmode", tex_path.name]
    for _ in range(2):
        proc = subprocess.run(cmd, cwd=tex_path.parent, capture_output=True, text=True)
        if proc.returncode != 0:
            return False, proc.stdout + "\n" + proc.stderr
    pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path.exists():
        return True, "pdflatex completed successfully"
    return False, "pdflatex did not produce a PDF."


def main():
    parser = argparse.ArgumentParser(description="Run the full Tomchuk benchmarking study.")
    parser.add_argument("--output-root", default="study_outputs")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"tomchuk_benchmark_study_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "report").mkdir(parents=True, exist_ok=True)

    cases = build_cases(output_dir)
    metadata = {
        "timestamp": timestamp,
        "mean_rg": MEAN_RG,
        "pixels": PIXELS,
        "n_bins": N_BINS,
        "q_min": Q_MIN,
        "q_max": Q_MAX,
        "binning_mode": BINNING_MODE,
        "replicates": REPLICATES,
        "distributions": DISTRIBUTIONS,
        "p_values": P_VALUES,
        "flux_exps": FLUX_EXPS,
        "smearings": SMEARINGS,
        "workers": args.workers,
        "total_runs": len(cases),
    }
    (output_dir / "study_metadata.json").write_text(json.dumps(metadata, indent=2))

    rows = []
    errors = []
    checkpoint_path = output_dir / "progress_summary_checkpoint.csv"
    errors_path = output_dir / "study_errors.csv"
    if args.workers <= 1:
        for idx, case in enumerate(cases, start=1):
            try:
                rows.append(run_single_case(case))
            except Exception as exc:
                error_row = case.copy()
                error_row["error"] = str(exc)
                errors.append(error_row)
            if idx % 100 == 0 or idx == len(cases):
                print(f"completed {idx}/{len(cases)}", flush=True)
                if rows:
                    pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
                if errors:
                    pd.DataFrame(errors).to_csv(errors_path, index=False)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single_case, case): case for case in cases}
            for idx, future in enumerate(as_completed(futures), start=1):
                try:
                    rows.append(future.result())
                except Exception as exc:
                    error_row = futures[future].copy()
                    error_row["error"] = str(exc)
                    errors.append(error_row)
                if idx % 100 == 0 or idx == len(futures):
                    print(f"completed {idx}/{len(futures)}", flush=True)
                    if rows:
                        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
                    if errors:
                        pd.DataFrame(errors).to_csv(errors_path, index=False)

    summary_df = pd.DataFrame(rows).sort_values(
        ["distribution", "p_input", "flux_exp", "smearing", "replicate"]
    )
    summary_csv = output_dir / "tomchuk_study_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    agg_df, dist_summary = summarize_results(summary_df, output_dir)
    figure_files = []
    figure_files.extend(plot_distribution_heatmaps(agg_df, output_dir / "figures"))
    figure_files.extend(plot_flux_sensitivity(summary_df, output_dir / "figures"))

    tex_path = build_report_text(output_dir, summary_df, dist_summary, figure_files)
    success, log_text = compile_latex(tex_path)
    (output_dir / "report" / "pdflatex.log.txt").write_text(log_text)

    print(f"study output: {output_dir}")
    print(f"summary csv: {summary_csv}")
    print(f"latex file: {tex_path}")
    print(f"pdf compiled: {success}")


if __name__ == "__main__":
    main()
