from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path("studies/tomchuk_benchmark_20260329")
TABLES_DIR = BASE_DIR / "tables"
FIGURES_DIR = BASE_DIR / "figures"
REPORTS_DIR = BASE_DIR / "reports"

DISTRIBUTIONS = ["Gaussian", "Lognormal", "Schulz", "Boltzmann", "Triangular", "Uniform"]
PARAM_COLS = [
    ("abs_rel_Rg", r"|rel $R_g$|"),
    ("abs_rel_G", r"|rel $G$|"),
    ("abs_rel_B", r"|rel $B$|"),
    ("abs_rel_Q", r"|rel $Q$|"),
    ("abs_rel_lc", r"|rel $l_c$|"),
]


def _best_pair_map(summary_df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    best = {}
    for dist_type, group in summary_df.groupby("distribution"):
        pair = (
            group.groupby(["flux_exp", "smearing"])[["abs_err_p_pdi", "abs_err_p_pdi2"]]
            .mean()
            .assign(score=lambda df: df["abs_err_p_pdi"] + df["abs_err_p_pdi2"])
            .sort_values("score")
            .reset_index()
            .iloc[0]
        )
        best[dist_type] = (int(pair["flux_exp"]), int(pair["smearing"]))
    return best


def plot_sanity_heatmap(aggregate_df: pd.DataFrame) -> Path:
    fluxes = sorted(aggregate_df["flux_exp"].unique())
    smearings = sorted(aggregate_df["smearing"].unique())
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    axes = axes.ravel()
    for ax, dist_type in zip(axes, DISTRIBUTIONS):
        group = aggregate_df.loc[aggregate_df["distribution"] == dist_type]
        pivot = group.groupby(["smearing", "flux_exp"])["sanity_pass_rate"].mean().unstack()
        pivot = pivot.reindex(index=smearings, columns=fluxes)
        im = ax.imshow(pivot.values, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="magma")
        ax.set_title(dist_type)
        ax.set_xticks(range(len(fluxes)))
        ax.set_xticklabels(fluxes)
        ax.set_yticks(range(len(smearings)))
        ax.set_yticklabels(smearings)
        ax.set_xlabel("Flux exponent")
        ax.set_ylabel("Smearing (pixels)")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.85, label="Mean sanity-pass rate")
    fig.suptitle("Mean sanity-pass rate over p and replicate")
    out = FIGURES_DIR / "sanity_pass_heatmap.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_recovered_p_consolidated(summary_df: pd.DataFrame, best_pairs: dict[str, tuple[int, int]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, col, title in [
        (axes[0], "p_rec_pdi", "PDI recovery"),
        (axes[1], "p_rec_pdi2", "PDI2 recovery"),
    ]:
        for dist_type in DISTRIBUTIONS:
            flux_exp, smearing = best_pairs[dist_type]
            subset = summary_df.loc[
                (summary_df["distribution"] == dist_type)
                & (summary_df["flux_exp"] == flux_exp)
                & (summary_df["smearing"] == smearing)
            ]
            grouped = subset.groupby("p_input").agg(mean=(col, "mean"), std=(col, "std")).reset_index()
            ax.errorbar(
                grouped["p_input"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                linewidth=1.2,
                capsize=2.5,
                label=dist_type,
            )
        ax.plot([0.1, 1.0], [0.1, 1.0], "k--", linewidth=1, label="Ideal" if title == "PDI recovery" else None)
        ax.set_title(title)
        ax.set_xlabel("Input p")
        ax.set_ylabel("Recovered p")
        ax.set_xlim(0.08, 1.02)
        ax.set_ylim(0.0, 1.25)
    axes[0].legend(fontsize=8, ncol=2)
    out = FIGURES_DIR / "recovered_p_consolidated_mean_std.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_rg_consolidated(summary_df: pd.DataFrame, best_pairs: dict[str, tuple[int, int]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, mode, title in [
        (axes[0], "selected", "Selected Tomchuk Rg"),
        (axes[1], "guinier", "Raw Guinier Rg"),
    ]:
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        for dist_type in DISTRIBUTIONS:
            flux_exp, smearing = best_pairs[dist_type]
            subset = summary_df.loc[
                (summary_df["distribution"] == dist_type)
                & (summary_df["flux_exp"] == flux_exp)
                & (summary_df["smearing"] == smearing)
            ]
            grouped = subset.groupby("p_input").agg(
                mean_sel=("rel_err_Rg", "mean"),
                std_sel=("rel_err_Rg", "std"),
                mean_gui=("Rg_guinier", "mean"),
                std_gui=("Rg_guinier", "std"),
                rg_theory=("Rg_theory", "mean"),
            ).reset_index()
            if mode == "selected":
                y = grouped["mean_sel"]
                yerr = grouped["std_sel"]
            else:
                y = (grouped["mean_gui"] - grouped["rg_theory"]) / grouped["rg_theory"]
                yerr = grouped["std_gui"] / grouped["rg_theory"]
            ax.errorbar(
                grouped["p_input"],
                y,
                yerr=yerr,
                marker="o",
                linewidth=1.2,
                capsize=2.5,
                label=dist_type,
            )
        ax.set_title(title)
        ax.set_xlabel("Input p")
        ax.set_ylabel("Relative Rg error")
        ax.set_xlim(0.08, 1.02)
    axes[0].legend(fontsize=8, ncol=2)
    out = FIGURES_DIR / "rg_consolidated_mean_std.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_rrms_consolidated(summary_df: pd.DataFrame, best_pairs: dict[str, tuple[int, int]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, col, title in [
        (axes[0], "rrms_pdi", "PDI reconstruction"),
        (axes[1], "rrms_pdi2", "PDI2 reconstruction"),
    ]:
        for dist_type in DISTRIBUTIONS:
            flux_exp, smearing = best_pairs[dist_type]
            subset = summary_df.loc[
                (summary_df["distribution"] == dist_type)
                & (summary_df["flux_exp"] == flux_exp)
                & (summary_df["smearing"] == smearing)
            ]
            grouped = subset.groupby("p_input").agg(mean=(col, "mean"), std=(col, "std")).reset_index()
            ax.errorbar(
                grouped["p_input"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                linewidth=1.2,
                capsize=2.5,
                label=dist_type,
            )
        ax.set_title(title)
        ax.set_xlabel("Input p")
        ax.set_ylabel("Relative RMS error")
        ax.set_xlim(0.08, 1.02)
    axes[0].legend(fontsize=8, ncol=2)
    out = FIGURES_DIR / "rrms_consolidated_mean_std.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_parameter_driver_summary(summary_df: pd.DataFrame) -> Path:
    flux_df = summary_df.groupby("flux_exp").agg(
        abs_err_pdi=("abs_err_p_pdi", "mean"),
        abs_err_pdi2=("abs_err_p_pdi2", "mean"),
        abs_rel_Rg=("rel_err_Rg", lambda x: x.abs().mean()),
        abs_rel_G=("rel_err_G", lambda x: x.abs().mean()),
        abs_rel_B=("rel_err_B", lambda x: x.abs().mean()),
        abs_rel_Q=("rel_err_Q", lambda x: x.abs().mean()),
        abs_rel_lc=("rel_err_lc", lambda x: x.abs().mean()),
    ).reset_index()
    smear_df = summary_df.groupby("smearing").agg(
        abs_err_pdi=("abs_err_p_pdi", "mean"),
        abs_err_pdi2=("abs_err_p_pdi2", "mean"),
        abs_rel_Rg=("rel_err_Rg", lambda x: x.abs().mean()),
        abs_rel_G=("rel_err_G", lambda x: x.abs().mean()),
        abs_rel_B=("rel_err_B", lambda x: x.abs().mean()),
        abs_rel_Q=("rel_err_Q", lambda x: x.abs().mean()),
        abs_rel_lc=("rel_err_lc", lambda x: x.abs().mean()),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for ax, df, xcol, title in [
        (axes[0], flux_df, "flux_exp", "Driver metrics versus photon count"),
        (axes[1], smear_df, "smearing", "Driver metrics versus smearing"),
    ]:
        for col, label in PARAM_COLS:
            ax.plot(df[xcol], df[col], marker="o", linewidth=1.2, label=label)
        ax.set_title(title)
        ax.set_xlabel("Flux exponent" if xcol == "flux_exp" else "Smearing (pixels)")
        ax.set_ylabel("Mean absolute relative extraction error")
        ax.legend(fontsize=8, ncol=2)
    out = FIGURES_DIR / "parameter_driver_summary.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def _distribution_summary_text(summary_df: pd.DataFrame) -> str:
    lines = []
    best_pairs = _best_pair_map(summary_df)
    for dist_type in DISTRIBUTIONS:
        group = summary_df.loc[summary_df["distribution"] == dist_type]
        flux_exp, smearing = best_pairs[dist_type]
        best_group = group.loc[(group["flux_exp"] == flux_exp) & (group["smearing"] == smearing)]
        mean_pdi = group["abs_err_p_pdi"].mean()
        mean_pdi2 = group["abs_err_p_pdi2"].mean()
        sanity = group["sanity_pass"].mean()
        better = "PDI2" if mean_pdi2 < mean_pdi else "PDI"
        lines.append(
            f"\\paragraph{{{dist_type}.}} Across the full matrix, the mean absolute error in recovered polydispersity was "
            f"{mean_pdi:.3f} for PDI and {mean_pdi2:.3f} for PDI2, so {better} was the more accurate estimator on average. "
            f"The best combined condition occurred at flux exponent {flux_exp} and smearing {smearing}. "
            f"At that condition, the mean relative RMS reconstruction errors were {best_group['rrms_pdi'].mean():.3f} for PDI and "
            f"{best_group['rrms_pdi2'].mean():.3f} for PDI2, while the full-matrix sanity-pass rate was {100.0 * sanity:.1f}\\%."
        )
    return "\n".join(lines)


def _correlation_rows(summary_df: pd.DataFrame) -> str:
    rows = []
    for label, target in [("PDI", "abs_err_p_pdi"), ("PDI2", "abs_err_p_pdi2")]:
        rows.append(
            f"{label} & "
            f"{summary_df[target].corr(summary_df['rel_err_Rg'].abs()):.3f} & "
            f"{summary_df[target].corr(summary_df['rel_err_G'].abs()):.3f} & "
            f"{summary_df[target].corr(summary_df['rel_err_B'].abs()):.3f} & "
            f"{summary_df[target].corr(summary_df['rel_err_Q'].abs()):.3f} & "
            f"{summary_df[target].corr(summary_df['rel_err_lc'].abs()):.3f} \\\\"
        )
    return "\n".join(rows)


def _driver_notes(summary_df: pd.DataFrame) -> str:
    notes = []
    for dist_type, group in summary_df.groupby("distribution"):
        pdi_corr = {
            "Rg": group["abs_err_p_pdi"].corr(group["rel_err_Rg"].abs()),
            "G": group["abs_err_p_pdi"].corr(group["rel_err_G"].abs()),
            "B": group["abs_err_p_pdi"].corr(group["rel_err_B"].abs()),
            "Q": group["abs_err_p_pdi"].corr(group["rel_err_Q"].abs()),
            "lc": group["abs_err_p_pdi"].corr(group["rel_err_lc"].abs()),
        }
        pdi2_corr = {
            "Rg": group["abs_err_p_pdi2"].corr(group["rel_err_Rg"].abs()),
            "G": group["abs_err_p_pdi2"].corr(group["rel_err_G"].abs()),
            "B": group["abs_err_p_pdi2"].corr(group["rel_err_B"].abs()),
            "Q": group["abs_err_p_pdi2"].corr(group["rel_err_Q"].abs()),
            "lc": group["abs_err_p_pdi2"].corr(group["rel_err_lc"].abs()),
        }
        pdi_driver = max(pdi_corr, key=pdi_corr.get)
        pdi2_driver = max(pdi2_corr, key=pdi2_corr.get)
        notes.append(
            f"\\paragraph{{{dist_type}.}} The dominant driver of PDI error is {pdi_driver} "
            f"($r={pdi_corr[pdi_driver]:.3f}$), whereas the dominant driver of PDI2 error is {pdi2_driver} "
            f"($r={pdi2_corr[pdi2_driver]:.3f}$)."
        )
    return "\n".join(notes)


def write_report(summary_df: pd.DataFrame, aggregate_df: pd.DataFrame) -> Path:
    overall_pdi = summary_df["abs_err_p_pdi"].mean()
    overall_pdi2 = summary_df["abs_err_p_pdi2"].mean()
    overall_sanity = summary_df["sanity_pass"].mean()
    flux_df = summary_df.groupby("flux_exp").agg(
        abs_err_pdi=("abs_err_p_pdi", "mean"),
        abs_err_pdi2=("abs_err_p_pdi2", "mean"),
        abs_rel_Rg=("rel_err_Rg", lambda x: x.abs().mean()),
        abs_rel_G=("rel_err_G", lambda x: x.abs().mean()),
        abs_rel_B=("rel_err_B", lambda x: x.abs().mean()),
        abs_rel_Q=("rel_err_Q", lambda x: x.abs().mean()),
        abs_rel_lc=("rel_err_lc", lambda x: x.abs().mean()),
    ).reset_index()
    smear_df = summary_df.groupby("smearing").agg(
        abs_err_pdi=("abs_err_p_pdi", "mean"),
        abs_err_pdi2=("abs_err_p_pdi2", "mean"),
        abs_rel_Rg=("rel_err_Rg", lambda x: x.abs().mean()),
        abs_rel_G=("rel_err_G", lambda x: x.abs().mean()),
        abs_rel_B=("rel_err_B", lambda x: x.abs().mean()),
        abs_rel_Q=("rel_err_Q", lambda x: x.abs().mean()),
        abs_rel_lc=("rel_err_lc", lambda x: x.abs().mean()),
    ).reset_index()
    low_flux = flux_df.loc[flux_df["flux_exp"] == flux_df["flux_exp"].min()].iloc[0]
    high_flux = flux_df.loc[flux_df["flux_exp"] == flux_df["flux_exp"].max()].iloc[0]
    low_smear = smear_df.loc[smear_df["smearing"] == smear_df["smearing"].min()].iloc[0]
    high_smear = smear_df.loc[smear_df["smearing"] == smear_df["smearing"].max()].iloc[0]

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{setspace}}
\title{{Appendix: Benchmark Study of Tomchuk Polydispersity Analysis in the Polydispersity App}}
\author{{Roy Beck Barkai and OpenAI Codex-assisted benchmark workflow}}
\date{{{datetime.now().strftime("%Y-%m-%d")}}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This appendix presents a systematic benchmark of the Tomchuk polydispersity-analysis workflow implemented in the polydispersity app.
The study spans six particle-size distribution families, input polydispersities from 0.1 to 1.0, photon-count exponents from $10^5$ to $10^9$, detector smearing from 1 to 10 pixels, and five stochastic repeats per condition at fixed $R_g = 2$ nm.
In addition to reporting width-recovery accuracy, the appendix analyzes the extraction errors in $R_g$, $G$, $B$, $Q$ and $l_c$ to identify the dominant mechanisms responsible for deviations in recovered polydispersity.
The main conclusions are that PDI2 is generally the more reliable width estimator, low photon count primarily harms the low-$q$ radius extraction, high smearing primarily harms the Porod and invariant terms, and the strongest departures from the original Tomchuk paper arise from practical finite-data effects and from the app's declared-family workflow.
\end{{abstract}}
\setstretch{{1.05}}

\section{{Study Design}}
This appendix reports a reproducible benchmark study of the app's Tomchuk analysis across six supported distribution families: Gaussian, Lognormal, Schulz, Boltzmann, Triangular, and Uniform.
The benchmark fixed $R_g = 2$ nm, detector size $1024 \times 1024$, 1024 radial bins, logarithmic binning and $q_{{\max}} = 5.0$ nm$^{{-1}}$.
The input polydispersity ranged from $p=0.1$ to $1.0$ in steps of 0.1, the photon-count exponent ranged from $10^5$ to $10^9$, the detector smearing ranged from 1 to 10 pixels, and each condition was repeated five times with distinct random seeds.
This produced 15\,000 analyzed SAXS profiles and a matched set of 15\,000 detector TIFF files.

\section{{Workflow Implemented In The App}}
For each synthetic run, the app simulated a 2D detector pattern and corresponding 1D radial profile, normalized the 1D sphere intensity onto the input-theory amplitude scale, extracted Tomchuk quantities using the current unified-fit-first pipeline, reconstructed SAXS curves from the recovered PDI and PDI2 solutions, and evaluated reconstruction quality with a relative RMS error metric.
After the recent Guinier refresh, the study tables also store the raw Guinier estimates separately from the selected Tomchuk extraction path, allowing direct comparison between low-$q$ radius extraction and the final values propagated into the invariant analysis.

\section{{Relation To The Original Tomchuk Method}}
Tomchuk's 2023 paper proposes a seven-step workflow: fit the unified exponential/power-law law to obtain $R_g$, $G$ and $B$; determine PDI; calculate $Q$ and $l_c$; determine PDI2; use the PDI--PDI2 map to identify the distribution family; recover $p$; and finally recover the mean particle size.
\subsection{{Points Of Agreement}}
The present implementation follows the same overall physical logic for spherical particles, and it agrees with the paper in several key respects:
\begin{{itemize}}
\item The unified exponential/power-law form is used as the preferred route to $G$, $R_g$ and $B$.
\item $Q$ and $l_c$ are evaluated from the analytic expressions tied to that fit rather than relying only on finite-range numerical integrals, matching the paper's recommendation for reducing PDI2 error.
\item The benchmark confirms the paper's warning that low-polydispersity cases are the most fragile, especially around and below the practical sensitivity threshold near $p \approx 0.25$.
\item Simultaneous consideration of PDI and PDI2 remains informative because the two indices fail differently for different distribution families.
\end{{itemize}}
\subsection{{Main Deviations}}
The main deviations from the paper are also clear:
\begin{{itemize}}
\item The distribution family is declared in advance instead of being inferred from the PDI--PDI2 map.
\item A hybrid fallback path remains in the code for edge cases, whereas the paper is centered on the unified-fit route.
\item The benchmark includes detector smearing, finite binning and Poisson noise, so it is deliberately more demanding than the ideal algebraic derivation.
\item Earlier versions of the code used explicit tail corrections for $Q$ and $l_c$; although the benchmark here prefers the analytic unified-fit path, that historical implementation choice still helps explain some practical deviations from the ideal paper behavior.
\end{{itemize}}

\section{{Global Accuracy Trends}}
Across the full benchmark, the average absolute error in recovered $p$ was {overall_pdi:.3f} for PDI and {overall_pdi2:.3f} for PDI2, while the overall sanity-pass rate was {100.0 * overall_sanity:.1f}\%.
The dominant global pattern is that PDI2 is usually the more accurate width estimator, whereas PDI can still give comparable or better forward reconstructions in selected families because it emphasizes higher moments differently.
Figures~\ref{{fig:pdi_heatmap}} and \ref{{fig:pdi2_heatmap}} show that this is strongly family-dependent.

Figure~\ref{{fig:pdi_heatmap}} confirms that the classical PDI route is most fragile for families whose tails or truncations strongly perturb the higher moments, especially Triangular and Uniform distributions.
That is qualitatively consistent with Tomchuk's discussion of PDI as a quantity involving high-order moments up to eighth order.
Figure~\ref{{fig:pdi2_heatmap}} shows a broader low-error operating region for PDI2, particularly for Gaussian and Boltzmann families, which is again in line with Tomchuk's motivation for introducing a lower-order-moment companion index.

Figure~\ref{{fig:p_error_curves}} shows that even after selecting the best average flux/smearing condition for each family, both very small and very large input widths remain harder to recover than the mid-range.
This agrees with the paper's sensitivity warning: when $p$ is small, the indices approach unity and become inversion-sensitive; when $p$ is large, errors in fitted moments and tails are amplified.

\section{{Consolidated Multi-Run Results}}
\subsection{{Recovered Width}}
The study figures were reorganized so that distributions are compared directly in shared panels, with separate panels for PDI and PDI2.
Figure~\ref{{fig:recovered_p_errorbars}} shows the recovered width itself, averaged over the five repeats with one-standard-deviation error bars.
This makes the estimator bias directly visible.
Gaussian and Boltzmann families remain the most consistent with the identity line, whereas Triangular and Uniform families remain strongly biased, especially in the PDI panel.

\subsection{{Radius Extraction Versus Final Selected Radius}}
Figure~\ref{{fig:rg_errorbars}} separates the raw Guinier radius from the selected Tomchuk radius used downstream.
The key point is that the raw Guinier estimate is not always the dominant source of error under otherwise favorable conditions.
The selected Tomchuk radius can still drift away from the raw Guinier value because the unified fit rebalances the entire curve, not only the strict low-$q$ regime.
This helps explain why visually acceptable low-$q$ behavior does not guarantee correct recovery of $p$.

\subsection{{Forward-Reconstruction Quality}}
Figure~\ref{{fig:rrms_errorbars}} then shows the quality of the reconstructed SAXS curves.
This figure highlights an important practical distinction: the estimator that recovers $p$ most accurately is not always the one that gives the best forward reconstruction.
For example, PDI2 often recovers the width better for Boltzmann and Lognormal families, yet its reconstructed curve can still be worse because the lower-order-moment solution does not necessarily preserve the entire curve shape.

\section{{What Drives The Errors In Recovered Polydispersity}}
\subsection{{Photon Count And Smearing As Separate Failure Mechanisms}}
The central question is which extracted quantities are most responsible for the drift in recovered $p$.
Figure~\ref{{fig:parameter_drivers}} addresses this directly by showing how the absolute extraction errors in $R_g$, $G$, $B$, $Q$ and $l_c$ change with photon count and smearing.
The photon-count panel shows a rapid improvement of the low-$q$ radius term: the mean absolute relative error in $R_g$ falls from {low_flux['abs_rel_Rg']:.3f} at $10^{{{int(low_flux['flux_exp'])}}}$ to {high_flux['abs_rel_Rg']:.3f} at $10^{{{int(high_flux['flux_exp'])}}}$, whereas the corresponding errors in $G$, $B$, $Q$ and $l_c$ change only weakly once the count is above roughly $10^6$.
This indicates that low photon count primarily hurts the low-$q$ extraction stage, which in turn explains why noisy cases can fail even before the invariant calculation becomes dominant.

The smearing panel reveals a different mechanism.
From smearing {int(low_smear['smearing'])} to {int(high_smear['smearing'])} pixels, the mean absolute relative errors in $B$, $Q$ and $l_c$ rise from {low_smear['abs_rel_B']:.3f}, {low_smear['abs_rel_Q']:.3f} and {low_smear['abs_rel_lc']:.3f} to {high_smear['abs_rel_B']:.3f}, {high_smear['abs_rel_Q']:.3f} and {high_smear['abs_rel_lc']:.3f}, respectively.
This is the clearest signature that high smearing corrupts the Porod and invariant terms much more strongly than the low-$q$ radius term.
As a result, the PDI2 pathway, which depends directly on $B$, $Q$ and $l_c$, deteriorates mainly at high smearing, whereas low photon count mostly attacks the quality of the initial radius extraction.

\subsection{{Correlation Analysis Of Error Propagation}}
This interpretation is supported quantitatively by the correlation analysis in Table~\ref{{tab:drivers}}.
Globally, PDI error correlates most strongly with the extracted $Q$, $B$ and $G$ errors, while PDI2 error correlates most strongly with $Q$, $l_c$ and $B$ errors.
The weak global correlation with $R_g$ for PDI reflects the fact that a modest radius drift alone is usually not enough to explain the large width errors unless it also perturbs the invariant-carrying quantities.
In other words, the core reason the recovered width fails is usually not a pure Guinier-radius failure but a drift in the combination of fitted amplitude and invariant terms that feed the Tomchuk indices.

\begin{{table}}[H]
\centering
\caption{{Correlation between absolute width-recovery error and absolute relative extraction error in the Tomchuk input quantities over the full benchmark. Larger values indicate a stronger association between parameter drift and failure in recovered $p$.}}
\label{{tab:drivers}}
\begin{{tabular}}{{lccccc}}
\toprule
Estimator & $R_g$ & $G$ & $B$ & $Q$ & $l_c$ \\
\midrule
{_correlation_rows(summary_df)}
\bottomrule
\end{{tabular}}
\end{{table}}

The distribution-resolved driver analysis also reinforces this interpretation:
{_driver_notes(summary_df)}

\section{{Distribution-Specific Interpretation}}
{_distribution_summary_text(summary_df)}

\section{{Comparison With Tomchuk's Conclusions}}
The benchmark is broadly consistent with the original paper in three main ways.
First, it confirms that the method is least reliable in the near-monodisperse limit.
Second, it confirms that PDI2 is often the more stable indicator of width once one moves beyond Gaussian-like shapes.
Third, it supports the paper's central claim that the combined behavior of PDI and PDI2 carries information about the underlying distribution family, even though the present app still requires that family to be declared in advance.

The main quantitative differences from the paper come from the fact that our benchmark is harder than the ideal derivation.
Tomchuk derives the method from the fitted unified law and then uses those fitted parameters directly.
In the app, the workflow must also contend with detector discretization, simulated smearing, Poisson noise, finite $q$ coverage and residual fallback logic for edge cases.
These effects matter most for distributions whose higher moments are especially sensitive to tails or truncation, which is why Triangular and Uniform families depart most strongly from the ideal paper behavior while Gaussian and Boltzmann cases remain closer to the original expectations.

\section{{Limitations Of The Technique}}
\begin{{itemize}}
\item The method remains highly sensitive to the quality of the extracted unified parameters. Even a modest drift in $G$, $B$, $Q$ or $l_c$ can produce a much larger drift in recovered $p$.
\item The practical operating window depends strongly on the assumed distribution family. There is no single universal flux or smearing target that guarantees success across all families.
\item Low-polydispersity cases remain intrinsically ill-conditioned because PDI and PDI2 both approach unity, matching the sensitivity warning near $p \approx 0.25$ emphasized by Tomchuk.
\item Very broad distributions are also difficult because the relevant indices depend on moments that are strongly affected by tails, truncation and finite-$q$ coverage.
\item The benchmark still does not solve the paper's unknown-family problem, since the app presently analyzes each run under the declared family rather than inferring the family from a PDI--PDI2 map.
\item The simulated benchmark omits real experimental complications such as backgrounds, detector masks, calibration uncertainty and interparticle structure factors.
\end{{itemize}}

\section{{Data Products}}
The full run-by-run results are stored in \texttt{{tables/benchmark\_summary.csv}}, the aggregated condition table in \texttt{{tables/aggregate\_by\_condition.csv}}, and the refreshed best/worst catalog in \texttt{{tables/best\_worst\_cases.csv}}.
The heavy raw detector TIFF files and 1D CSV profiles are stored outside the Git repository in the external Dropbox study-data directory documented previously.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/heatmaps_abs_err_pdi.png}}
\caption{{\textbf{{Flux--smearing dependence of PDI-based recovery.}} Each panel shows, for one distribution family, the mean absolute error $|p_{{\mathrm{{rec}}}}-p_{{\mathrm{{input}}}}|$ obtained from the classical PDI estimator after averaging over all input $p$ values and the five replicates. Darker colors indicate lower error. The figure emphasizes the strong family dependence of the high-order-moment PDI route: Gaussian systems retain a moderate working region, whereas Triangular and Uniform distributions exhibit broad regions of failure.}}
\label{{fig:pdi_heatmap}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/heatmaps_abs_err_pdi2.png}}
\caption{{\textbf{{Flux--smearing dependence of PDI2-based recovery.}} Each panel shows the same benchmark averaging as in Fig.~\ref{{fig:pdi_heatmap}}, but for the lower-order-moment PDI2 estimator. Relative to PDI, PDI2 retains a noticeably broader low-error region in most families, particularly Gaussian and Boltzmann. This figure provides the clearest global evidence that PDI2 is the more robust width estimator in the current implementation, although its advantage is still reduced for strongly truncated families such as Uniform and Triangular.}}
\label{{fig:pdi2_heatmap}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/p_error_curves_best_conditions.png}}
\caption{{\textbf{{Best-condition recovery error as a function of input polydispersity.}} For each distribution family, the flux exponent and smearing value that minimize the combined mean PDI and PDI2 error were first selected. The plotted curves then show the mean absolute width-recovery error versus input $p$ for those family-specific best conditions. This representation isolates the shape of the $p$ dependence after the experimental settings have been locally optimized and shows that both very small and very large widths remain difficult even under otherwise favorable conditions.}}
\label{{fig:p_error_curves}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/representative_best_fits.png}}
\caption{{\textbf{{Representative best-case forward reconstructions.}} For each distribution family, the run with the smallest combined PDI and PDI2 width error is shown. Black markers are the normalized simulated SAXS data; colored lines are the forward reconstructions obtained from the PDI-based and PDI2-based recovered parameters. The figure demonstrates that the current Tomchuk workflow can recover both the curve shape and the width accurately for selected families and operating points, especially when the underlying extraction remains self-consistent.}}
\label{{fig:best_fits}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/representative_worst_fits.png}}
\caption{{\textbf{{Representative worst-case forward reconstructions.}} For each distribution family, the run with the largest combined width error is shown. These cases illustrate several distinct failure modes: collapse toward an artificially narrow distribution, excessive broadening, and reconstructions that remain visually plausible while encoding the wrong width. The contrast with Fig.~\ref{{fig:best_fits}} emphasizes that curve agreement alone is not sufficient evidence for correct Tomchuk parameter recovery.}}
\label{{fig:worst_fits}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/sanity_pass_heatmap.png}}
\caption{{\textbf{{Mean sanity-pass rate across the benchmark grid.}} For each distribution family, the plotted value is the fraction of runs that pass all theory-versus-extraction sanity checks after averaging over the five replicates and the input-$p$ grid. The sanity checks compare extracted $R_g$, $G$, $B$, $Q$, $l_c$, PDI, PDI2 and recovered sizes against the simulated-theory values. This figure therefore maps not only recovery accuracy but also internal physical consistency of the extracted Tomchuk quantities.}}
\label{{fig:sanity_heatmap}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/recovered_p_consolidated_mean_std.png}}
\caption{{\textbf{{Replicate-resolved recovered width in consolidated panels.}} The left panel shows PDI-based recovery and the right panel shows PDI2-based recovery. In each panel, all six distribution families are plotted together at their respective best average flux/smearing condition, with the marker indicating the mean recovered $p$ over five repeats and the error bar indicating one standard deviation. This format makes inter-family bias and repeatability directly comparable within each estimator.}}
\label{{fig:recovered_p_errorbars}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/rg_consolidated_mean_std.png}}
\caption{{\textbf{{Replicate-resolved $R_g$ accuracy in consolidated panels.}} The left panel shows the selected Tomchuk radius propagated into the invariant analysis, and the right panel shows the raw Guinier radius extracted from the low-$q$ region. All distributions are shown together at their respective best average operating condition. The contrast between the two panels distinguishes pure low-$q$ radius extraction from the final full-curve radius selected by the unified-fit workflow.}}
\label{{fig:rg_errorbars}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/rrms_consolidated_mean_std.png}}
\caption{{\textbf{{Replicate-resolved reconstruction quality in consolidated panels.}} The left panel shows the relative RMS error of the PDI-based reconstructions and the right panel shows the same metric for PDI2-based reconstructions. Error bars are one standard deviation over the five repeats. Comparing this figure with Fig.~\ref{{fig:recovered_p_errorbars}} demonstrates that the most accurate width estimator is not necessarily the estimator that best reconstructs the full SAXS curve.}}
\label{{fig:rrms_errorbars}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\textwidth]{{../figures/parameter_driver_summary.png}}
\caption{{\textbf{{Mean extraction-error drivers versus photon count and smearing.}} The left panel shows how the mean absolute relative extraction errors in $R_g$, $G$, $B$, $Q$ and $l_c$ vary with photon-count exponent after averaging over all distributions, $p$ values and repeats. The right panel shows the same quantities as a function of detector smearing. These consolidated curves identify the primary mechanism of failure in different experimental regimes: low photon count mainly degrades the low-$q$ radius term, whereas increasing smearing predominantly inflates the Porod and invariant terms $B$, $Q$ and $l_c$.}}
\label{{fig:parameter_drivers}}
\end{{figure}}

\section{{Reproducibility}}
The benchmark data were produced with \texttt{{studies/run\_tomchuk\_benchmark.py}}.
The present expanded appendix and the consolidated figures were refreshed from the saved benchmark tables with \texttt{{studies/refresh\_appendix\_report.py}}, and the full benchmark tables were updated after the Guinier extraction fix with \texttt{{studies/reanalyze\_saved\_profiles.py}}.

\end{{document}}
"""
    tex_path = REPORTS_DIR / "tomchuk_benchmark_appendix.tex"
    tex_path.write_text(tex)
    return tex_path


def compile_latex(tex_path: Path) -> None:
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )


def main() -> None:
    summary_df = pd.read_csv(TABLES_DIR / "benchmark_summary.csv")
    aggregate_df = pd.read_csv(TABLES_DIR / "aggregate_by_condition.csv")
    best_pairs = _best_pair_map(summary_df)

    plot_sanity_heatmap(aggregate_df)
    plot_recovered_p_consolidated(summary_df, best_pairs)
    plot_rg_consolidated(summary_df, best_pairs)
    plot_rrms_consolidated(summary_df, best_pairs)
    plot_parameter_driver_summary(summary_df)
    tex_path = write_report(summary_df, aggregate_df)
    compile_latex(tex_path)
    print(f"Updated report: {tex_path}")


if __name__ == "__main__":
    main()
