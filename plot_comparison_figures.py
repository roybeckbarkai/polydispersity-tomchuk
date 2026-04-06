"""
plot_comparison_figures.py
===========================
Generate publication-quality figures for the TENOR-SAXS vs. Tomchuk comparison study.

Usage:
    python plot_comparison_figures.py --study-dir <path_to_study_output>

Produces PDF + PNG figures in <study_dir>/figures/.
All source data for each figure is also saved as a CSV in <study_dir>/figures/data/.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Publication style ──────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.framealpha": 0.85,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# Colour palette – distinguishable, colour-blind friendly
C_TENOR   = "#1B7FC4"   # blue
C_TOMCHUK = "#E05C1A"   # orange-red
C_PDI     = "#E05C1A"
C_PDI2    = "#C48B1B"
C_GRAY    = "#888888"

LINE_TENOR   = dict(color=C_TENOR,   marker="o", ls="-", lw=1.6)
LINE_TOMCHUK = dict(color=C_PDI,     marker="s", ls="--", lw=1.6)
LINE_PDI2    = dict(color=C_PDI2,    marker="^", ls=":", lw=1.4)

DIST_COLORS = {
    "Lognormal": "#1B7FC4",
    "Gaussian":  "#E05C1A",
    "Schulz":    "#27AE60",
    "Boltzmann": "#8E44AD",
}
DIST_MARKERS = {"Lognormal": "o", "Gaussian": "s", "Schulz": "^", "Boltzmann": "D"}

# Column shorthand
COL_TEN_ERR   = "tenor_abs_err_p"
COL_TOM_PDI   = "tomchuk_abs_err_p_pdi"
COL_TOM_PDI2  = "tomchuk_abs_err_p_pdi2"
COL_TEN_RG    = "tenor_rel_err_rg"
COL_TOM_RG    = "tomchuk_rel_err_rg"


# ── Helpers ────────────────────────────────────────────────────────────────

def _median_iqr(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return math.nan, math.nan, math.nan
    med = s.median()
    lo  = s.quantile(0.25)
    hi  = s.quantile(0.75)
    return med, lo, hi


def _save(fig, path_stem: Path, tight=True):
    """Save as PDF and PNG."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(str(path_stem) + ".pdf", bbox_inches="tight")
        fig.savefig(str(path_stem) + ".png", bbox_inches="tight")
    else:
        fig.savefig(str(path_stem) + ".pdf")
        fig.savefig(str(path_stem) + ".png")
    plt.close(fig)


def _add_legend_handles(ax, items):
    """items: list of (label, kwargs-for-Line2D-or-Patch)"""
    handles = []
    for label, kw in items:
        if "facecolor" in kw:
            handles.append(Patch(label=label, **kw))
        else:
            handles.append(Line2D([0], [0], label=label, **kw))
    ax.legend(handles=handles, framealpha=0.85)


def _annotate_winner(ax, x_vals, ten_med, tom_med, ymax):
    """Small arrows / text indicating which method wins at each x."""
    for xv, tm, tc in zip(x_vals, ten_med, tom_med):
        if math.isnan(tm) or math.isnan(tc):
            continue
        if tm < tc * 0.85:
            ax.annotate("T", (xv, tm), fontsize=5.5, color=C_TENOR,
                        ha="center", va="top", fontweight="bold")
        elif tc < tm * 0.85:
            ax.annotate("K", (xv, tc), fontsize=5.5, color=C_TOMCHUK,
                        ha="center", va="top", fontweight="bold")


# ── Figure 1: p-sweep ──────────────────────────────────────────────────────

def plot_fig1_p_sweep(df: pd.DataFrame, fig_dir: Path):
    sub = df[df["experiment"] == "exp1_p_sweep"].copy()
    if sub.empty:
        print("  [skip] Fig 1: no exp1_p_sweep data")
        return

    p_vals = sorted(sub["p_val"].unique())
    ten_med, ten_lo, ten_hi = [], [], []
    tom_pdi_med, tom_pdi_lo, tom_pdi_hi = [], [], []
    tom_pdi2_med, tom_pdi2_lo, tom_pdi2_hi = [], [], []

    for p in p_vals:
        s = sub[sub["p_val"] == p]
        m, lo, hi = _median_iqr(s[COL_TEN_ERR]);    ten_med.append(m);   ten_lo.append(lo);   ten_hi.append(hi)
        m, lo, hi = _median_iqr(s[COL_TOM_PDI]);    tom_pdi_med.append(m);tom_pdi_lo.append(lo);tom_pdi_hi.append(hi)
        m, lo, hi = _median_iqr(s[COL_TOM_PDI2]);   tom_pdi2_med.append(m);tom_pdi2_lo.append(lo);tom_pdi2_hi.append(hi)

    # Save source data
    src = pd.DataFrame({
        "p_input": p_vals,
        "tenor_median_abs_err_p": ten_med, "tenor_q25": ten_lo, "tenor_q75": ten_hi,
        "tomchuk_pdi_median": tom_pdi_med, "tomchuk_pdi_q25": tom_pdi_lo, "tomchuk_pdi_q75": tom_pdi_hi,
        "tomchuk_pdi2_median": tom_pdi2_med, "tomchuk_pdi2_q25": tom_pdi2_lo, "tomchuk_pdi2_q75": tom_pdi2_hi,
    })
    src.to_csv(fig_dir / "data" / "fig1_p_sweep.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4), constrained_layout=True)

    # Panel A – absolute error in p
    pv = np.array(p_vals, float)
    ax1.fill_between(pv, ten_lo, ten_hi, alpha=0.18, color=C_TENOR)
    ax1.fill_between(pv, tom_pdi_lo, tom_pdi_hi, alpha=0.18, color=C_TOMCHUK)
    ax1.fill_between(pv, tom_pdi2_lo, tom_pdi2_hi, alpha=0.12, color=C_PDI2)
    ax1.plot(pv, ten_med,     label="TENOR-SAXS",    **LINE_TENOR)
    ax1.plot(pv, tom_pdi_med, label="Tomchuk (PDI)", **LINE_TOMCHUK)
    ax1.plot(pv, tom_pdi2_med, label="Tomchuk (PDI2)", **LINE_PDI2)
    ax1.axhline(0.05, ls=":", lw=0.8, color="grey", label="5% error line")
    ax1.set_xlabel("Input polydispersity $p$")
    ax1.set_ylabel(r"Median $|p_\mathrm{rec} - p|$")
    ax1.set_title("(A) Absolute polydispersity error vs. $p$")
    ax1.legend()

    # Panel B – relative Rg error
    ten_rg_med, ten_rg_lo, ten_rg_hi = [], [], []
    tom_rg_med, tom_rg_lo, tom_rg_hi = [], [], []
    for p in p_vals:
        s = sub[sub["p_val"] == p]
        m, lo, hi = _median_iqr(s[COL_TEN_RG].abs()); ten_rg_med.append(m); ten_rg_lo.append(lo); ten_rg_hi.append(hi)
        m, lo, hi = _median_iqr(s[COL_TOM_RG].abs()); tom_rg_med.append(m); tom_rg_lo.append(lo); tom_rg_hi.append(hi)

    ax2.fill_between(pv, ten_rg_lo, ten_rg_hi, alpha=0.18, color=C_TENOR)
    ax2.fill_between(pv, tom_rg_lo, tom_rg_hi, alpha=0.18, color=C_TOMCHUK)
    ax2.plot(pv, ten_rg_med, label="TENOR-SAXS",    **LINE_TENOR)
    ax2.plot(pv, tom_rg_med, label="Tomchuk (PDI)", **LINE_TOMCHUK)
    ax2.set_xlabel("Input polydispersity $p$")
    ax2.set_ylabel(r"Median $|R_g^{\rm rec}/R_g^{\rm true} - 1|$")
    ax2.set_title(r"(B) Relative $R_g$ error vs. $p$")
    ax2.legend()

    _save(fig, fig_dir / "fig1_p_sweep")
    print("  [ok] Fig 1 saved")


# ── Figure 2: flux / photon count ──────────────────────────────────────────

def plot_fig2_flux(df: pd.DataFrame, fig_dir: Path):
    sub = df[df["experiment"] == "exp2_flux"].copy()
    if sub.empty:
        print("  [skip] Fig 2: no exp2_flux data"); return

    flux_exps = sorted(sub["flux_exp"].unique())
    ten_med, ten_lo, ten_hi = [], [], []
    tom_pdi_med, tom_pdi_lo, tom_pdi_hi = [], [], []

    for fe in flux_exps:
        s = sub[sub["flux_exp"] == fe]
        m, lo, hi = _median_iqr(s[COL_TEN_ERR]);  ten_med.append(m); ten_lo.append(lo); ten_hi.append(hi)
        m, lo, hi = _median_iqr(s[COL_TOM_PDI]);  tom_pdi_med.append(m); tom_pdi_lo.append(lo); tom_pdi_hi.append(hi)

    src = pd.DataFrame({
        "flux_exp": flux_exps,
        "tenor_median_abs_err_p": ten_med, "tenor_q25": ten_lo, "tenor_q75": ten_hi,
        "tomchuk_pdi_median": tom_pdi_med, "tomchuk_q25": tom_pdi_lo, "tomchuk_q75": tom_pdi_hi,
    })
    src.to_csv(fig_dir / "data" / "fig2_flux.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    fe = np.array(flux_exps, float)
    ax.fill_between(fe, ten_lo, ten_hi, alpha=0.18, color=C_TENOR)
    ax.fill_between(fe, tom_pdi_lo, tom_pdi_hi, alpha=0.18, color=C_TOMCHUK)
    ax.plot(fe, ten_med, label="TENOR-SAXS", **LINE_TENOR)
    ax.plot(fe, tom_pdi_med, label="Tomchuk (PDI)", **LINE_TOMCHUK)
    ax.axhline(0.05, ls=":", lw=0.8, color=C_GRAY)
    ax.set_xlabel(r"$\log_{10}$(forward photon count)")
    ax.set_ylabel(r"Median $|p_{\rm rec} - p|$")
    ax.set_title("(Fig.\u00a02) Flux / photon-count sensitivity")
    ax.legend()

    _save(fig, fig_dir / "fig2_flux")
    print("  [ok] Fig 2 saved")


# ── Figure 3: smearing ─────────────────────────────────────────────────────

def plot_fig3_smearing(df: pd.DataFrame, fig_dir: Path):
    sub = df[df["experiment"] == "exp3_smearing"].copy()
    if sub.empty:
        print("  [skip] Fig 3: no exp3_smearing data"); return

    smears = sorted(sub["smearing"].unique())
    ten_med, ten_lo, ten_hi = [], [], []
    tom_pdi_med, tom_pdi_lo, tom_pdi_hi = [], [], []
    tom_pdi2_med, _, _ = [], [], []

    for sm in smears:
        s = sub[sub["smearing"] == sm]
        m, lo, hi = _median_iqr(s[COL_TEN_ERR]);  ten_med.append(m); ten_lo.append(lo); ten_hi.append(hi)
        m, lo, hi = _median_iqr(s[COL_TOM_PDI]);  tom_pdi_med.append(m); tom_pdi_lo.append(lo); tom_pdi_hi.append(hi)
        m, lo, hi = _median_iqr(s[COL_TOM_PDI2]); tom_pdi2_med.append(m)
    # Fix slicing bug
    tom_pdi2_lo2, tom_pdi2_hi2 = [], []
    for sm in smears:
        s = sub[sub["smearing"] == sm]
        _, lo, hi = _median_iqr(s[COL_TOM_PDI2]); tom_pdi2_lo2.append(lo); tom_pdi2_hi2.append(hi)

    src = pd.DataFrame({
        "smearing_px": smears,
        "tenor_median": ten_med, "tenor_q25": ten_lo, "tenor_q75": ten_hi,
        "tomchuk_pdi_median": tom_pdi_med, "tomchuk_pdi_q25": tom_pdi_lo, "tomchuk_pdi_q75": tom_pdi_hi,
        "tomchuk_pdi2_median": tom_pdi2_med, "tomchuk_pdi2_q25": tom_pdi2_lo2, "tomchuk_pdi2_q75": tom_pdi2_hi2,
    })
    src.to_csv(fig_dir / "data" / "fig3_smearing.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    sm_arr = np.array(smears, float)
    ax.fill_between(sm_arr, ten_lo, ten_hi, alpha=0.18, color=C_TENOR)
    ax.fill_between(sm_arr, tom_pdi_lo, tom_pdi_hi, alpha=0.18, color=C_TOMCHUK)
    ax.fill_between(sm_arr, tom_pdi2_lo2, tom_pdi2_hi2, alpha=0.12, color=C_PDI2)
    ax.plot(sm_arr, ten_med, label="TENOR-SAXS", **LINE_TENOR)
    ax.plot(sm_arr, tom_pdi_med, label="Tomchuk (PDI)", **LINE_TOMCHUK)
    ax.plot(sm_arr, tom_pdi2_med, label="Tomchuk (PDI2)", **LINE_PDI2)
    ax.axhline(0.05, ls=":", lw=0.8, color=C_GRAY, label="5% target")
    ax.set_xlabel("Beam smearing (Gaussian $\\sigma$, pixels)")
    ax.set_ylabel(r"Median $|p_{\rm rec} - p|$")
    ax.set_title("(Fig.\u00a03) Smearing sensitivity")
    ax.legend()

    _save(fig, fig_dir / "fig3_smearing")
    print("  [ok] Fig 3 saved")


# ── Figure 4: distribution shape ──────────────────────────────────────────

def plot_fig4_distribution(df: pd.DataFrame, fig_dir: Path):
    sub = df[df["experiment"] == "exp4_distribution"].copy()
    if sub.empty:
        print("  [skip] Fig 4: no exp4_distribution data"); return

    dists = sub["dist_type"].unique()
    ten_vals, tom_pdi_vals, tom_pdi2_vals = {}, {}, {}
    for dist in dists:
        s = sub[sub["dist_type"] == dist]
        ten_vals[dist] = s[COL_TEN_ERR].dropna().values
        tom_pdi_vals[dist] = s[COL_TOM_PDI].dropna().values
        tom_pdi2_vals[dist] = s[COL_TOM_PDI2].dropna().values

    src_rows = []
    for dist in dists:
        for v in ten_vals.get(dist, []):
            src_rows.append({"dist_type": dist, "method": "TENOR-SAXS", "abs_err_p": v})
        for v in tom_pdi_vals.get(dist, []):
            src_rows.append({"dist_type": dist, "method": "Tomchuk-PDI", "abs_err_p": v})
        for v in tom_pdi2_vals.get(dist, []):
            src_rows.append({"dist_type": dist, "method": "Tomchuk-PDI2", "abs_err_p": v})
    pd.DataFrame(src_rows).to_csv(fig_dir / "data" / "fig4_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    x = np.arange(len(dists))
    w = 0.26
    for i, (dist, vals) in enumerate(ten_vals.items()):
        ax.bar(x[i] - w, np.median(vals) if len(vals) else 0,
               w, color=C_TENOR, alpha=0.85, edgecolor="k", lw=0.4)
        ax.errorbar(x[i] - w, np.median(vals) if len(vals) else 0,
                    yerr=np.std(vals) / np.sqrt(len(vals)) if len(vals) else 0,
                    fmt="none", color="k", lw=1, capsize=2)
    for i, (dist, vals) in enumerate(tom_pdi_vals.items()):
        ax.bar(x[i], np.median(vals) if len(vals) else 0,
               w, color=C_TOMCHUK, alpha=0.85, edgecolor="k", lw=0.4)
        ax.errorbar(x[i], np.median(vals) if len(vals) else 0,
                    yerr=np.std(vals) / np.sqrt(len(vals)) if len(vals) else 0,
                    fmt="none", color="k", lw=1, capsize=2)
    for i, (dist, vals) in enumerate(tom_pdi2_vals.items()):
        ax.bar(x[i] + w, np.median(vals) if len(vals) else 0,
               w, color=C_PDI2, alpha=0.85, edgecolor="k", lw=0.4)
        ax.errorbar(x[i] + w, np.median(vals) if len(vals) else 0,
                    yerr=np.std(vals) / np.sqrt(len(vals)) if len(vals) else 0,
                    fmt="none", color="k", lw=1, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(list(dists), rotation=15, ha="right")
    ax.set_ylabel(r"Median $|p_{\rm rec} - p|$")
    ax.set_title("(Fig.\u00a04) Distribution shape sensitivity")
    ax.axhline(0.05, ls=":", lw=0.8, color=C_GRAY)
    _add_legend_handles(ax, [
        ("TENOR-SAXS",     {"color": C_TENOR,   "lw": 5, "alpha": 0.85}),
        ("Tomchuk (PDI)",  {"color": C_TOMCHUK, "lw": 5, "alpha": 0.85}),
        ("Tomchuk (PDI2)", {"color": C_PDI2,    "lw": 5, "alpha": 0.85}),
    ])

    _save(fig, fig_dir / "fig4_distribution")
    print("  [ok] Fig 4 saved")


# ── Figure 5: PSF anisotropy ───────────────────────────────────────────────

def plot_fig5_anisotropy(df: pd.DataFrame, fig_dir: Path):
    sub = df[df["experiment"] == "exp5_anisotropy"].copy()
    if sub.empty:
        print("  [skip] Fig 5: no exp5_anisotropy data"); return

    sub["beam_label"] = sub.apply(
        lambda r: f"{r['smearing_x']:.0f}×{r['smearing_y']:.0f}", axis=1
    )
    labels = sub.groupby(["smearing_x", "smearing_y"])["beam_label"].first().values.tolist()
    labels_unique = list(dict.fromkeys(labels))  # preserve order, dedup

    ten_med_list, tom_med_list = [], []
    for label in labels_unique:
        s = sub[sub["beam_label"] == label]
        ten_med_list.append(_median_iqr(s[COL_TEN_ERR])[0])
        tom_med_list.append(_median_iqr(s[COL_TOM_PDI])[0])

    src = pd.DataFrame({
        "beam_label": labels_unique,
        "tenor_median_abs_err_p": ten_med_list,
        "tomchuk_median_abs_err_p": tom_med_list,
    })
    src.to_csv(fig_dir / "data" / "fig5_anisotropy.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.5, 3.0), constrained_layout=True)
    x = np.arange(len(labels_unique))
    w = 0.35
    ax.bar(x - w/2, ten_med_list, w, color=C_TENOR,   alpha=0.85,
           label="TENOR-SAXS", edgecolor="k", lw=0.4)
    ax.bar(x + w/2, tom_med_list, w, color=C_TOMCHUK, alpha=0.85,
           label="Tomchuk (PDI)", edgecolor="k", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\sigma_x{{=}}{l.split('×')[0]},\\sigma_y{{=}}{l.split('×')[1]}$"
                        for l in labels_unique], rotation=15, ha="right", fontsize=6.5)
    ax.set_ylabel(r"Median $|p_{\rm rec} - p|$")
    ax.set_title("(Fig.\u00a05) PSF anisotropy immunity")
    ax.axhline(0.05, ls=":", lw=0.8, color=C_GRAY)
    ax.legend()

    _save(fig, fig_dir / "fig5_anisotropy")
    print("  [ok] Fig 5 saved")


# ── Figure 6: joint p × flux heatmap ──────────────────────────────────────

def plot_fig6_joint(df: pd.DataFrame, fig_dir: Path):
    sub = df[df["experiment"] == "exp6_joint"].copy()
    if sub.empty:
        print("  [skip] Fig 6: no exp6_joint data"); return

    p_vals   = sorted(sub["p_val"].unique())
    flux_exps = sorted(sub["flux_exp"].unique())
    smearings = sorted(sub["smearing"].unique())

    src_rows = []
    for sm in smearings:
        s_sm = sub[sub["smearing"] == sm]
        for p in p_vals:
            for fe in flux_exps:
                s = s_sm[(s_sm["p_val"] == p) & (s_sm["flux_exp"] == fe)]
                t_med = _median_iqr(s[COL_TEN_ERR])[0]
                k_med = _median_iqr(s[COL_TOM_PDI])[0]
                src_rows.append({"smearing": sm, "p_val": p, "flux_exp": fe,
                                  "tenor_median": t_med, "tomchuk_median": k_med})
    pd.DataFrame(src_rows).to_csv(fig_dir / "data" / "fig6_joint.csv", index=False)

    src_df = pd.DataFrame(src_rows)
    n_sm = len(smearings)
    fig, axes = plt.subplots(2, n_sm, figsize=(3.8 * n_sm, 6.0), constrained_layout=True)
    if n_sm == 1:
        axes = axes[:, None]

    vmax = 0.6
    cmap = "RdYlGn_r"

    for col_idx, sm in enumerate(smearings):
        s_sm = src_df[src_df["smearing"] == sm]

        for row_idx, (col_key, title_method) in enumerate([
            ("tenor_median", "TENOR-SAXS"),
            ("tomchuk_median", "Tomchuk (PDI)"),
        ]):
            ax = axes[row_idx][col_idx]
            mat = np.full((len(p_vals), len(flux_exps)), math.nan)
            for i, p in enumerate(p_vals):
                for j, fe in enumerate(flux_exps):
                    sub_cell = s_sm[(s_sm["p_val"] == p) & (s_sm["flux_exp"] == fe)]
                    if not sub_cell.empty:
                        mat[i, j] = sub_cell[col_key].values[0]
            im = ax.imshow(mat, aspect="auto", origin="lower",
                           vmin=0, vmax=vmax, cmap=cmap)
            ax.set_xticks(range(len(flux_exps)))
            ax.set_xticklabels([f"$10^{{{fe}}}$" for fe in flux_exps], fontsize=6)
            ax.set_yticks(range(len(p_vals)))
            ax.set_yticklabels([f"{p:.2f}" for p in p_vals], fontsize=6)
            ax.set_xlabel("Flux (photons)")
            ax.set_ylabel("Polydispersity $p$")
            ax.set_title(f"{title_method}\n$\\sigma={sm:.0f}$ px", fontsize=8)
            fig.colorbar(im, ax=ax, label=r"$|p_{\rm rec}-p|$", fraction=0.046, pad=0.04)

    fig.suptitle("(Fig.\u00a06) Joint $p \\times$ flux performance map", fontsize=9)
    _save(fig, fig_dir / "fig6_joint_heatmap")
    print("  [ok] Fig 6 saved")


# ── Figure 7: Rg pilot ─────────────────────────────────────────────────────

def plot_fig7_rg_pilot(df: pd.DataFrame, fig_dir: Path):
    sub = df[df["experiment"] == "pilot_rg"].copy()
    if sub.empty:
        print("  [skip] Fig 7 (Rg pilot): no pilot_rg data"); return

    rg_vals = sorted(sub["mean_rg"].unique())
    ten_med, tom_med = [], []
    for rg in rg_vals:
        s = sub[sub["mean_rg"] == rg]
        ten_med.append(_median_iqr(s[COL_TEN_ERR])[0])
        tom_med.append(_median_iqr(s[COL_TOM_PDI])[0])

    src = pd.DataFrame({"mean_rg": rg_vals, "tenor_median": ten_med, "tomchuk_median": tom_med})
    src.to_csv(fig_dir / "data" / "fig7_rg_pilot.csv", index=False)

    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    ax.plot(rg_vals, ten_med, label="TENOR-SAXS",    **LINE_TENOR)
    ax.plot(rg_vals, tom_med, label="Tomchuk (PDI)", **LINE_TOMCHUK)
    ax.axhline(0.05, ls=":", lw=0.8, color=C_GRAY)
    ax.set_xlabel(r"Mean $R_g$ (nm)")
    ax.set_ylabel(r"Median $|p_{\rm rec} - p|$")
    ax.set_title(r"(Pilot) $R_g$-dependence check")
    ax.legend()
    _save(fig, fig_dir / "fig7_rg_pilot")
    rel_change = abs(np.nanmax(ten_med) - np.nanmin(ten_med)) / (abs(np.nanmean(ten_med)) + 1e-9)
    note = "NEGLIGIBLE (<10%)" if rel_change < 0.10 else f"SIGNIFICANT ({rel_change*100:.0f}%)"
    print(f"  [ok] Fig 7 (Rg pilot) saved – Rg effect on TENOR: {note}")
    return rel_change


# ── Summary table figure ───────────────────────────────────────────────────

def plot_summary_table(df: pd.DataFrame, fig_dir: Path):
    """Render a publication-quality summary table as a figure."""
    rows = []
    EXP_LABELS = {
        "exp1_p_sweep": "p-sweep (Fig. 1)",
        "exp2_flux":    "Flux sensitivity (Fig. 2)",
        "exp3_smearing": "Smearing (Fig. 3)",
        "exp4_distribution": "Distribution shape (Fig. 4)",
        "exp5_anisotropy": "PSF anisotropy (Fig. 5)",
        "pilot_rg": r"$R_g$ dependence (pilot)",
    }
    for exp_key, exp_label in EXP_LABELS.items():
        s = df[df["experiment"] == exp_key]
        if s.empty:
            continue
        t_med = s[COL_TEN_ERR].median()
        k_med = s[COL_TOM_PDI].median()
        winner = "TENOR" if t_med < k_med * 0.9 else ("Tomchuk" if k_med < t_med * 0.9 else "Tie")
        rows.append((exp_label, f"{t_med:.3f}", f"{k_med:.3f}", winner))

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(7.0, 0.5 + 0.45 * len(rows)), constrained_layout=True)
    ax.axis("off")
    col_labels = ["Experiment", "TENOR-SAXS\nmedian |Δp|", "Tomchuk (PDI)\nmedian |Δp|", "Winner"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    # Colour header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i, (_, _, _, winner) in enumerate(rows, 1):
        if winner == "TENOR":
            table[i, 3].set_facecolor("#D6EAF8")
        elif winner == "Tomchuk":
            table[i, 3].set_facecolor("#FDEBD0")

    ax.set_title("Summary: TENOR-SAXS vs. Tomchuk", fontsize=9, pad=12)
    pd.DataFrame(rows, columns=col_labels).to_csv(
        fig_dir / "data" / "summary_table.csv", index=False)
    _save(fig, fig_dir / "fig0_summary_table")
    print("  [ok] Summary table saved")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot comparison figures")
    parser.add_argument("--study-dir", required=True,
                        help="Path to the study output directory containing summary_results.csv")
    args = parser.parse_args()

    root = Path(args.study_dir)
    csv_path = root / "summary_results.csv"
    if not csv_path.exists():
        # also check checkpoint
        csv_path = root / "progress_checkpoint.csv"
    if not csv_path.exists():
        print(f"No summary_results.csv or progress_checkpoint.csv found in {root}")
        return

    fig_dir = root / "figures"
    (fig_dir / "data").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Experiments: {df['experiment'].value_counts().to_dict()}")

    plot_fig1_p_sweep(df, fig_dir)
    plot_fig2_flux(df, fig_dir)
    plot_fig3_smearing(df, fig_dir)
    plot_fig4_distribution(df, fig_dir)
    plot_fig5_anisotropy(df, fig_dir)
    plot_fig6_joint(df, fig_dir)
    plot_fig8_p_synchrotron(df, fig_dir)
    plot_fig9_p_home(df, fig_dir)
    plot_fig10_flux_synchrotron(df, fig_dir)
    rg_effect = plot_fig7_rg_pilot(df, fig_dir)
    plot_summary_table(df, fig_dir)

    # Record whether Rg matters
    rg_note = {}
    if rg_effect is not None:
        rg_note = {"rg_relative_effect": float(rg_effect),
                   "rg_sweep_recommended": bool(rg_effect > 0.10)}
    (root / "rg_pilot_conclusion.json").write_text(json.dumps(rg_note, indent=2))

    print(f"\nAll figures saved to: {fig_dir}")




def plot_fig8_p_synchrotron(df: pd.DataFrame, fig_dir: Path):
    sub = df[df["experiment"] == "exp7_p_synchrotron"].copy()
    if sub.empty: return
    p_vals = sorted(sub["p_val"].unique())
    ten_med, tom_med = [], []
    for p in p_vals:
        s = sub[sub["p_val"] == p]
        ten_med.append(_median_iqr(s[COL_TEN_ERR])[0])
        tom_med.append(_median_iqr(s[COL_TOM_PDI])[0])
    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    ax.plot(p_vals, ten_med, label="TENOR", **LINE_TENOR)
    ax.plot(p_vals, tom_med, label="Tomchuk", **LINE_TOMCHUK)
    ax.axhline(0.05, ls=":", lw=0.8, color=C_GRAY)
    ax.set_xlabel("Input polydispersity $p$")
    ax.set_ylabel(r"Median $|p_{\rm rec} - p|$")
    ax.set_title("Synchrotron Smearing (3x15)\nAbsolute error vs. $p$")
    ax.legend(); _save(fig, fig_dir / "fig8_p_synchrotron")
    print("  [ok] Fig 8 saved")

def plot_fig9_p_home(df: pd.DataFrame, fig_dir: Path):
    sub = df[df["experiment"] == "exp8_p_home"].copy()
    if sub.empty: return
    p_vals = sorted(sub["p_val"].unique())
    ten_med, tom_med = [], []
    for p in p_vals:
        s = sub[sub["p_val"] == p]
        ten_med.append(_median_iqr(s[COL_TEN_ERR])[0])
        tom_med.append(_median_iqr(s[COL_TOM_PDI])[0])
    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    ax.plot(p_vals, ten_med, label="TENOR", **LINE_TENOR)
    ax.plot(p_vals, tom_med, label="Tomchuk", **LINE_TOMCHUK)
    ax.axhline(0.05, ls=":", lw=0.8, color=C_GRAY)
    ax.set_xlabel("Input polydispersity $p$")
    ax.set_ylabel(r"Median $|p_{\rm rec} - p|$")
    ax.set_title("Home Source Smearing (20x20)\nAbsolute error vs. $p$")
    ax.legend(); _save(fig, fig_dir / "fig9_p_home")
    print("  [ok] Fig 9 saved")

def plot_fig10_flux_synchrotron(df: pd.DataFrame, fig_dir: Path):
    sub = df[df["experiment"] == "exp9_flux_synchrotron"].copy()
    if sub.empty: return
    flux_exps = sorted(sub["flux_exp"].unique())
    ten_med, tom_med = [], []
    for fe in flux_exps:
        s = sub[sub["flux_exp"] == fe]
        ten_med.append(_median_iqr(s[COL_TEN_ERR])[0])
        tom_med.append(_median_iqr(s[COL_TOM_PDI])[0])
    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)
    ax.plot(flux_exps, ten_med, label="TENOR", **LINE_TENOR)
    ax.plot(flux_exps, tom_med, label="Tomchuk", **LINE_TOMCHUK)
    ax.axhline(0.05, ls=":", lw=0.8, color=C_GRAY)
    ax.set_xlabel("log10(Flux)")
    ax.set_ylabel(r"Median $|p_{\rm rec} - p|$")
    ax.set_title("Synchrotron Smearing (3x15)\nFlux sensitivity")
    ax.legend(); _save(fig, fig_dir / "fig10_flux_synchrotron")
    print("  [ok] Fig 10 saved")

if __name__ == "__main__":
    main()


