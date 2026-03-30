from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_utils import calculate_indices_from_p, solve_p_tomchuk


OUT_DIR = ROOT / "studies" / "tomchuk_benchmark_20260329" / "figures"
CURVE_CSV = ROOT / "studies" / "tomchuk_benchmark_20260329" / "tables" / "tomchuk_pdi_curves.csv"

DISTS = [
    ("Gaussian", "Gaussian"),
    ("Lognormal", "Lognormal"),
    ("Schulz", "Schulz"),
    ("Boltzmann", "Boltzmann"),
    ("Triangular", "Triangular"),
    ("Uniform", "Uniform"),
]

P_MAX_BY_DIST = {
    "Gaussian": 1.5,
    "Lognormal": 1.5,
    "Schulz": 1.5,
    "Boltzmann": 1.5,
    # For bounded-support distributions, p is limited by the requirement that
    # the lower edge of the support remains non-negative:
    #   Triangular: 1 - sqrt(6) p >= 0  -> p <= 1/sqrt(6)
    #   Uniform:    1 - sqrt(3) p >= 0  -> p <= 1/sqrt(3)
    "Triangular": (1.0 / np.sqrt(6.0)) - 1e-4,
    "Uniform": (1.0 / np.sqrt(3.0)) - 1e-4,
}


def build_curve_df() -> pd.DataFrame:
    rows = []
    for dist_type, label in DISTS:
        p_grid = np.linspace(0.01, P_MAX_BY_DIST[dist_type], 600)
        for p in p_grid:
            pdi, pdi2 = calculate_indices_from_p(float(p), dist_type)
            p_rec_pdi = solve_p_tomchuk(pdi, "PDI", dist_type)
            p_rec_pdi2 = solve_p_tomchuk(pdi2, "PDI2", dist_type)
            rows.append(
                {
                    "distribution": label,
                    "p": float(p),
                    "PDI": float(pdi),
                    "PDI2": float(pdi2),
                    "p_rec_from_PDI": float(p_rec_pdi),
                    "p_rec_from_PDI2": float(p_rec_pdi2),
                    "abs_err_pdi_inverse": abs(float(p_rec_pdi) - float(p)),
                    "abs_err_pdi2_inverse": abs(float(p_rec_pdi2) - float(p)),
                    "p_max_used": float(P_MAX_BY_DIST[dist_type]),
                }
            )
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    specs = [
        ("PDI", "Reconstruction of Tomchuk Fig. 4", axes[0], (1.0, 13.0)),
        ("PDI2", "Reconstruction of Tomchuk Fig. 5a", axes[1], (1.0, 2.0)),
    ]
    for metric, title, ax, xlim in specs:
        for dist_type, label in DISTS:
            sub = df[df["distribution"] == label]
            ax.plot(sub[metric], sub["p"], linewidth=1.8, label=label)
        # Paper marks p < 0.25 as a practical low-sensitivity region.
        ax.axhspan(0.0, 0.25, color="0.85", alpha=0.8, hatch="//", edgecolor="0.6")
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel("p")
        ax.set_xlim(*xlim)
        ax.set_ylim(0.0, 1.5)
    axes[0].legend(fontsize=8, ncol=2)
    out = OUT_DIR / "tomchuk_p_vs_pdi_curves_paper_ranges.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    df = build_curve_df()
    CURVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CURVE_CSV, index=False)
    out = make_plot(df)

    summary = (
        df.groupby("distribution")
        .agg(
            max_abs_err_pdi_inverse=("abs_err_pdi_inverse", "max"),
            max_abs_err_pdi2_inverse=("abs_err_pdi2_inverse", "max"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(f"Saved curves: {CURVE_CSV}")
    print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
