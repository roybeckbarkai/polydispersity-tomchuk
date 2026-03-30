"""Validate the standalone TENOR-SAXS implementation on simulated 2D cases."""

from __future__ import annotations

import argparse

import pandas as pd

from sim_utils import run_simulation_core
from tenor_saxs import analyze_tenor_saxs_2d


def run_case(case):
    params = {
        "mean_rg": case["mean_rg"],
        "p_val": case["p_val"],
        "dist_type": case["dist_type"],
        "mode": "Sphere",
        "pixels": case["pixels"],
        "q_min": 0.0,
        "q_max": case["q_max"],
        "n_bins": case["n_bins"],
        "smearing_x": case["smearing_x"],
        "smearing_y": case["smearing_y"],
        "flux": case["flux"],
        "noise": case["noise"],
        "binning_mode": "Logarithmic",
        "radius_samples": case.get("radius_samples", 400),
        "q_samples": case.get("q_samples", 200),
    }
    _, _, i_2d, _, _ = run_simulation_core(params)
    tenor = analyze_tenor_saxs_2d(
        i_2d=i_2d,
        q_max=case["q_max"],
        dist_type=case["dist_type"],
        initial_rg_guess=case["mean_rg"],
    )
    return {
        "label": case["label"],
        "dist_type": case["dist_type"],
        "noise": case["noise"],
        "smearing_x": case["smearing_x"],
        "smearing_y": case["smearing_y"],
        "flux": case["flux"],
        "mean_rg_true": case["mean_rg"],
        "mean_rg_rec": tenor["mean_rg_rec"],
        "p_true": case["p_val"],
        "p_rec": tenor["p_rec"],
        "weighted_v_rec": tenor["weighted_v"],
        "raw_g1_over_g0": tenor["observable_raw_g1_over_g0"],
        "rg_app": tenor["rg_app"],
        "rel_err_rg": (tenor["mean_rg_rec"] - case["mean_rg"]) / case["mean_rg"],
        "rel_err_p": (tenor["p_rec"] - case["p_val"]) / case["p_val"],
        "candidate_count": tenor["candidate_count"],
        "g_rmse": tenor["best_g_rmse"],
        "m_rmse": tenor["best_m_rmse"],
    }


def default_cases():
    return [
        {
            "label": "gaussian_p010_clean",
            "mean_rg": 4.0,
            "p_val": 0.10,
            "dist_type": "Gaussian",
            "pixels": 384,
            "q_max": 2.0,
            "n_bins": 256,
            "smearing_x": 0.0,
            "smearing_y": 0.0,
            "flux": 1e8,
            "noise": False,
        },
        {
            "label": "gaussian_p030_clean",
            "mean_rg": 4.0,
            "p_val": 0.30,
            "dist_type": "Gaussian",
            "pixels": 384,
            "q_max": 2.0,
            "n_bins": 256,
            "smearing_x": 0.0,
            "smearing_y": 0.0,
            "flux": 1e8,
            "noise": False,
        },
        {
            "label": "lognormal_p030_clean",
            "mean_rg": 4.0,
            "p_val": 0.30,
            "dist_type": "Lognormal",
            "pixels": 384,
            "q_max": 2.0,
            "n_bins": 256,
            "smearing_x": 0.0,
            "smearing_y": 0.0,
            "flux": 1e8,
            "noise": False,
        },
        {
            "label": "gaussian_p030_noisy",
            "mean_rg": 4.0,
            "p_val": 0.30,
            "dist_type": "Gaussian",
            "pixels": 384,
            "q_max": 2.0,
            "n_bins": 256,
            "smearing_x": 0.0,
            "smearing_y": 0.0,
            "flux": 1e9,
            "noise": True,
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="", help="Optional path to save the summary table.")
    args = parser.parse_args()

    rows = [run_case(case) for case in default_cases()]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved CSV summary to {args.csv}")


if __name__ == "__main__":
    main()
