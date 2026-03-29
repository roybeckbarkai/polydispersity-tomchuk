import argparse
import sys

import numpy as np
import pandas as pd

from analysis_utils import (
    run_simulation_analysis_case,
    build_summary_row,
    build_sanity_summary_row,
)


DEFAULT_DISTRIBUTIONS = ["Gaussian"]
DEFAULT_P_VALUES = [0.7, 1.0]


def run_case(dist_type, p_val, mean_rg, q_max, n_bins, flux, noise, pixels, smearing, seed):
    if noise:
        np.random.seed(seed)

    params = {
        "mean_rg": mean_rg,
        "p_val": p_val,
        "dist_type": dist_type,
        "mode": "Sphere",
        "pixels": pixels,
        "q_min": 0.0,
        "q_max": q_max,
        "n_bins": n_bins,
        "smearing": smearing,
        "flux": flux,
        "noise": noise,
        "binning_mode": "Logarithmic",
        "method": "Tomchuk",
        "nnls_max_rg": mean_rg * (1 + 8 * p_val),
    }

    q_sim, i_sim, r_vals, pdf_vals, res, _ = run_simulation_analysis_case(params)
    summary = build_summary_row(params, res)
    sanity_summary = build_sanity_summary_row(q_sim, i_sim, r_vals, pdf_vals, res)
    return {
        "distribution": dist_type,
        "p_input": p_val,
        "p_pdi": res.get("p_rec_pdi", 0.0),
        "p_pdi2": res.get("p_rec_pdi2", 0.0),
        "rel_err_pdi": summary.get("Rel_Err_p", 0.0),
        "rel_err_pdi2": summary.get("Rel_Err_p_PDI2", 0.0),
        "mean_radius_pdi": res.get("mean_r_rec_pdi", 0.0),
        "mean_radius_pdi2": res.get("mean_r_rec_pdi2", 0.0),
        "PDI": res.get("PDI", 0.0),
        "PDI2": res.get("PDI2", 0.0),
        "chi2_pdi": res.get("chi2_pdi", 0.0),
        "chi2_pdi2": res.get("chi2_pdi2", 0.0),
        "sanity_failures": sanity_summary["Sanity_Failures"],
        "sanity_pass": sanity_summary["Sanity_Pass"],
        "sanity_suggestions": sanity_summary["Sanity_Suggestions"],
    } | {k: v for k, v in sanity_summary.items() if k.startswith("Sanity_RelErr_")}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Tomchuk recovery on simulated polydisperse spheres.")
    parser.add_argument("--distributions", nargs="+", default=DEFAULT_DISTRIBUTIONS)
    parser.add_argument("--p-values", nargs="+", type=float, default=DEFAULT_P_VALUES)
    parser.add_argument("--mean-rg", type=float, default=4.0)
    parser.add_argument("--q-max", type=float, default=0.8)
    parser.add_argument("--n-bins", type=int, default=256)
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument("--flux", type=float, default=1e12)
    parser.add_argument("--smearing", type=float, default=0.0)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-rel-error", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for dist_type in args.distributions:
        for idx, p_val in enumerate(args.p_values):
            rows.append(
                run_case(
                    dist_type=dist_type,
                    p_val=p_val,
                    mean_rg=args.mean_rg,
                    q_max=args.q_max,
                    n_bins=args.n_bins,
                    flux=args.flux,
                    noise=args.noise,
                    pixels=args.pixels,
                    smearing=args.smearing,
                    seed=args.seed + idx,
                )
            )

    df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    max_err_pdi = float(df["rel_err_pdi"].abs().max())
    max_err_pdi2 = float(df["rel_err_pdi2"].abs().max())
    print(f"\nmax |relative error| PDI : {max_err_pdi:.4f}")
    print(f"max |relative error| PDI2: {max_err_pdi2:.4f}")

    failing_sanity = df.loc[~df["sanity_pass"]]
    if len(failing_sanity) > 0:
        print("\nSanity-check warnings:")
        for _, row in failing_sanity.iterrows():
            print(
                f"- {row['distribution']} p={row['p_input']:.4f}: failed [{row['sanity_failures']}] -> {row['sanity_suggestions']}"
            )

    if max(max_err_pdi, max_err_pdi2) > args.max_rel_error:
        print(
            f"\nFAIL: at least one recovery error exceeded the threshold of {args.max_rel_error:.4f}.",
            file=sys.stderr,
        )
        return 1

    print("\nPASS: all PDI and PDI2 recovery errors are within threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
