import argparse
import json
from pathlib import Path

import numpy as np

from analysis_utils import perform_saxs_analysis


SUPPORTED_DISTRIBUTIONS = (
    "Gaussian",
    "Lognormal",
    "Schulz",
    "Boltzmann",
    "Triangular",
    "Uniform",
)


def _clean_profile(q, intensity):
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    i_arr = np.asarray(intensity, dtype=float).reshape(-1)
    if q_arr.size != i_arr.size:
        raise ValueError("q and I must have the same length.")
    if q_arr.size < 5:
        raise ValueError("At least 5 q/I points are required.")

    mask = np.isfinite(q_arr) & np.isfinite(i_arr) & (q_arr > 0) & (i_arr > 0)
    q_arr = q_arr[mask]
    i_arr = i_arr[mask]
    if q_arr.size < 5:
        raise ValueError("Need at least 5 finite points with q > 0 and I > 0.")

    order = np.argsort(q_arr)
    q_arr = q_arr[order]
    i_arr = i_arr[order]
    return q_arr, i_arr


def _default_initial_rg_guess(q):
    q_arr = np.asarray(q, dtype=float)
    q_pos = q_arr[np.isfinite(q_arr) & (q_arr > 0)]
    if q_pos.size == 0:
        raise ValueError("Cannot derive an initial Rg guess from empty q data.")

    q_ref = np.percentile(q_pos, 20)
    q_ref = max(float(q_ref), 1e-6)
    return 0.8 / q_ref


def _best_variant(result):
    rrms_pdi = result.get("rrms_pdi", np.inf)
    rrms_pdi2 = result.get("rrms_pdi2", np.inf)
    return "PDI2" if rrms_pdi2 < rrms_pdi else "PDI"


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def analyze_tomchuk(
    q,
    intensity,
    dist_type,
    initial_rg_guess=None,
    nnls_max_rg=None,
    return_full=False,
):
    if dist_type not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"Unsupported distribution '{dist_type}'. Supported values: {', '.join(SUPPORTED_DISTRIBUTIONS)}"
        )

    q_arr, i_arr = _clean_profile(q, intensity)
    if initial_rg_guess is None:
        initial_rg_guess = _default_initial_rg_guess(q_arr)
    initial_rg_guess = max(float(initial_rg_guess), 1e-6)

    if nnls_max_rg is None:
        nnls_max_rg = max(10.0 * initial_rg_guess, 5.0)

    result = perform_saxs_analysis(
        q_exp=q_arr,
        i_exp=i_arr,
        dist_type=dist_type,
        initial_rg_guess=initial_rg_guess,
        mode="Sphere",
        method="Tomchuk",
        max_rg_nnls=float(nnls_max_rg),
        analysis_settings={},
    )

    best_variant = _best_variant(result)
    best_variant_key = best_variant.lower()
    summary = {
        "distribution": dist_type,
        "initial_rg_guess": initial_rg_guess,
        "tomchuk_extraction": result.get("tomchuk_extraction"),
        "best_variant": best_variant,
        "Rg_scattering": result.get("Rg"),
        "Rg_guinier": result.get("Rg_guinier"),
        "G": result.get("G"),
        "B": result.get("B"),
        "Q": result.get("Q"),
        "lc": result.get("lc"),
        "PDI": result.get("PDI"),
        "PDI2": result.get("PDI2"),
        "p_from_pdi": result.get("p_rec_pdi"),
        "p_from_pdi2": result.get("p_rec_pdi2"),
        "mean_radius_from_pdi": result.get("mean_r_rec_pdi"),
        "mean_radius_from_pdi2": result.get("mean_r_rec_pdi2"),
        "mean_rg_from_pdi": result.get("rg_num_rec_pdi"),
        "mean_rg_from_pdi2": result.get("rg_num_rec_pdi2"),
        "rrms_pdi": result.get("rrms_pdi"),
        "rrms_pdi2": result.get("rrms_pdi2"),
        "reported_p": result.get(f"p_rec_{best_variant_key}"),
        "reported_mean_radius": result.get(f"mean_r_rec_{best_variant_key}"),
        "reported_mean_rg": result.get(f"rg_num_rec_{best_variant_key}"),
    }
    summary["Rg"] = summary["Rg_scattering"]

    if return_full:
        summary["full_result"] = result

    return _to_builtin(summary)


def _parse_array_text(text):
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty array input.")
    try:
        parsed = json.loads(stripped)
        return np.asarray(parsed, dtype=float)
    except json.JSONDecodeError:
        return np.fromstring(stripped.replace(",", " "), sep=" ", dtype=float)


def _load_profile_file(path):
    data = np.loadtxt(path, comments="#", delimiter=None)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError("Input file must contain at least two columns: q and I.")
    return data[:, 0], data[:, 1]


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Standalone Tomchuk analysis for 1D SAXS sphere profiles."
    )
    parser.add_argument(
        "--distribution",
        required=True,
        choices=SUPPORTED_DISTRIBUTIONS,
        help="Declared size-distribution family.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a text file with at least two columns: q and I.",
    )
    parser.add_argument(
        "--q",
        help="Inline q array as JSON or comma/space-separated numbers.",
    )
    parser.add_argument(
        "--i",
        help="Inline intensity array as JSON or comma/space-separated numbers.",
    )
    parser.add_argument(
        "--initial-rg",
        type=float,
        default=None,
        help="Optional initial guess for the scattering Rg.",
    )
    parser.add_argument(
        "--nnls-max-rg",
        type=float,
        default=None,
        help="Optional upper scale used by the shared analysis wrapper.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include the full raw result dictionary in the JSON output.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level.",
    )
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.input is not None:
        q_vals, i_vals = _load_profile_file(args.input)
    else:
        if args.q is None or args.i is None:
            parser.error("Provide either --input or both --q and --i.")
        q_vals = _parse_array_text(args.q)
        i_vals = _parse_array_text(args.i)

    result = analyze_tomchuk(
        q=q_vals,
        intensity=i_vals,
        dist_type=args.distribution,
        initial_rg_guess=args.initial_rg,
        nnls_max_rg=args.nnls_max_rg,
        return_full=args.full,
    )
    print(json.dumps(result, indent=args.indent, sort_keys=True))


if __name__ == "__main__":
    main()
