"""
tenor_tomchuk_comparison_study.py
==================================
Publication-quality comparative study: TENOR-SAXS vs. Tomchuk method.

Runs 6 independent experiments on identical simulated 2-D SAXS data, applying
both analysis methods to every realization.  All parameters, raw data and
analysis outcomes are saved for later re-plotting.

Usage:
    python tenor_tomchuk_comparison_study.py [--fast] [--workers N] [--output-root DIR]

--fast   : reduced grid (N_rep=3, fewer p/flux/smearing points) for smoke testing
--workers: number of parallel processes (default: cpu_count - 1)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import hashlib
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ── Experiment grids
# ---------------------------------------------------------------------------

# Distributions used in the study (Triangular/Uniform excluded by design)
DISTRIBUTIONS = ["Lognormal", "Gaussian", "Schulz", "Boltzmann"]

# Baseline parameters (from app_settings.json)
BASELINE = dict(
    mean_rg=3.0983866769659336,  # nm, from app_settings
    pixels=1500,
    n_bins=1000,
    q_min=0.0,
    binning_mode="Logarithmic",
    pixel_size_um=70.0,
    sample_detector_distance_cm=360.0,
    wavelength_nm=0.1,
    smearing_x=3.0,
    smearing_y=3.0,
    flux_exp=8,          # 1e8
    dist_type="Lognormal",
    form_factor_model="Exact Sphere",
    ensemble_sampling="Continuous",
    radius_samples=2000,
    q_samples=2000,
    ensemble_members=41,
    # TENOR settings
    tenor_psf_sigma_x_start=0.5,
    tenor_psf_sigma_y_start=0.6,
    tenor_psf_sigma_step=0.4,
    tenor_psf_count=20,
    tenor_psf_secondary_ratio=0.5,
    tenor_psf_truncate=10.0,
    tenor_qrg_limit=0.8,
    tenor_radial_bins=10,
    tenor_reconstruction_trials=5,
    tenor_calibration_p_count=15,
    tenor_calibration_p_min=0.01,
    tenor_calibration_p_max=1.0,
    tenor_use_g3=False,
    tenor_use_m3=False,
    # Tomchuk target tolerance
    tomchuk_target_abs_error=0.001,
)

# Full grids
P_GRID_FULL = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]
FLUX_GRID_FULL = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
SMEAR_GRID_FULL = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0]
ANISO_BEAMS_FULL = [
    (3.0,  3.0),
    (3.0,  5.0),
    (3.0,  8.0),
    (3.0, 12.0),
    (3.0, 15.0),
    (3.0, 20.0),
    (3.0, 25.0),
    (3.0, 30.0),
    (15.0, 3.0),
    (20.0, 20.0),
]
RG_GRID_PILOT = [2.0, 3.1, 6.0]   # for Rg-dependence pilot

# Fast / reduced grids
P_GRID_FAST = [0.10, 0.30, 0.60]
FLUX_GRID_FAST = [7, 8, 9]
SMEAR_GRID_FAST = [1, 3, 8]
ANISO_BEAMS_FAST = [(3.0, 3.0), (3.0, 15.0), (1.0, 20.0)]
DIST_FAST = ["Lognormal", "Schulz"]

N_REP_FULL = 10
N_REP_FAST = 3
NNLS_FACTOR = 8.0
DEFAULT_LARGE_DATA_ROOT = Path("/Users/roybeck/Library/CloudStorage/Dropbox/python code copy/tomchuk study files")
CACHE_SCHEMA_VERSION = "v2_seeded_2d_rgmatch"


# ---------------------------------------------------------------------------
# ── Helpers
# ---------------------------------------------------------------------------

def base_sim_params(overrides: dict) -> dict:
    """Merge baseline with experiment-specific overrides."""
    p = dict(BASELINE)
    p.update(overrides)
    p["flux"] = 10 ** p.get("flux_exp", 8)
    p["mode"] = "Sphere"
    p["q_max"] = _q_max_from_geometry(p)
    p["smearing"] = 0.5 * (p["smearing_x"] + p["smearing_y"])
    p["noise"] = True
    p["nnls_max_rg"] = p["mean_rg"] * (1 + NNLS_FACTOR * p.get("p_val", 0.3))
    p["method"] = "Tomchuk"
    return p


def make_large_file_layout(study_name: str, large_data_root: str | os.PathLike[str]) -> dict:
    root = Path(large_data_root).expanduser().resolve()
    study_root = root / study_name
    return {
        "root": root,
        "study_root": study_root,
        "cache_dir": root / "cache",
        "data_dir": study_root / "data",
    }


def _q_max_from_geometry(p: dict) -> float:
    from sim_utils import get_detector_q_max
    return get_detector_q_max(
        pixels=p["pixels"],
        pixel_size_um=p.get("pixel_size_um", 70.0),
        sample_detector_distance_cm=p.get("sample_detector_distance_cm", 360.0),
        wavelength_nm=p.get("wavelength_nm", 0.1),
    )


def _build_tenor_calibration_grid(p: dict) -> np.ndarray:
    """Build calibration p-grid for TENOR from settings."""
    n = int(p.get("tenor_calibration_p_count", 15))
    lo = float(p.get("tenor_calibration_p_min", 0.01))
    hi = float(p.get("tenor_calibration_p_max", 1.0))
    return np.concatenate([
        np.linspace(lo, 0.12, max(n // 3, 4)),
        np.linspace(0.15, hi, max(n - n // 3, 4)),
    ])


def _build_tenor_psf_pairs(p: dict):
    from tenor_saxs import build_default_psf_pairs
    return build_default_psf_pairs(
        sigma_x_start=p.get("tenor_psf_sigma_x_start", 0.5),
        sigma_y_start=p.get("tenor_psf_sigma_y_start", 0.6),
        sigma_step=p.get("tenor_psf_sigma_step", 0.4),
        pair_count=int(p.get("tenor_psf_count", 20)),
        secondary_ratio=p.get("tenor_psf_secondary_ratio", 0.5),
    )


def safe_rel_err(obs, true):
    if true == 0:
        return math.nan
    return (obs - true) / abs(true)


def safe_abs_err(obs, true):
    return abs(obs - true)


def apply_tenor_overrides(case: dict, ridge_lambda_scale: float, match_apparent_rg) -> dict:
    updated = dict(case)
    updated["tenor_ridge_lambda_scale"] = float(ridge_lambda_scale)
    updated["tenor_match_apparent_rg"] = match_apparent_rg
    updated["tenor_match_apparent_rg_flux_threshold"] = 1e7
    return updated


# ---------------------------------------------------------------------------
# ── Single case runner
# ---------------------------------------------------------------------------

def run_single_case(case: dict) -> dict:
    """
    Run one simulation + both analyses.

    Returns a flat dict with all inputs and outputs.
    Raises on critical failure (will be caught by the executor).
    """
    # ── imports here so each subprocess gets fresh state ──────────────────
    import numpy as np
    from sim_utils import run_simulation_core
    from analysis_utils import (
        extract_tomchuk_parameters,
        normalize_simulated_sphere_intensity,
        calculate_sphere_input_theoretical_parameters,
        build_reconstruction_quality_summary,
    )
    from tenor_saxs import analyze_tenor_saxs_2d

    np.random.seed(case["seed"])

    params = base_sim_params(case)
    q_max = params["q_max"]

    # ─── Simulation with Caching ──────────────────────────────────────────
    cache_dir = Path(case["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    sim_dict_for_hash = {k: v for k, v in params.items() if not callable(v)}
    sim_dict_for_hash["_cache_schema_version"] = CACHE_SCHEMA_VERSION
    if bool(params.get("noise", False)):
        sim_dict_for_hash["_seed"] = int(case["seed"])
    param_hash = hashlib.md5(json.dumps(sim_dict_for_hash, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"sim_cache_{param_hash}.npz"

    t0_sim = time.perf_counter()
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                q_sim = data["q_sim"]
                i_raw = data["i_raw"]
                i_2d = data["i_2d"]
                r_vals = data["r_vals"]
                pdf_vals = data["pdf_vals"]
        except Exception:
            q_sim, i_raw, i_2d, r_vals, pdf_vals = run_simulation_core(params)
    else:
        q_sim, i_raw, i_2d, r_vals, pdf_vals = run_simulation_core(params)
        np.savez_compressed(
            cache_path,
            q_sim=q_sim,
            i_raw=i_raw,
            i_2d=i_2d,
            r_vals=r_vals,
            pdf_vals=pdf_vals,
            seed=int(case["seed"]),
        )
    t_sim = time.perf_counter() - t0_sim

    # Normalize for Tomchuk (removes amplitude scaling)
    i_norm, norm_scale = normalize_simulated_sphere_intensity(q_sim, i_raw, r_vals, pdf_vals)

    # Theoretical ground-truth parameters
    theory = calculate_sphere_input_theoretical_parameters(
        params["mean_rg"], params["p_val"], params["dist_type"]
    )

    # ─── Tomchuk analysis ─────────────────────────────────────────────────
    tomchuk_err = None
    tom_res = {}
    t_tom = 0.0
    try:
        t0_tom = time.perf_counter()
        extraction = extract_tomchuk_parameters(q_sim, i_norm, params["mean_rg"])
        t_tom = time.perf_counter() - t0_tom
        sel = extraction["selected"]
        from analysis_utils import solve_p_tomchuk, get_calculated_mean_radius
        p_pdi = solve_p_tomchuk(sel["PDI"], "PDI", params["dist_type"])
        p_pdi2 = solve_p_tomchuk(sel["PDI2"], "PDI2", params["dist_type"])
        rg_tom = sel["Rg"]
        mean_r_pdi = get_calculated_mean_radius(rg_tom, p_pdi, params["dist_type"])
        mean_r_pdi2 = get_calculated_mean_radius(rg_tom, p_pdi2, params["dist_type"])
        tom_res = dict(
            tomchuk_rg=rg_tom,
            tomchuk_p_pdi=p_pdi,
            tomchuk_p_pdi2=p_pdi2,
            tomchuk_mean_r_pdi=mean_r_pdi,
            tomchuk_mean_r_pdi2=mean_r_pdi2,
            tomchuk_PDI=sel["PDI"],
            tomchuk_PDI2=sel["PDI2"],
            tomchuk_B=sel["B"],
            tomchuk_Q=sel["Q"],
            tomchuk_lc=sel["lc"],
            tomchuk_extraction=extraction["source"],
            tomchuk_abs_err_p_pdi=safe_abs_err(p_pdi, params["p_val"]),
            tomchuk_abs_err_p_pdi2=safe_abs_err(p_pdi2, params["p_val"]),
            tomchuk_rel_err_p_pdi=safe_rel_err(p_pdi, params["p_val"]),
            tomchuk_rel_err_p_pdi2=safe_rel_err(p_pdi2, params["p_val"]),
            tomchuk_rel_err_rg=safe_rel_err(rg_tom, theory["Rg"]),
            tomchuk_time_s=t_tom,
            tomchuk_success=True,
        )
    except Exception as exc:
        tomchuk_err = str(exc)
        tom_res = dict(
            tomchuk_rg=math.nan, tomchuk_p_pdi=math.nan, tomchuk_p_pdi2=math.nan,
            tomchuk_PDI=math.nan, tomchuk_PDI2=math.nan,
            tomchuk_abs_err_p_pdi=math.nan, tomchuk_abs_err_p_pdi2=math.nan,
            tomchuk_rel_err_p_pdi=math.nan, tomchuk_rel_err_p_pdi2=math.nan,
            tomchuk_rel_err_rg=math.nan, tomchuk_time_s=0.0,
            tomchuk_extraction="failed", tomchuk_success=False,
            tomchuk_mean_r_pdi=math.nan, tomchuk_mean_r_pdi2=math.nan,
            tomchuk_B=math.nan, tomchuk_Q=math.nan, tomchuk_lc=math.nan,
        )

    # ─── TENOR-SAXS analysis ──────────────────────────────────────────────
    tenor_err = None
    ten_res = {}
    t_ten = 0.0
    try:
        psf_pairs = _build_tenor_psf_pairs(params)
        cal_grid = _build_tenor_calibration_grid(params)

        t0_ten = time.perf_counter()
        tenor = analyze_tenor_saxs_2d(
            i_2d=i_2d,
            q_max=q_max,
            dist_type=params["dist_type"],
            initial_rg_guess=params["mean_rg"],
            psf_pairs=psf_pairs,
            phi2=float(params.get("phi2", -1.0 / 63.0)),
            n_radial_bins=int(params.get("tenor_radial_bins", 10)),
            qrg_limit=float(params.get("tenor_qrg_limit", 1.0)),
            guinier_bins=min(int(params.get("n_bins", 1000)), 256),
            calibration_p_grid=cal_grid,
            psf_truncate=float(params.get("tenor_psf_truncate", 10.0)),
            use_m3=bool(params.get("tenor_use_m3", False)),
            use_g3=bool(params.get("tenor_use_g3", False)),
            ridge_lambda_scale=float(params.get("tenor_ridge_lambda_scale", 1e-6)),
            simulation_params_for_calibration=params,
            reconstruction_trials=int(params.get("tenor_reconstruction_trials", 5)),
        )
        t_ten = time.perf_counter() - t0_ten

        p_ten = tenor["p_rec"]
        rg_ten = tenor["mean_rg_rec"]
        ten_res = dict(
            tenor_rg=rg_ten,
            tenor_p=p_ten,
            tenor_mean_r=tenor["mean_r_rec"],
            tenor_rg_app=tenor["rg_app"],
            tenor_weighted_v=tenor["weighted_v"],
            tenor_dimless_jg=tenor["observable_dimless_jg"],
            tenor_abs_err_p=safe_abs_err(p_ten, params["p_val"]),
            tenor_rel_err_p=safe_rel_err(p_ten, params["p_val"]),
            tenor_rel_err_rg=safe_rel_err(rg_ten, theory["Rg"]),
            tenor_candidate_count=tenor["candidate_count"],
            tenor_plausible_count=tenor["candidate_plausible_count"],
            tenor_unstable=bool(tenor.get("unstable_no_plausible_candidate", False)),
            tenor_recon_rrms=tenor.get("best_recon_rrms_2d", math.nan),
            tenor_time_s=t_ten,
            tenor_success=True,
        )
    except Exception as exc:
        tenor_err = str(exc)
        ten_res = dict(
            tenor_rg=math.nan, tenor_p=math.nan, tenor_mean_r=math.nan,
            tenor_rg_app=math.nan, tenor_weighted_v=math.nan,
            tenor_dimless_jg=math.nan,
            tenor_abs_err_p=math.nan, tenor_rel_err_p=math.nan,
            tenor_rel_err_rg=math.nan,
            tenor_candidate_count=0, tenor_plausible_count=0,
            tenor_unstable=True, tenor_recon_rrms=math.nan,
            tenor_time_s=0.0, tenor_success=False,
        )

    # ─── Assemble output row ──────────────────────────────────────────────
    row = dict(
        experiment=case["experiment"],
        replicate=case["replicate"],
        seed=case["seed"],
        # Inputs
        dist_type=params["dist_type"],
        p_val=params["p_val"],
        mean_rg=params["mean_rg"],
        flux_exp=params.get("flux_exp", 8),
        flux=params["flux"],
        smearing_x=params["smearing_x"],
        smearing_y=params["smearing_y"],
        smearing=params["smearing"],
        pixels=params["pixels"],
        q_max=q_max,
        norm_scale=norm_scale,
        t_sim_s=t_sim,
        # Theory
        theory_rg=theory["Rg"],
        theory_p=params["p_val"],
        theory_mean_r=theory["mean_radius"],
        theory_PDI=theory["PDI"],
        theory_PDI2=theory["PDI2"],
    )
    row.update(tom_res)
    row.update(ten_res)
    if tomchuk_err:
        row["tomchuk_error"] = tomchuk_err
    if tenor_err:
        row["tenor_error"] = tenor_err

    # ─── Per-case JSON log ────────────────────────────────────────────────
    if case.get("log_dir"):
        log_path = Path(case["log_dir"]) / f"case_{case['case_id']:06d}.json"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Build serialisable log
            log_data = {
                "case_id": case["case_id"],
                "experiment": case["experiment"],
                "replicate": case["replicate"],
                "seed": case["seed"],
                "sim_params": {
                    k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in params.items()
                    if isinstance(v, (int, float, str, bool, list))
                },
                "artifacts": {
                    "cache_path": str(cache_path),
                    "case_data_path": str(Path(case["data_dir"]) / f"case_{case['case_id']:06d}.npz"),
                },
                "analysis": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                             for k, v in row.items()},
                "errors": {"tomchuk": tomchuk_err, "tenor": tenor_err},
            }
            log_path.write_text(json.dumps(log_data, indent=2, default=str))
        except Exception:
            pass

    # ─── NPZ data persistence ─────────────────────────────────────────────
    if case.get("data_dir"):
        npz_path = Path(case["data_dir"]) / f"case_{case['case_id']:06d}.npz"
        try:
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_path,
                q_sim=q_sim,
                i_raw=i_raw,
                i_norm=i_norm,
                i_2d=i_2d,
                r_vals=r_vals,
                pdf_vals=pdf_vals,
                seed=int(case["seed"]),
                q_max=float(q_max),
            )
        except Exception:
            pass

    return row


# ---------------------------------------------------------------------------
# ── Experiment builders
# ---------------------------------------------------------------------------

def _case_base(experiment, replicate, seed, overrides, case_id, root_dir):
    case = dict(
        experiment=experiment,
        replicate=replicate,
        seed=seed,
        case_id=case_id,
        log_dir=str(root_dir / "logs"),
    )
    case.update(overrides)
    return case


def build_exp1_p_sweep(root_dir, fast=False):
    """Experiment 1: polydispersity sweep at fixed flux/smearing."""
    grid = P_GRID_FAST if fast else P_GRID_FULL
    n_rep = N_REP_FAST if fast else N_REP_FULL
    cases, seed = [], 1000
    for p in grid:
        for rep in range(1, n_rep + 1):
            cases.append(_case_base(
                "exp1_p_sweep", rep, seed,
                dict(p_val=p, flux_exp=8, smearing_x=3.0, smearing_y=3.0,
                     dist_type="Lognormal"),
                len(cases) + 1, root_dir))
            seed += 1
    return cases


def build_exp2_flux(root_dir, fast=False):
    """Experiment 2: photon count (flux) sensitivity."""
    grid = FLUX_GRID_FAST if fast else FLUX_GRID_FULL
    n_rep = N_REP_FAST if fast else N_REP_FULL
    cases, seed = [], 5000
    for fexp in grid:
        for rep in range(1, n_rep + 1):
            cases.append(_case_base(
                "exp2_flux", rep, seed,
                dict(p_val=0.3, flux_exp=fexp, smearing_x=3.0, smearing_y=3.0,
                     dist_type="Lognormal"),
                len(cases) + 1, root_dir))
            seed += 1
    return cases


def build_exp3_smearing(root_dir, fast=False):
    """Experiment 3: beam smearing (isotropic) sensitivity."""
    grid = SMEAR_GRID_FAST if fast else SMEAR_GRID_FULL
    n_rep = N_REP_FAST if fast else N_REP_FULL
    cases, seed = [], 10000
    for sm in grid:
        for rep in range(1, n_rep + 1):
            cases.append(_case_base(
                "exp3_smearing", rep, seed,
                dict(p_val=0.3, flux_exp=8, smearing_x=float(sm), smearing_y=float(sm),
                     dist_type="Lognormal"),
                len(cases) + 1, root_dir))
            seed += 1
    return cases


def build_exp4_distribution(root_dir, fast=False):
    """Experiment 4: distribution shape robustness."""
    dists = DIST_FAST if fast else DISTRIBUTIONS
    n_rep = N_REP_FAST if fast else N_REP_FAST + 2  # 5 for full
    cases, seed = [], 20000
    for dist in dists:
        for rep in range(1, n_rep + 1):
            cases.append(_case_base(
                "exp4_distribution", rep, seed,
                dict(p_val=0.3, flux_exp=8, smearing_x=3.0, smearing_y=3.0,
                     dist_type=dist),
                len(cases) + 1, root_dir))
            seed += 1
    return cases


def build_exp5_anisotropy(root_dir, fast=False):
    """Experiment 5: PSF anisotropy immunity."""
    beams = ANISO_BEAMS_FAST if fast else ANISO_BEAMS_FULL
    n_rep = N_REP_FAST if fast else N_REP_FULL
    cases, seed = [], 30000
    for (sx, sy) in beams:
        for rep in range(1, n_rep + 1):
            cases.append(_case_base(
                "exp5_anisotropy", rep, seed,
                dict(p_val=0.3, flux_exp=8, smearing_x=sx, smearing_y=sy,
                     dist_type="Lognormal"),
                len(cases) + 1, root_dir))
            seed += 1
    return cases


def build_exp6_joint_pxflux(root_dir, fast=False):
    """Experiment 6: joint p × flux × smearing performance map."""
    p_vals = [0.1, 0.3, 0.6]
    flux_exps = FLUX_GRID_FAST if fast else [7, 8, 9]
    smearings = [3.0, 10.0]
    n_rep = N_REP_FAST if fast else 5
    cases, seed = [], 40000
    for p in p_vals:
        for fexp in flux_exps:
            for sm in smearings:
                for rep in range(1, n_rep + 1):
                    cases.append(_case_base(
                        "exp6_joint", rep, seed,
                        dict(p_val=p, flux_exp=fexp,
                             smearing_x=float(sm), smearing_y=float(sm),
                             dist_type="Lognormal"),
                        len(cases) + 1, root_dir))
                    seed += 1
    return cases


def build_pilot_rg(root_dir):
    """Pilot: Rg-dependence check (small scale)."""
    cases, seed = [], 50000
    for rg in RG_GRID_PILOT:
        for rep in range(1, N_REP_FAST + 1):
            case = _case_base(
                "pilot_rg", rep, seed,
                dict(p_val=0.3, flux_exp=8, smearing_x=3.0, smearing_y=3.0,
                     dist_type="Lognormal", mean_rg=rg),
                len(cases) + 1, root_dir)
            cases.append(case)
            seed += 1
    return cases

def build_exp7_psweep_synchrotron(root_dir, fast=False):
    """Experiment 7: p-sweep at baseline synchrotron smearing (3x15)."""
    grid = P_GRID_FAST if fast else P_GRID_FULL
    n_rep = N_REP_FAST if fast else N_REP_FULL
    cases, seed = [], 60000
    for p in grid:
        for rep in range(1, n_rep + 1):
            cases.append(_case_base(
                "exp7_p_synchrotron", rep, seed,
                dict(p_val=p, flux_exp=8, smearing_x=3.0, smearing_y=15.0,
                     dist_type="Lognormal"),
                len(cases) + 1, root_dir))
            seed += 1
    return cases

def build_exp8_psweep_home(root_dir, fast=False):
    """Experiment 8: p-sweep at baseline home-source smearing (20x20)."""
    grid = P_GRID_FAST if fast else P_GRID_FULL
    n_rep = N_REP_FAST if fast else N_REP_FULL
    cases, seed = [], 70000
    for p in grid:
        for rep in range(1, n_rep + 1):
            cases.append(_case_base(
                "exp8_p_home", rep, seed,
                dict(p_val=p, flux_exp=8, smearing_x=20.0, smearing_y=20.0,
                     dist_type="Lognormal"),
                len(cases) + 1, root_dir))
            seed += 1
    return cases

def build_exp9_flux_synchrotron(root_dir, fast=False):
    """Experiment 9: flux sweep at baseline synchrotron smearing (3x15)."""
    grid = FLUX_GRID_FAST if fast else FLUX_GRID_FULL
    n_rep = N_REP_FAST if fast else N_REP_FULL
    cases, seed = [], 80000
    for fexp in grid:
        for rep in range(1, n_rep + 1):
            cases.append(_case_base(
                "exp9_flux_synchrotron", rep, seed,
                dict(p_val=0.3, flux_exp=fexp, smearing_x=3.0, smearing_y=15.0,
                     dist_type="Lognormal"),
                len(cases) + 1, root_dir))
            seed += 1
    return cases


# ---------------------------------------------------------------------------
# ── Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TENOR-SAXS vs. Tomchuk comparative study"
    )
    parser.add_argument("--fast", action="store_true",
                        help="Use reduced grids for smoke testing")
    parser.add_argument("--output-root", default="studies",
                        help="Parent directory for study output")
    parser.add_argument("--workers", type=int,
                        default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument(
        "--large-data-root",
        default=str(DEFAULT_LARGE_DATA_ROOT),
        help="Directory for large cached simulation files and saved 2D detector data",
    )
    parser.add_argument(
        "--tenor-ridge-lambda-scale",
        type=float,
        default=1e-6,
        help="Trace-scaled ridge regularization strength for TENOR polynomial extraction",
    )
    parser.add_argument(
        "--disable-tenor-apparent-rg-match",
        action="store_true",
        help="Disable apparent-Rg matching when building TENOR calibration curves",
    )
    parser.add_argument("--experiments", nargs="+",
                        default=["all"],
                        help="Experiments to run: all, 1, 2, 3, 4, 5, 6, 7, 8, 9, pilot_rg")
    args = parser.parse_args()

    # ── Output directory tree ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "fast" if args.fast else "full"
    root = Path(args.output_root) / f"tenor_vs_tomchuk_{mode_tag}_{timestamp}"
    large_layout = make_large_file_layout(root.name, args.large_data_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    large_layout["cache_dir"].mkdir(parents=True, exist_ok=True)
    large_layout["data_dir"].mkdir(parents=True, exist_ok=True)

    print(f"Study directory: {root}")
    print(f"Large-file directory: {large_layout['study_root']}")
    print(f"Mode: {'FAST' if args.fast else 'FULL'} | Workers: {args.workers}")

    # ── Build cases ────────────────────────────────────────────────────────
    run_all = "all" in args.experiments
    all_cases: list[dict] = []

    builders = {
        "1": ("exp1_p_sweep",    build_exp1_p_sweep),
        "2": ("exp2_flux",       build_exp2_flux),
        "3": ("exp3_smearing",   build_exp3_smearing),
        "4": ("exp4_distribution", build_exp4_distribution),
        "5": ("exp5_anisotropy", build_exp5_anisotropy),
        "6": ("exp6_joint",      build_exp6_joint_pxflux),
        "7": ("exp7_p_synchrotron", build_exp7_psweep_synchrotron),
        "8": ("exp8_p_home", build_exp8_psweep_home),
        "9": ("exp9_flux_synchrotron", build_exp9_flux_synchrotron),
        "pilot_rg": ("pilot_rg", build_pilot_rg),
    }

    for key, (name, fn) in builders.items():
        if run_all or key in args.experiments or name in args.experiments:
            if key == "pilot_rg":
                batch = fn(root)  # pilot always fast
            else:
                batch = fn(root, fast=args.fast)
            batch = [
                apply_tenor_overrides(
                    {
                        **case,
                        "cache_dir": str(large_layout["cache_dir"]),
                        "data_dir": str(large_layout["data_dir"]),
                    },
                    ridge_lambda_scale=args.tenor_ridge_lambda_scale,
                    match_apparent_rg=("adaptive" if not args.disable_tenor_apparent_rg_match else False),
                )
                for case in batch
            ]
            all_cases.extend(batch)
            print(f"  {name}: {len(batch)} cases")

    print(f"Total cases: {len(all_cases)}")

    # ── Save metadata ──────────────────────────────────────────────────────
    metadata = {
        "timestamp": timestamp,
        "mode": mode_tag,
        "workers": args.workers,
        "total_cases": len(all_cases),
        "large_data_root": str(large_layout["root"]),
        "large_data_study_dir": str(large_layout["study_root"]),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "tenor_settings_override": {
            "tenor_ridge_lambda_scale": float(args.tenor_ridge_lambda_scale),
            "tenor_match_apparent_rg": ("adaptive" if not args.disable_tenor_apparent_rg_match else False),
            "tenor_match_apparent_rg_flux_threshold": 1e7,
        },
        "baseline": {k: (v.tolist() if hasattr(v, "tolist") else v)
                     for k, v in BASELINE.items()},
        "grids": {
            "P_GRID": P_GRID_FAST if args.fast else P_GRID_FULL,
            "FLUX_GRID": FLUX_GRID_FAST if args.fast else FLUX_GRID_FULL,
            "SMEAR_GRID": SMEAR_GRID_FAST if args.fast else SMEAR_GRID_FULL,
            "DISTRIBUTIONS": DIST_FAST if args.fast else DISTRIBUTIONS,
        },
    }
    (root / "study_metadata.json").write_text(json.dumps(metadata, indent=2))

    # ── Execute ────────────────────────────────────────────────────────────
    rows: list[dict] = []
    errors: list[dict] = []
    checkpoint = root / "progress_checkpoint.csv"

    t_start = time.perf_counter()

    if args.workers <= 1:
        for idx, case in enumerate(all_cases, 1):
            try:
                rows.append(run_single_case(case))
            except Exception as exc:
                err = case.copy()
                err["error"] = traceback.format_exc()
                errors.append(err)
            if idx % 20 == 0 or idx == len(all_cases):
                elapsed = time.perf_counter() - t_start
                eta = (elapsed / idx) * (len(all_cases) - idx)
                print(f"  [{idx}/{len(all_cases)}] elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                      flush=True)
                if rows:
                    pd.DataFrame(rows).to_csv(checkpoint, index=False)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_single_case, c): c for c in all_cases}
            for idx, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    err = case.copy()
                    err["error"] = traceback.format_exc()
                    errors.append(err)
                if idx % 20 == 0 or idx == len(all_cases):
                    elapsed = time.perf_counter() - t_start
                    eta = (elapsed / idx) * (len(all_cases) - idx)
                    print(f"  [{idx}/{len(all_cases)}] elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                          flush=True)
                    if rows:
                        pd.DataFrame(rows).to_csv(checkpoint, index=False)

    # ── Save final results ─────────────────────────────────────────────────
    df = pd.DataFrame(rows).sort_values(
        ["experiment", "dist_type", "p_val", "flux_exp",
         "smearing", "replicate"],
        ignore_index=True,
    )
    summary_csv = root / "summary_results.csv"
    df.to_csv(summary_csv, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(root / "errors.csv", index=False)

    elapsed_total = time.perf_counter() - t_start
    print(f"\nDone! {len(rows)} successful, {len(errors)} failed.")
    print(f"Total elapsed: {elapsed_total/60:.1f} min")
    print(f"Summary CSV: {summary_csv}")
    print(f"Study root:  {root}")

    # ── Print quick sanity digest ──────────────────────────────────────────
    if not df.empty:
        print("\n── Quick results digest ─────────────────────────────────────")
        for exp in df["experiment"].unique():
            sub = df[df["experiment"] == exp]
            ten_ok = sub["tenor_success"].sum() if "tenor_success" in sub.columns else "-"
            tom_ok = sub["tomchuk_success"].sum() if "tomchuk_success" in sub.columns else "-"
            tenor_mae = sub["tenor_abs_err_p"].median() if "tenor_abs_err_p" in sub.columns else math.nan
            tomchuk_mae = sub["tomchuk_abs_err_p_pdi"].median() if "tomchuk_abs_err_p_pdi" in sub.columns else math.nan
            print(f"  {exp:30s}: TENOR ok={ten_ok} MAE_p={tenor_mae:.3f} | "
                  f"Tomchuk ok={tom_ok} MAE_p={tomchuk_mae:.3f}")

    return root, df


if __name__ == "__main__":
    main()
