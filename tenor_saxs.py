"""Standalone TENOR-SAXS analysis utilities for 2D sphere data.

This module follows the practical recipe in TenorSAXS.pdf:
1. Estimate the apparent Guinier radius from the raw 2D pattern.
2. Create pairs of anisotropic digital Gaussian PSFs.
3. Fit the log-ratio map to the azimuthal form a(q) + b(q) cos(2 chi).
4. Recover the TENOR observable y_G = g1 / g0 from the radial function G(q).
5. Infer the weighted variance V, then map it to p for a chosen distribution.
6. Convert the weighted mean Rg back to the arithmetic mean Rg.

The current implementation is intentionally limited to polydisperse solid
spheres, because the paper's table of form-factor curvatures is currently used
only for the solid-sphere case in this codebase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.ndimage import gaussian_filter
from scipy.optimize import bisect

from analysis_utils import estimate_guinier_parameters, get_normalized_moment
from app_settings import get_forward_flux
from sim_utils import (
    build_detector_q_grid,
    radial_average_detector_image,
    run_simulation_core,
)


SOLID_SPHERE_PHI2 = -1.0 / 63.0


@dataclass(frozen=True)
class PsfPair:
    sigma_x_1: float
    sigma_y_1: float
    sigma_x_2: float
    sigma_y_2: float


def build_default_psf_pairs(
    sigma_x_start=1.2,
    sigma_y_start=0.6,
    sigma_step=0.4,
    pair_count=5,
    secondary_ratio=0.5,
):
    """Build the default TENOR PSF quartet scan from user-facing settings."""
    pairs = []
    pair_count = max(int(pair_count), 1)
    sigma_step = max(float(sigma_step), 1e-6)
    secondary_ratio = max(float(secondary_ratio), 1e-3)
    for idx in range(pair_count):
        sigma_x_1 = float(sigma_x_start) + idx * sigma_step
        sigma_y_1 = float(sigma_y_start) + idx * (sigma_step / 2.0)
        pairs.append(
            PsfPair(
                sigma_x_1=sigma_x_1,
                sigma_y_1=sigma_y_1,
                sigma_x_2=sigma_x_1 * secondary_ratio,
                sigma_y_2=sigma_y_1 * secondary_ratio,
            )
        )
    return pairs


def apparent_rg_from_2d(i_2d, q_max, initial_rg_guess=4.0, n_bins=256):
    """Estimate the apparent Guinier radius from the raw 2D image."""
    pixels = np.asarray(i_2d).shape[0]
    _, _, _, q_r, _ = build_detector_q_grid(pixels, q_max=q_max)
    q_1d, i_1d = radial_average_detector_image(
        i_2d=i_2d,
        q_r=q_r,
        q_min=0.0,
        q_max=q_max,
        n_bins=n_bins,
        binning_mode="Linear",
    )
    rg_app, g_app, valid = estimate_guinier_parameters(
        q_1d,
        i_1d,
        initial_rg_guess=initial_rg_guess,
        mode="Sphere",
    )
    return {
        "q_1d": q_1d,
        "i_1d": i_1d,
        "rg_app": float(rg_app),
        "g_app": float(g_app),
        "valid": bool(valid),
    }


def raw_g1_over_g0_from_v(v, phi2=SOLID_SPHERE_PHI2):
    num = 1.0 + 18.0 * phi2 + 10.0 * v + 108.0 * v * phi2
    den = 4.0 * (1.0 + v) ** 2
    return num / den


def dimless_jg_from_v(v, phi2=SOLID_SPHERE_PHI2):
    """Dimensionless J_G observable from Eq. (18) of the paper."""
    num = 1.0 + 18.0 * phi2 + 10.0 * v + 108.0 * v * phi2
    den = -3.0 * (1.0 + v) ** 2
    return num / den


def weighted_variance_from_p(p_val, dist_type):
    """Return the scattering-strength-weighted variance V for a size family.

    For spheres, scattering strength scales as R^6, and Rg is proportional to R,
    so the same normalized moments apply to Rg up to a constant scale factor.
    """
    m6 = get_normalized_moment(6, p_val, dist_type)
    m7 = get_normalized_moment(7, p_val, dist_type)
    m8 = get_normalized_moment(8, p_val, dist_type)
    if m6 <= 0 or m7 <= 0 or m8 <= 0:
        return 0.0
    return max((m6 * m8) / (m7 ** 2) - 1.0, 0.0)


def weighted_mean_to_arithmetic_mean_ratio(p_val, dist_type):
    m6 = get_normalized_moment(6, p_val, dist_type)
    m7 = get_normalized_moment(7, p_val, dist_type)
    if m6 <= 0 or m7 <= 0:
        return 1.0
    return m7 / m6


def solve_v_from_j_g(target_j_g, phi2=SOLID_SPHERE_PHI2, v_max=0.8):
    if not np.isfinite(target_j_g):
        return 0.0

    f0 = dimless_jg_from_v(0.0, phi2=phi2) - target_j_g
    f1 = dimless_jg_from_v(v_max, phi2=phi2) - target_j_g
    if abs(f0) < 1e-12:
        return 0.0
    if f0 * f1 > 0:
        samples = np.linspace(0.0, v_max, 400)
        vals = dimless_jg_from_v(samples, phi2=phi2)
        idx = int(np.argmin(np.abs(vals - target_j_g)))
        return float(samples[idx])
    return float(
        bisect(
            lambda v: dimless_jg_from_v(v, phi2=phi2) - target_j_g,
            0.0,
            v_max,
            xtol=1e-6,
        )
    )


def solve_p_from_weighted_v(target_v, dist_type, p_max=6.0):
    if not np.isfinite(target_v) or target_v <= 1e-12:
        return 0.0

    f0 = weighted_variance_from_p(1e-6, dist_type) - target_v
    f1 = weighted_variance_from_p(p_max, dist_type) - target_v
    if f0 * f1 > 0:
        samples = np.linspace(1e-4, p_max, 600)
        vals = np.array([weighted_variance_from_p(p, dist_type) for p in samples])
        idx = int(np.argmin(np.abs(vals - target_v)))
        return float(samples[idx])
    return float(
        bisect(
            lambda p: weighted_variance_from_p(p, dist_type) - target_v,
            1e-6,
            p_max,
            xtol=1e-6,
        )
    )


def apply_anisotropic_gaussian(i_2d, sigma_x, sigma_y):
    return gaussian_filter(np.asarray(i_2d, dtype=float), sigma=(sigma_y, sigma_x))


def fit_weighted_centered_tenor_model(q_sq, chi, log_ratio, intensity_weights, use_m3=True, use_g3=True):
    """Port of the MATLAB weighted centered fit used for TENOR observable extraction."""
    q_sq = np.asarray(q_sq, dtype=float).ravel()
    chi = np.asarray(chi, dtype=float).ravel()
    log_ratio = np.asarray(log_ratio, dtype=float).ravel()
    intensity_weights = np.asarray(intensity_weights, dtype=float).ravel()

    cos2 = np.cos(2.0 * chi)
    valid = (
        np.isfinite(q_sq)
        & np.isfinite(cos2)
        & np.isfinite(log_ratio)
        & np.isfinite(intensity_weights)
        & (intensity_weights > 0)
    )
    if np.count_nonzero(valid) < 30:
        return None

    q_sq = q_sq[valid]
    cos2 = cos2[valid]
    y = log_ratio[valid]
    w = np.sqrt(intensity_weights[valid])
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return None

    mu_q = float(np.sum(w * q_sq) / w_sum)
    q_centered = q_sq - mu_q

    if use_g3:
        g_cols = [np.ones_like(q_sq), q_centered, q_centered**2, q_centered**3]
    else:
        g_cols = [np.ones_like(q_sq), q_centered, q_centered**2]

    if use_m3:
        m_cols = [q_sq * cos2, (q_sq**2) * cos2, (q_sq**3) * cos2]
    else:
        m_cols = [q_sq * cos2, (q_sq**2) * cos2]

    x = np.column_stack(g_cols + m_cols)
    xw = x * w[:, None]
    yw = y * w

    try:
        q_mat, r_mat = qr(xw, mode="economic")
    except Exception:
        return None

    rank = int(np.linalg.matrix_rank(r_mat))
    if rank < x.shape[1]:
        coeff_centered, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        if coeff_centered.shape[0] != x.shape[1]:
            return None
    else:
        coeff_centered = solve_triangular(r_mat, q_mat.T @ yw)

    fitted = x @ coeff_centered
    resid = y - fitted
    rmse = float(np.sqrt(np.mean(resid**2)))

    # Estimate covariance in the centered basis, following the MATLAB workflow
    # that grades ratios by their CI95 width.
    n_obs = x.shape[0]
    dof = max(n_obs - x.shape[1], 1)
    sse = float(np.sum((w * resid) ** 2))
    s2 = sse / dof
    try:
        r_inv = np.linalg.solve(r_mat, np.eye(r_mat.shape[0]))
        cov_centered = s2 * (r_inv @ r_inv.T)
    except Exception:
        cov_centered = None

    # Map centered G-block coefficients back to the original Q basis.
    k_g = len(g_cols)
    k_m = len(m_cols)
    k_total = k_g + k_m
    transform = np.zeros((k_total, k_total), dtype=float)
    transform[:k_g, :k_g] = np.eye(k_g)
    if k_g >= 3:
        transform[0, 0] = 1.0
        transform[0, 1] = -mu_q
        transform[0, 2] = mu_q**2
        transform[1, 1] = 1.0
        transform[1, 2] = -2.0 * mu_q
        transform[2, 2] = 1.0
    if k_g >= 4:
        transform[0, 3] = -(mu_q**3)
        transform[1, 3] = 3.0 * (mu_q**2)
        transform[2, 3] = -3.0 * mu_q
        transform[3, 3] = 1.0
    transform[k_g:, k_g:] = np.eye(k_m)
    coeffs = transform @ coeff_centered
    cov_coeffs = None if cov_centered is None else (transform @ cov_centered @ transform.T)

    g_coeffs = coeffs[:k_g]
    m_coeffs = coeffs[k_g:]
    g0 = float(g_coeffs[0])
    g1 = float(g_coeffs[1]) if len(g_coeffs) > 1 else np.nan
    g2 = float(g_coeffs[2]) if len(g_coeffs) > 2 else np.nan
    m0 = float(m_coeffs[0]) if len(m_coeffs) > 0 else np.nan
    m1 = float(m_coeffs[1]) if len(m_coeffs) > 1 else np.nan

    raw_g1_over_g0 = g1 / g0 if abs(g0) > 1e-20 else np.nan
    g100_ratio = g1 / (g0**2) if abs(g0) > 1e-20 else np.nan
    g210_ratio = g2 / (g1 * g0) if abs(g0 * g1) > 1e-20 else np.nan
    m210_ratio = m1 / (m0 * g0) if abs(m0 * g0) > 1e-20 else np.nan
    g_ratio_ci95_width = np.nan
    g_ratio_grade = 0.0
    if cov_coeffs is not None and abs(g0) > 1e-20:
        grad = np.zeros(k_total, dtype=float)
        grad[0] = -g1 / (g0**2)
        if k_total > 1:
            grad[1] = 1.0 / g0
        var_ratio = float(grad @ cov_coeffs @ grad.T)
        if np.isfinite(var_ratio):
            se_ratio = math.sqrt(max(var_ratio, 0.0))
            z95 = 1.95996398454005
            g_ratio_ci95_width = 2.0 * z95 * se_ratio
            g_ratio_grade = float(1.0 / (1.0 + g_ratio_ci95_width))

    return {
        "g_coeffs": g_coeffs,
        "m_coeffs": m_coeffs,
        "fit_rmse": rmse,
        "fit_sse": sse,
        "q_sq": q_sq,
        "fitted": fitted,
        "mu_q": mu_q,
        "raw_g1_over_g0": float(raw_g1_over_g0),
        "raw_g100_ratio": float(g100_ratio),
        "raw_g210_ratio": float(g210_ratio) if np.isfinite(g210_ratio) else np.nan,
        "raw_m210_ratio": float(m210_ratio) if np.isfinite(m210_ratio) else np.nan,
        "g_ratio_ci95_width": float(g_ratio_ci95_width) if np.isfinite(g_ratio_ci95_width) else np.nan,
        "g_ratio_grade": float(g_ratio_grade),
    }


def extract_pair_observables(
    i_2d,
    q_max,
    pair,
    rg_app,
    qrg_limit=0.85,
    n_radial_bins=18,
    psf_truncate=4.0,
    use_m3=True,
    use_g3=True,
):
    pixels = i_2d.shape[0]
    dq = (2.0 * q_max) / max(pixels - 1, 1)
    _, qx, qy, q_r, _ = build_detector_q_grid(pixels, q_max=q_max)
    base_weights = np.maximum(i_2d, np.nanmax(i_2d) * 1e-12 if np.isfinite(i_2d).any() else 1e-12)

    img1 = apply_anisotropic_gaussian(i_2d, pair.sigma_x_1, pair.sigma_y_1)
    img2 = apply_anisotropic_gaussian(i_2d, pair.sigma_x_2, pair.sigma_y_2)
    log_ratio = np.log(np.maximum(img1, 1e-12)) - np.log(np.maximum(img2, 1e-12))

    q_fit_max = min(q_max, qrg_limit / max(rg_app, 1e-6))
    dead_pixels = int(
        2
        * max(
            int(math.ceil(psf_truncate * pair.sigma_x_1)),
            int(math.ceil(psf_truncate * pair.sigma_y_1)),
            int(math.ceil(psf_truncate * pair.sigma_x_2)),
            int(math.ceil(psf_truncate * pair.sigma_y_2)),
            1,
        )
    )
    q_dead = dead_pixels * abs(dq)
    fit_mask = (q_r < min(q_fit_max, np.nanmax(q_r) - q_dead)) & (q_r > q_dead)
    if np.count_nonzero(fit_mask) < 30:
        return None

    dsx2 = (pair.sigma_x_1**2 - pair.sigma_x_2**2) * (dq**2)
    dsy2 = (pair.sigma_y_1**2 - pair.sigma_y_2**2) * (dq**2)
    sigma0 = 0.5 * (dsx2 + dsy2)
    delta0 = 0.5 * (dsx2 - dsy2)
    if abs(sigma0) < 1e-20 or abs(delta0) < 1e-20:
        return None

    fit_res = fit_weighted_centered_tenor_model(
        q_sq=(q_r[fit_mask] ** 2),
        chi=np.arctan2(qy[fit_mask], qx[fit_mask]),
        log_ratio=log_ratio[fit_mask],
        intensity_weights=base_weights[fit_mask],
        use_m3=use_m3,
        use_g3=use_g3,
    )
    if fit_res is None:
        return None

    g_coeffs = np.asarray(fit_res["g_coeffs"], dtype=float)
    m_coeffs = np.asarray(fit_res["m_coeffs"], dtype=float)
    g0 = float(g_coeffs[0])
    if abs(g0) < 1e-20:
        return None

    raw_g1_over_g0 = float(fit_res["raw_g1_over_g0"])
    score = abs(fit_res["fit_rmse"])
    return {
        "pair": pair,
        "rg_app": float(rg_app),
        "q_vals": np.asarray(np.sqrt(fit_res["q_sq"]), dtype=float),
        "g_radial": np.asarray([], dtype=float),
        "m_radial": np.asarray([], dtype=float),
        "g_coeffs": g_coeffs,
        "m_coeffs": m_coeffs,
        "g_rmse": float(fit_res["fit_rmse"]),
        "m_rmse": float(fit_res["fit_rmse"]),
        "raw_g1_over_g0": float(raw_g1_over_g0),
        "raw_g100_ratio": float(fit_res["raw_g100_ratio"]),
        "raw_g210_ratio": float(fit_res["raw_g210_ratio"]),
        "raw_m210_ratio": float(fit_res["raw_m210_ratio"]),
        "raw_m1_over_m0": float(np.nan),
        "dimless_jg": float(raw_g1_over_g0 / max(rg_app**2, 1e-12)),
        "score": float(score),
        "grade_g": float(fit_res.get("g_ratio_grade", 0.0)),
        "g_ratio_ci95_width": float(fit_res.get("g_ratio_ci95_width", np.nan)),
        "q_fit_max": float(q_fit_max),
    }


def _compute_relative_rmse(observed, model, mask=None):
    observed = np.asarray(observed, dtype=float)
    model = np.asarray(model, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        observed = observed[mask]
        model = model[mask]
    valid = np.isfinite(observed) & np.isfinite(model)
    if np.count_nonzero(valid) < 10:
        return np.inf, 1.0
    observed = observed[valid]
    model = model[valid]
    denom = float(np.dot(model, model))
    scale = float(np.dot(observed, model) / denom) if denom > 1e-20 else 1.0
    residual = observed - scale * model
    norm = max(float(np.linalg.norm(observed)), 1e-12)
    return float(np.linalg.norm(residual) / norm), scale


def _select_reconstruction_candidates(candidates, j_lo, j_hi, reconstruction_trials):
    reconstruction_trials = max(int(reconstruction_trials), 1)
    if not candidates:
        return []

    def candidate_rank(item):
        j_val = float(item.get("dimless_jg", np.nan))
        if np.isfinite(j_val):
            if j_lo <= j_val <= j_hi:
                distance = 0.0
            else:
                distance = min(abs(j_val - j_lo), abs(j_val - j_hi))
        else:
            distance = np.inf
        grade = -float(item.get("grade_g", 0.0))
        ci_width = float(item.get("g_ratio_ci95_width", np.inf))
        score = float(item.get("score", np.inf))
        sigma = float(getattr(item["pair"], "sigma_x_1", np.inf))
        return (distance, grade, ci_width, score, sigma)

    ranked = sorted(candidates, key=candidate_rank)
    return ranked[: min(reconstruction_trials, len(ranked))]


def _evaluate_reconstruction_candidate(
    *,
    candidate,
    observed_i_2d,
    q_max,
    dist_type,
    initial_rg_guess,
    calibration_p_grid,
    guinier_bins,
    qrg_limit,
    n_radial_bins,
    psf_truncate,
    use_m3,
    use_g3,
    simulation_params_for_calibration,
):
    calibration = calibrate_p_from_simulation(
        target_j_g=float(candidate["dimless_jg"]),
        dist_type=dist_type,
        q_max=q_max,
        pixels=observed_i_2d.shape[0],
        pair=candidate["pair"],
        mean_rg_ref=initial_rg_guess,
        p_grid=calibration_p_grid,
        tenor_settings={
            "tenor_guinier_bins": guinier_bins,
            "tenor_qrg_limit": qrg_limit,
            "tenor_radial_bins": n_radial_bins,
            "tenor_psf_truncate": psf_truncate,
            "tenor_use_m3": use_m3,
            "tenor_use_g3": use_g3,
        },
        simulation_params=simulation_params_for_calibration,
    )
    if calibration is None:
        v_weighted = solve_v_from_j_g(float(candidate["dimless_jg"]))
        p_rec = solve_p_from_weighted_v(v_weighted, dist_type=dist_type)
    else:
        v_weighted = float(calibration["weighted_v"])
        p_rec = float(calibration["p_rec"])

    rg_app = float(simulation_params_for_calibration.get("mean_rg", initial_rg_guess))
    rg_app = float(candidate.get("rg_app", initial_rg_guess))
    r0_weighted = rg_app / math.sqrt(max(1.0 + v_weighted, 1e-12))
    mean_ratio = weighted_mean_to_arithmetic_mean_ratio(p_rec, dist_type)
    mean_rg = r0_weighted / mean_ratio if mean_ratio > 0 else r0_weighted
    mean_radius = mean_rg * math.sqrt(5.0 / 3.0)

    sim_params = dict(simulation_params_for_calibration or {})
    if "flux" not in sim_params:
        try:
            sim_params["flux"] = float(get_forward_flux(sim_params))
        except Exception:
            sim_params["flux"] = 1e8
    if "n_bins" not in sim_params:
        sim_params["n_bins"] = max(int(guinier_bins), 64)
    if "binning_mode" not in sim_params:
        sim_params["binning_mode"] = "Logarithmic"
    if "smearing_x" not in sim_params:
        sim_params["smearing_x"] = 0.0
    if "smearing_y" not in sim_params:
        sim_params["smearing_y"] = 0.0
    sim_params.update(
        {
            "mean_rg": float(mean_rg),
            "p_val": float(p_rec),
            "dist_type": dist_type,
            "mode": "Sphere",
            "pixels": int(observed_i_2d.shape[0]),
            "q_min": 0.0,
            "q_max": float(q_max),
            "noise": False,
        }
    )
    _, _, recon_i_2d, _, _ = run_simulation_core(sim_params)
    pixels = observed_i_2d.shape[0]
    dq = (2.0 * q_max) / max(pixels - 1, 1)
    _, _, _, q_r, _ = build_detector_q_grid(pixels, q_max=q_max)
    pair = candidate["pair"]
    dead_pixels = int(
        2
        * max(
            int(math.ceil(psf_truncate * pair.sigma_x_1)),
            int(math.ceil(psf_truncate * pair.sigma_y_1)),
            int(math.ceil(psf_truncate * pair.sigma_x_2)),
            int(math.ceil(psf_truncate * pair.sigma_y_2)),
            1,
        )
    )
    q_dead = dead_pixels * abs(dq)
    q_fit_max = min(q_max, qrg_limit / max(candidate.get("rg_app", initial_rg_guess), 1e-6))
    fit_mask = (q_r < min(q_fit_max, np.nanmax(q_r) - q_dead)) & (q_r > q_dead)
    rrms_2d, scale_2d = _compute_relative_rmse(observed_i_2d, recon_i_2d, mask=fit_mask)

    return {
        "candidate": candidate,
        "calibration": calibration,
        "weighted_v": float(v_weighted),
        "p_rec": float(p_rec),
        "mean_rg_rec": float(mean_rg),
        "mean_r_rec": float(mean_radius),
        "weighted_mean_rg": float(r0_weighted),
        "recon_rrms_2d": float(rrms_2d),
        "recon_scale_2d": float(scale_2d),
    }


def calibrate_p_from_simulation(
    target_j_g,
    dist_type,
    q_max,
    pixels,
    pair,
    mean_rg_ref=4.0,
    p_grid=None,
    tenor_settings=None,
    simulation_params=None,
):
    """Recipe-inspired calibration of the observable against forward simulations."""
    if p_grid is None:
        p_grid = np.concatenate(
            [
                np.linspace(0.01, 0.12, 8),
                np.linspace(0.15, 0.6, 10),
            ]
        )

    rows = []
    for p_val in p_grid:
        row = build_tenor_simulation_row(
            mean_rg=float(mean_rg_ref),
            p_val=float(p_val),
            dist_type=dist_type,
            q_max=float(q_max),
            pixels=int(pixels),
            pair=pair,
            tenor_settings=tenor_settings,
            simulation_params=simulation_params,
        )
        if row is not None:
            rows.append(row)

    if len(rows) < 4:
        return None

    # Section 4, step 4(b): interpolate the calibration curve to recover the
    # variance / distribution width that reproduces the measured observable.
    rows = sorted(rows, key=lambda row: row["dimless_jg"])
    j_vals = np.array([row["dimless_jg"] for row in rows], dtype=float)
    p_vals = np.array([row["p_val"] for row in rows], dtype=float)
    v_vals = np.array([row["weighted_v"] for row in rows], dtype=float)

    if target_j_g <= j_vals[0]:
        p_rec = p_vals[0]
        v_rec = v_vals[0]
    elif target_j_g >= j_vals[-1]:
        p_rec = p_vals[-1]
        v_rec = v_vals[-1]
    else:
        p_rec = float(np.interp(target_j_g, j_vals, p_vals))
        v_rec = float(np.interp(target_j_g, j_vals, v_vals))

    return {
        "p_rec": p_rec,
        "weighted_v": v_rec,
        "rows": rows,
    }


def build_tenor_simulation_row(
    *,
    mean_rg,
    p_val,
    dist_type,
    q_max,
    pixels,
    pair,
    tenor_settings=None,
    simulation_params=None,
):
    """Run the section-4 forward simulation for one p value and extract its observable."""
    params = {
        "mean_rg": float(mean_rg),
        "p_val": float(p_val),
        "dist_type": dist_type,
        "mode": "Sphere",
        "pixels": int(pixels),
        "q_min": 0.0,
        "q_max": float(q_max),
        "n_bins": 256,
        "smearing_x": 0.0,
        "smearing_y": 0.0,
        "flux": 1e8,
        "noise": False,
        "binning_mode": "Logarithmic",
        "radius_samples": int((tenor_settings or {}).get("radius_samples", 400)),
        "q_samples": int((tenor_settings or {}).get("q_samples", 200)),
        "form_factor_model": "Exact Sphere",
        "phi2": SOLID_SPHERE_PHI2,
        "phi3": 0.0,
        "ensemble_sampling": "Continuous",
        "ensemble_members": 11,
    }
    if simulation_params:
        params.update(dict(simulation_params))
        params["mean_rg"] = float(mean_rg)
        params["p_val"] = float(p_val)
        params["dist_type"] = dist_type
        params["mode"] = "Sphere"
        params["pixels"] = int(pixels)
        params["q_min"] = 0.0
        params["q_max"] = float(q_max)
        params["noise"] = False
    _, _, i_2d, _, _ = run_simulation_core(params)
    guinier = apparent_rg_from_2d(
        i_2d,
        q_max=q_max,
        initial_rg_guess=mean_rg,
        n_bins=int((tenor_settings or {}).get("tenor_guinier_bins", 256)),
    )
    if not guinier["valid"] or guinier["rg_app"] <= 0:
        return None
    obs = extract_pair_observables(
        i_2d,
        q_max=q_max,
        pair=pair,
        rg_app=guinier["rg_app"],
        qrg_limit=float((tenor_settings or {}).get("tenor_qrg_limit", 0.85)),
        n_radial_bins=int((tenor_settings or {}).get("tenor_radial_bins", 18)),
        psf_truncate=float((tenor_settings or {}).get("tenor_psf_truncate", 4.0)),
        use_m3=bool((tenor_settings or {}).get("tenor_use_m3", True)),
        use_g3=bool((tenor_settings or {}).get("tenor_use_g3", True)),
    )
    if obs is None:
        return None
    return {
        "p_val": float(p_val),
        "weighted_v": float(weighted_variance_from_p(p_val, dist_type)),
        "dimless_jg": float(obs["dimless_jg"]),
        "raw_g1_over_g0": float(obs["raw_g1_over_g0"]),
        "raw_g100_ratio": float(obs["raw_g100_ratio"]),
        "raw_g210_ratio": float(obs["raw_g210_ratio"]),
        "raw_m210_ratio": float(obs["raw_m210_ratio"]),
        "raw_m1_over_m0": float(obs["raw_m1_over_m0"]),
        "rg_app": float(guinier["rg_app"]),
        "g_app": float(guinier["g_app"]),
        "pair": pair,
        "candidate_count": 1,
    }


def analyze_tenor_saxs_2d(
    i_2d,
    q_max,
    dist_type,
    initial_rg_guess=4.0,
    psf_pairs=None,
    phi2=SOLID_SPHERE_PHI2,
    n_radial_bins=18,
    qrg_limit=0.85,
    guinier_bins=256,
    calibration_p_grid=None,
    psf_truncate=4.0,
    use_m3=True,
    use_g3=True,
    simulation_params_for_calibration=None,
    reconstruction_trials=1,
):
    """Estimate mean Rg and p from a 2D pattern using the TENOR-SAXS recipe."""
    i_2d = np.asarray(i_2d, dtype=float)
    if i_2d.ndim != 2 or i_2d.shape[0] != i_2d.shape[1]:
        raise ValueError("TENOR-SAXS expects a square 2D detector image.")

    if psf_pairs is None:
        psf_pairs = build_default_psf_pairs()

    # Section 4, step 2: perform a Guinier analysis of the 2D data after
    # reducing it to the same radial 1D profile used elsewhere in the app.
    guinier = apparent_rg_from_2d(
        i_2d,
        q_max=q_max,
        initial_rg_guess=initial_rg_guess,
        n_bins=guinier_bins,
    )
    rg_app = guinier["rg_app"]
    if not guinier["valid"] or rg_app <= 0:
        raise ValueError("Could not extract an apparent Guinier radius from the 2D data.")

    # Section 4, step 3(a-d):
    #   - scan candidate anisotropic PSF quartets,
    #   - digitally smear the image twice,
    #   - fit log-ratio maps to a(q) + b(q) cos(2 chi),
    #   - convert the fitted G(q) polynomial to the observable and keep the
    #     best-scoring quartet.
    candidates = []
    for pair in psf_pairs:
        obs = extract_pair_observables(
            i_2d=i_2d,
            q_max=q_max,
            pair=pair,
            rg_app=rg_app,
            qrg_limit=qrg_limit,
            n_radial_bins=n_radial_bins,
            psf_truncate=psf_truncate,
            use_m3=use_m3,
            use_g3=use_g3,
        )
        if obs is not None:
            candidates.append(obs)

    if not candidates:
        raise ValueError("TENOR-SAXS could not find a usable PSF pair for the supplied 2D data.")

    if calibration_p_grid is not None and len(calibration_p_grid) > 0:
        v_max_plausible = max(
            weighted_variance_from_p(float(np.max(calibration_p_grid)), dist_type),
            1e-6,
        )
    else:
        v_max_plausible = 0.8
    j_bounds = sorted(
        [
            dimless_jg_from_v(0.0, phi2=phi2),
            dimless_jg_from_v(v_max_plausible, phi2=phi2),
        ]
    )
    j_lo, j_hi = j_bounds[0], j_bounds[1]
    plausible_candidates = [
        item
        for item in candidates
        if np.isfinite(item["dimless_jg"]) and (j_lo <= item["dimless_jg"] <= j_hi)
    ]
    reconstruction_trials = max(int(reconstruction_trials), 1)
    candidate_subset = _select_reconstruction_candidates(
        plausible_candidates if plausible_candidates else candidates,
        j_lo=j_lo,
        j_hi=j_hi,
        reconstruction_trials=reconstruction_trials,
    )

    trial_rows = []
    best_eval = None
    for candidate in candidate_subset:
        eval_row = _evaluate_reconstruction_candidate(
            candidate=candidate,
            observed_i_2d=i_2d,
            q_max=q_max,
            dist_type=dist_type,
            initial_rg_guess=initial_rg_guess,
            calibration_p_grid=calibration_p_grid,
            guinier_bins=guinier_bins,
            qrg_limit=qrg_limit,
            n_radial_bins=n_radial_bins,
            psf_truncate=psf_truncate,
            use_m3=use_m3,
            use_g3=use_g3,
            simulation_params_for_calibration=simulation_params_for_calibration,
        )
        trial_rows.append(
            {
                "pair": candidate["pair"],
                "dimless_jg": float(candidate["dimless_jg"]),
                "score": float(candidate["score"]),
                "grade_g": float(candidate.get("grade_g", 0.0)),
                "recon_rrms_2d": float(eval_row["recon_rrms_2d"]),
                "p_rec": float(eval_row["p_rec"]),
                "mean_rg_rec": float(eval_row["mean_rg_rec"]),
            }
        )
        if best_eval is None or (
            eval_row["recon_rrms_2d"],
            -candidate.get("grade_g", 0.0),
            candidate["score"],
        ) < (
            best_eval["recon_rrms_2d"],
            -best_eval["candidate"].get("grade_g", 0.0),
            best_eval["candidate"]["score"],
        ):
            best_eval = eval_row

    if best_eval is None:
        raise ValueError("TENOR-SAXS could not evaluate any reconstruction candidates.")

    best = best_eval["candidate"]
    observable_dimless_jg = best["dimless_jg"]
    calibration = best_eval["calibration"]
    v_weighted = float(best_eval["weighted_v"])
    p_rec = float(best_eval["p_rec"])
    r0_weighted = float(best_eval["weighted_mean_rg"])
    mean_rg = float(best_eval["mean_rg_rec"])
    mean_radius = float(best_eval["mean_r_rec"])

    return {
        "method": "TENOR-SAXS",
        "dist_type": dist_type,
        "phi2": float(phi2),
        "rg_app": float(rg_app),
        "g_app": guinier["g_app"],
        "weighted_v": float(v_weighted),
        "p_rec": float(p_rec),
        "mean_rg_rec": float(mean_rg),
        "mean_r_rec": float(mean_radius),
        "weighted_mean_rg": float(r0_weighted),
        "observable_raw_g1_over_g0": float(best["raw_g1_over_g0"]),
        "observable_raw_g100_ratio": float(best["raw_g100_ratio"]),
        "observable_raw_g210_ratio": float(best["raw_g210_ratio"]),
        "observable_dimless_jg": float(observable_dimless_jg),
        "observable_raw_m210_ratio": float(best["raw_m210_ratio"]),
        "observable_raw_m1_over_m0": float(best["raw_m1_over_m0"]),
        "best_psf_pair": best["pair"],
        "best_g_rmse": float(best["g_rmse"]),
        "best_m_rmse": float(best["m_rmse"]),
        "best_g_coeffs": best["g_coeffs"],
        "best_m_coeffs": best["m_coeffs"],
        "q_fit_max": float(best["q_fit_max"]),
        "candidate_count": len(candidates),
        "candidate_plausible_count": len(plausible_candidates),
        "unstable_no_plausible_candidate": len(plausible_candidates) == 0,
        "candidate_reconstruction_count": len(candidate_subset),
        "candidate_trials": trial_rows,
        "best_recon_rrms_2d": float(best_eval["recon_rrms_2d"]),
        "best_recon_scale_2d": float(best_eval["recon_scale_2d"]),
        "calibration_rows": [] if calibration is None else calibration["rows"],
        "guinier_profile_q": guinier["q_1d"],
        "guinier_profile_i": guinier["i_1d"],
        "q_vals": best["q_vals"],
        "g_radial": best["g_radial"],
        "m_radial": best["m_radial"],
    }


def simulate_tenor_ground_truth(
    *,
    mean_rg,
    p_val,
    dist_type,
    q_max,
    pixels,
    pixel_size_um=None,
    sample_detector_distance_cm=None,
    wavelength_nm=None,
    psf_pairs=None,
    phi2=SOLID_SPHERE_PHI2,
    n_radial_bins=18,
    qrg_limit=0.85,
    guinier_bins=256,
    calibration_p_grid=None,
    psf_truncate=4.0,
    use_m3=True,
    use_g3=True,
    radius_samples=400,
    q_samples=200,
    form_factor_model="Exact Sphere",
    ensemble_sampling="Continuous",
    ensemble_members=11,
    simulation_params=None,
):
    """Generate TENOR-SAXS truth values from the current noise-free simulation details."""
    sim_params = {
        "mean_rg": float(mean_rg),
        "p_val": float(p_val),
        "dist_type": dist_type,
        "mode": "Sphere",
        "pixels": int(pixels),
        "pixel_size_um": pixel_size_um,
        "sample_detector_distance_cm": sample_detector_distance_cm,
        "wavelength_nm": wavelength_nm,
        "q_min": 0.0,
        "q_max": float(q_max),
        "n_bins": max(int(guinier_bins), 64),
        "smearing_x": 0.0,
        "smearing_y": 0.0,
        "flux": 1e8,
        "noise": False,
        "binning_mode": "Linear",
        "radius_samples": int(radius_samples),
        "q_samples": int(q_samples),
        "form_factor_model": form_factor_model,
        "phi2": float(phi2),
        "phi3": 0.0,
        "ensemble_sampling": ensemble_sampling,
        "ensemble_members": int(ensemble_members),
    }
    if simulation_params:
        sim_params.update(dict(simulation_params))
        sim_params["mean_rg"] = float(mean_rg)
        sim_params["p_val"] = float(p_val)
        sim_params["dist_type"] = dist_type
        sim_params["mode"] = "Sphere"
        sim_params["pixels"] = int(pixels)
        sim_params["q_min"] = 0.0
        sim_params["q_max"] = float(q_max)
        sim_params["noise"] = False
    _, _, i_2d_truth, _, _ = run_simulation_core(sim_params)
    truth = analyze_tenor_saxs_2d(
        i_2d=i_2d_truth,
        q_max=float(q_max),
        dist_type=dist_type,
        initial_rg_guess=float(mean_rg),
        psf_pairs=psf_pairs,
        phi2=float(phi2),
        n_radial_bins=int(n_radial_bins),
        qrg_limit=float(qrg_limit),
        guinier_bins=int(guinier_bins),
        calibration_p_grid=calibration_p_grid,
        psf_truncate=float(psf_truncate),
        use_m3=bool(use_m3),
        use_g3=bool(use_g3),
        simulation_params_for_calibration=sim_params,
    )
    truth["weighted_v_true_from_input"] = float(weighted_variance_from_p(p_val, dist_type))
    truth["p_true_from_input"] = float(p_val)
    truth["mean_rg_true_from_input"] = float(mean_rg)
    truth["mean_r_true_from_input"] = float(mean_rg * math.sqrt(5.0 / 3.0))
    return truth
