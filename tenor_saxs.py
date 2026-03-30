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
from scipy.ndimage import gaussian_filter
from scipy.optimize import bisect

from analysis_utils import estimate_guinier_parameters, get_normalized_moment
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
    _, _, _, q_r = build_detector_q_grid(pixels, q_max)
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


def fit_angular_harmonics(log_ratio_map, intensity_weights, qx, qy, q_edges):
    """Fit a(q) + b(q) cos(2 chi) in radial bins."""
    q_r = np.sqrt(qx**2 + qy**2)
    chi = np.arctan2(qy, qx)
    cos2 = np.cos(2.0 * chi)

    q_centers = []
    a_vals = []
    b_vals = []
    bin_weights = []

    for q_lo, q_hi in zip(q_edges[:-1], q_edges[1:]):
        mask = (q_r >= q_lo) & (q_r < q_hi) & np.isfinite(log_ratio_map) & np.isfinite(intensity_weights)
        if np.count_nonzero(mask) < 30:
            continue

        y = log_ratio_map[mask]
        x = cos2[mask]
        w = np.sqrt(np.maximum(intensity_weights[mask], 1e-12))
        if np.count_nonzero(np.isfinite(y)) < 30:
            continue

        design = np.column_stack([np.ones_like(x), x])
        design_w = design * w[:, None]
        y_w = y * w
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design_w, y_w, rcond=None)
        except np.linalg.LinAlgError:
            continue

        q_centers.append(0.5 * (q_lo + q_hi))
        a_vals.append(float(coeffs[0]))
        b_vals.append(float(coeffs[1]))
        bin_weights.append(float(np.sum(w**2)))

    return (
        np.asarray(q_centers, dtype=float),
        np.asarray(a_vals, dtype=float),
        np.asarray(b_vals, dtype=float),
        np.asarray(bin_weights, dtype=float),
    )


def fit_even_polynomial(q_vals, y_vals, weights=None, degree=2):
    """Fit y(q) = c0 + c1 q^2 + c2 q^4 + ..."""
    q_vals = np.asarray(q_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    if len(q_vals) < degree + 2:
        raise ValueError("Not enough points to fit TENOR polynomial.")

    x = q_vals**2
    poly = np.polyfit(x, y_vals, deg=degree, w=weights)
    # np.polyfit returns highest-to-lowest order.
    fitted = np.polyval(poly, x)
    resid = y_vals - fitted
    rmse = float(np.sqrt(np.mean(resid**2)))
    coeffs = poly[::-1]
    return coeffs, fitted, rmse


def extract_pair_observables(i_2d, q_max, pair, rg_app, qrg_limit=0.85, n_radial_bins=18):
    pixels = i_2d.shape[0]
    dq = (2.0 * q_max) / max(pixels - 1, 1)
    _, qx, qy, _ = build_detector_q_grid(pixels, q_max)
    q_fit_max = min(q_max, qrg_limit / max(rg_app, 1e-6))
    q_edges = np.linspace(dq, q_fit_max, int(n_radial_bins) + 1)
    base_weights = np.maximum(i_2d, np.nanmax(i_2d) * 1e-12 if np.isfinite(i_2d).any() else 1e-12)

    img1 = apply_anisotropic_gaussian(i_2d, pair.sigma_x_1, pair.sigma_y_1)
    img2 = apply_anisotropic_gaussian(i_2d, pair.sigma_x_2, pair.sigma_y_2)
    log_ratio = np.log(np.maximum(img1, 1e-12)) - np.log(np.maximum(img2, 1e-12))

    q_vals, a_vals, b_vals, fit_weights = fit_angular_harmonics(
        log_ratio_map=log_ratio,
        intensity_weights=base_weights,
        qx=qx,
        qy=qy,
        q_edges=q_edges,
    )
    if len(q_vals) < 6:
        return None

    dsx2 = (pair.sigma_x_1**2 - pair.sigma_x_2**2) * (dq**2)
    dsy2 = (pair.sigma_y_1**2 - pair.sigma_y_2**2) * (dq**2)
    sigma0 = 0.5 * (dsx2 + dsy2)
    delta0 = 0.5 * (dsx2 - dsy2)
    if abs(sigma0) < 1e-20 or abs(delta0) < 1e-20:
        return None

    g_radial = 2.0 * a_vals / sigma0
    m_radial = 2.0 * b_vals / delta0
    g_coeffs, _, g_rmse = fit_even_polynomial(q_vals, g_radial, weights=fit_weights, degree=2)
    m_coeffs, _, m_rmse = fit_even_polynomial(q_vals, m_radial, weights=fit_weights, degree=2)

    g0 = float(g_coeffs[0])
    g1 = float(g_coeffs[1]) if len(g_coeffs) > 1 else 0.0
    m0 = float(m_coeffs[0])
    m1 = float(m_coeffs[1]) if len(m_coeffs) > 1 else 0.0
    if abs(g0) < 1e-20:
        return None

    raw_g1_over_g0 = g1 / g0
    score = abs(g_rmse) + 0.5 * abs(m_rmse)
    return {
        "pair": pair,
        "q_vals": q_vals,
        "g_radial": g_radial,
        "m_radial": m_radial,
        "g_coeffs": g_coeffs,
        "m_coeffs": m_coeffs,
        "g_rmse": g_rmse,
        "m_rmse": m_rmse,
        "raw_g1_over_g0": float(raw_g1_over_g0),
        "raw_m1_over_m0": float(m1 / m0) if abs(m0) > 1e-20 else np.nan,
        "dimless_jg": float(raw_g1_over_g0 / max(rg_app**2, 1e-12)),
        "score": float(score),
        "q_fit_max": float(q_fit_max),
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
        params = {
            "mean_rg": float(mean_rg_ref),
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
        }
        if tenor_settings:
            params["radius_samples"] = int(tenor_settings.get("radius_samples", 400))
            params["q_samples"] = int(tenor_settings.get("q_samples", 200))
        _, _, i_2d, _, _ = run_simulation_core(params)
        guinier = apparent_rg_from_2d(
            i_2d,
            q_max=q_max,
            initial_rg_guess=mean_rg_ref,
            n_bins=int((tenor_settings or {}).get("tenor_guinier_bins", 256)),
        )
        if not guinier["valid"] or guinier["rg_app"] <= 0:
            continue
        obs = extract_pair_observables(
            i_2d,
            q_max=q_max,
            pair=pair,
            rg_app=guinier["rg_app"],
            qrg_limit=float((tenor_settings or {}).get("tenor_qrg_limit", 0.85)),
            n_radial_bins=int((tenor_settings or {}).get("tenor_radial_bins", 18)),
        )
        if obs is None:
            continue
        rows.append(
            {
                "p_val": float(p_val),
                "weighted_v": float(weighted_variance_from_p(p_val, dist_type)),
                "dimless_jg": float(obs["dimless_jg"]),
            }
        )

    if len(rows) < 4:
        return None

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
):
    """Estimate mean Rg and p from a 2D pattern using the TENOR-SAXS recipe."""
    i_2d = np.asarray(i_2d, dtype=float)
    if i_2d.ndim != 2 or i_2d.shape[0] != i_2d.shape[1]:
        raise ValueError("TENOR-SAXS expects a square 2D detector image.")

    if psf_pairs is None:
        psf_pairs = build_default_psf_pairs()

    guinier = apparent_rg_from_2d(
        i_2d,
        q_max=q_max,
        initial_rg_guess=initial_rg_guess,
        n_bins=guinier_bins,
    )
    rg_app = guinier["rg_app"]
    if not guinier["valid"] or rg_app <= 0:
        raise ValueError("Could not extract an apparent Guinier radius from the 2D data.")

    candidates = []
    for pair in psf_pairs:
        obs = extract_pair_observables(
            i_2d=i_2d,
            q_max=q_max,
            pair=pair,
            rg_app=rg_app,
            qrg_limit=qrg_limit,
            n_radial_bins=n_radial_bins,
        )
        if obs is not None:
            candidates.append(obs)

    if not candidates:
        raise ValueError("TENOR-SAXS could not find a usable PSF pair for the supplied 2D data.")

    best = min(candidates, key=lambda item: item["score"])
    observable_dimless_jg = best["dimless_jg"]
    calibration = calibrate_p_from_simulation(
        target_j_g=observable_dimless_jg,
        dist_type=dist_type,
        q_max=q_max,
        pixels=i_2d.shape[0],
        pair=best["pair"],
        mean_rg_ref=initial_rg_guess,
        p_grid=calibration_p_grid,
        tenor_settings={
            "tenor_guinier_bins": guinier_bins,
            "tenor_qrg_limit": qrg_limit,
            "tenor_radial_bins": n_radial_bins,
        },
    )
    if calibration is None:
        v_weighted = solve_v_from_j_g(observable_dimless_jg, phi2=phi2)
        p_rec = solve_p_from_weighted_v(v_weighted, dist_type=dist_type)
    else:
        v_weighted = float(calibration["weighted_v"])
        p_rec = float(calibration["p_rec"])
    r0_weighted = rg_app / math.sqrt(max(1.0 + v_weighted, 1e-12))
    mean_ratio = weighted_mean_to_arithmetic_mean_ratio(p_rec, dist_type)
    mean_rg = r0_weighted / mean_ratio if mean_ratio > 0 else r0_weighted
    mean_radius = mean_rg * math.sqrt(5.0 / 3.0)

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
        "observable_dimless_jg": float(observable_dimless_jg),
        "observable_raw_m1_over_m0": float(best["raw_m1_over_m0"]),
        "best_psf_pair": best["pair"],
        "best_g_rmse": float(best["g_rmse"]),
        "best_m_rmse": float(best["m_rmse"]),
        "best_g_coeffs": best["g_coeffs"],
        "best_m_coeffs": best["m_coeffs"],
        "q_fit_max": float(best["q_fit_max"]),
        "candidate_count": len(candidates),
        "calibration_rows": [] if calibration is None else calibration["rows"],
        "guinier_profile_q": guinier["q_1d"],
        "guinier_profile_i": guinier["i_1d"],
        "q_vals": best["q_vals"],
        "g_radial": best["g_radial"],
        "m_radial": best["m_radial"],
    }
