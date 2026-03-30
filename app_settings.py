"""Shared simulation and analysis settings defaults for the app."""

from __future__ import annotations

import numpy as np


DEFAULT_APP_SETTINGS = {
    "mean_rg": 2.0,
    "p_val": 0.3,
    "dist_type": "Gaussian",
    "analysis_method": "NNLS",
    "nnls_max_rg": 30.0,
    "nnls_basis_count": 150,
    "nnls_smooth_sigma": 1.0,
    "pixels": 1000,
    "q_min": 0.0,
    "q_max": 5.0,
    "n_bins": 256,
    "binning_mode": "Logarithmic",
    "smearing_x": 3.0,
    "smearing_y": 3.0,
    "flux_pre": 1.0,
    "flux_exp": 8,
    "optimal_flux": False,
    "add_noise": True,
    "radius_samples": 400,
    "q_samples": 200,
    "form_factor_model": "Exact Sphere",
    "phi2": -1.0 / 63.0,
    "phi3": 0.0,
    "weight_power": 6.0,
    "ensemble_sampling": "Continuous",
    "ensemble_members": 11,
    "tenor_guinier_bins": 256,
    "tenor_radial_bins": 18,
    "tenor_qrg_limit": 0.85,
    "tenor_psf_count": 5,
    "tenor_psf_truncate": 4.0,
    "tenor_psf_sigma_x_start": 1.2,
    "tenor_psf_sigma_y_start": 0.6,
    "tenor_psf_sigma_step": 0.4,
    "tenor_psf_secondary_ratio": 0.5,
    "tenor_use_g3": True,
    "tenor_use_m3": True,
    "tenor_calibration_p_min": 0.01,
    "tenor_calibration_p_max": 0.6,
    "tenor_calibration_p_count": 18,
    "tomchuk_target_abs_error": 0.01,
}


def ensure_session_state_defaults(session_state):
    for key, value in DEFAULT_APP_SETTINGS.items():
        if key not in session_state:
            session_state[key] = value


def get_forward_flux(settings_like):
    return float(settings_like["flux_pre"]) * (10 ** int(settings_like["flux_exp"]))


def build_tenor_p_grid(settings_like):
    p_min = max(float(settings_like["tenor_calibration_p_min"]), 1e-4)
    p_max = max(float(settings_like["tenor_calibration_p_max"]), p_min + 1e-4)
    p_count = max(int(settings_like["tenor_calibration_p_count"]), 4)
    return np.linspace(p_min, p_max, p_count)
