"""Shared simulation and analysis settings defaults for the app."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SETTINGS_FILE = Path(__file__).resolve().parent / "app_settings.json"


DEFAULT_APP_SETTINGS = {
    "sim_mode": "Polydisperse Spheres",
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


PARAMETER_DESCRIPTIONS = {
    "sim_mode": "Top-level simulation family used by the GUI and analysis pipeline. Allowed values: 'Polydisperse Spheres', 'Polydisperse IDP'.",
    "mean_rg": "Mean radius of gyration of the simulated ensemble, in the app's working length units.",
    "p_val": "Polydispersity parameter used to define the width of the selected size distribution.",
    "dist_type": "Probability distribution family used to generate the ensemble. Allowed values: 'Gaussian', 'Lognormal', 'Schulz', 'Boltzmann', 'Triangular', 'Uniform'.",
    "analysis_method": "Default analysis method used to recover size and polydispersity from the simulated data. Allowed values: 'NNLS', 'Tomchuk', 'Tenor'.",
    "nnls_max_rg": "Largest radius of gyration included in the NNLS basis grid.",
    "nnls_basis_count": "Number of basis functions used in the NNLS inversion grid.",
    "nnls_smooth_sigma": "Gaussian smoothing width applied to the recovered NNLS distribution.",
    "pixels": "Number of detector pixels along one image dimension for the simulated 2D pattern.",
    "q_min": "Minimum q value included in the generated or analyzed 1D profile.",
    "q_max": "Maximum q value included in the generated or analyzed 1D profile.",
    "n_bins": "Number of q bins used when reducing the detector image to a 1D intensity profile.",
    "binning_mode": "Whether the 1D q bins are spaced linearly or logarithmically. Allowed values: 'Linear', 'Logarithmic'.",
    "smearing_x": "Gaussian detector smearing width along the horizontal detector axis, in pixels.",
    "smearing_y": "Gaussian detector smearing width along the vertical detector axis, in pixels.",
    "flux_pre": "Mantissa used in scientific notation for the central-pixel flux target.",
    "flux_exp": "Base-10 exponent used in scientific notation for the central-pixel flux target.",
    "optimal_flux": "Whether the app should override the user flux and choose an internally optimized flux value.",
    "add_noise": "Whether Poisson counting noise is added to the simulated detector data.",
    "radius_samples": "Number of radius or Rg support points used to build continuous ensemble distributions.",
    "q_samples": "Number of q samples used internally when evaluating model form factors before detector binning.",
    "form_factor_model": "Forward model used for the particle form factor. Allowed values: 'Exact Sphere', 'Guinier Curvature', 'Exact Gaussian Chain', 'Exact Shell', 'Exact Thin Rod', 'Exact Thin Disk'.",
    "phi2": "Second-order correction coefficient used by the Guinier-curvature forward model.",
    "phi3": "Third-order correction coefficient used by the Guinier-curvature forward model.",
    "weight_power": "Additional radius-power weighting applied during ensemble averaging for selected forward models.",
    "ensemble_sampling": "Whether the ensemble is integrated continuously or approximated with a finite number of members. Allowed values: 'Continuous', 'Discrete'.",
    "ensemble_members": "Number of discrete ensemble members used when finite ensemble sampling is selected.",
    "tenor_guinier_bins": "Number of radial bins used when estimating the apparent Guinier radius for Tenor-SAXS.",
    "tenor_radial_bins": "Number of radial sectors used in the Tenor-SAXS observable extraction workflow.",
    "tenor_qrg_limit": "Upper qRg cutoff used when selecting the apparent Guinier fitting region for Tenor-SAXS.",
    "tenor_psf_count": "Number of PSF quartets scanned when searching for a stable Tenor-SAXS observable extraction.",
    "tenor_psf_truncate": "Gaussian kernel truncation radius, in sigma units, for digitally applied Tenor PSFs.",
    "tenor_psf_sigma_x_start": "Starting horizontal PSF sigma for the Tenor-SAXS PSF scan, in pixels.",
    "tenor_psf_sigma_y_start": "Starting vertical PSF sigma for the Tenor-SAXS PSF scan, in pixels.",
    "tenor_psf_sigma_step": "Increment applied between successive Tenor-SAXS PSF scan candidates, in pixels.",
    "tenor_psf_secondary_ratio": "Relative width factor used to define the secondary PSF in each Tenor-SAXS quartet.",
    "tenor_use_g3": "Whether the Tenor-SAXS G(Q) fit includes the cubic-in-Q term used in the MATLAB workflow.",
    "tenor_use_m3": "Whether the Tenor-SAXS M(Q) fit includes the cubic-in-Q term used in the MATLAB workflow.",
    "tenor_calibration_p_min": "Minimum p value included in the Tenor-SAXS simulation calibration grid.",
    "tenor_calibration_p_max": "Maximum p value included in the Tenor-SAXS simulation calibration grid.",
    "tenor_calibration_p_count": "Number of p values sampled in the Tenor-SAXS simulation calibration grid.",
    "tomchuk_target_abs_error": "Target absolute fitting tolerance used by the Tomchuk recovery procedure.",
}


def _build_settings_payload(settings_map):
    settings_block = {}
    for key, default_value in DEFAULT_APP_SETTINGS.items():
        value = settings_map.get(key, default_value)
        if isinstance(value, np.generic):
            value = value.item()
        settings_block[key] = {
            "value": value,
            "_comment": PARAMETER_DESCRIPTIONS.get(key, ""),
        }
    return {
        "_format": "app_settings_v2",
        "_comment": "Persisted GUI defaults. Each entry stores the startup value and a short explanation of its role.",
        "settings": settings_block,
    }


def load_persisted_settings():
    if not SETTINGS_FILE.exists():
        return {}
    try:
        payload = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    settings_block = payload.get("settings")
    if isinstance(settings_block, dict):
        parsed = {}
        for key in DEFAULT_APP_SETTINGS:
            entry = settings_block.get(key)
            if isinstance(entry, dict) and "value" in entry:
                parsed[key] = entry["value"]
            elif key in settings_block:
                parsed[key] = settings_block[key]
        return parsed
    return {key: payload[key] for key in DEFAULT_APP_SETTINGS if key in payload}


def ensure_session_state_defaults(session_state):
    persisted = load_persisted_settings()
    for key, value in DEFAULT_APP_SETTINGS.items():
        if key not in session_state:
            session_state[key] = persisted.get(key, value)


def persist_app_settings(session_state):
    payload = _build_settings_payload(session_state)
    SETTINGS_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def get_forward_flux(settings_like):
    return float(settings_like["flux_pre"]) * (10 ** int(settings_like["flux_exp"]))


def build_tenor_p_grid(settings_like):
    p_min = max(float(settings_like["tenor_calibration_p_min"]), 1e-4)
    p_max = max(float(settings_like["tenor_calibration_p_max"]), p_min + 1e-4)
    p_count = max(int(settings_like["tenor_calibration_p_count"]), 4)
    return np.linspace(p_min, p_max, p_count)
