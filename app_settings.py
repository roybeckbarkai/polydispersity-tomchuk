"""Shared simulation and analysis settings defaults for the app."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SETTINGS_FILE = Path(__file__).resolve().parent / "app_settings.json"

EXCLUDED_SESSION_KEYS = {
    "page",
    "batch_df",
    "tomchuk_recommendation",
    "q_samples_sensitivity_df",
    "q_samples_sensitivity_ref",
    "uploaded_batch",
}

TRANSIENT_WIDGET_KEYS = {
    "run_q_samples_sensitivity",
    "eval_tomchuk_grid",
}


DEFAULT_APP_SETTINGS = {
    "sim_mode": "Polydisperse Spheres",
    "mean_rg": 4.0,
    "p_val": 0.5477225575051661,
    "dist_type": "Lognormal",
    "analysis_method": "Tenor",
    "nnls_max_rg": 30.0,
    "nnls_basis_count": 150,
    "nnls_smooth_sigma": 1.0,
    "pixels": 500,
    "pixel_size_um": 70.0,
    "sample_detector_distance_cm": 360.0,
    "wavelength_nm": 0.1,
    "q_min": 0.0,
    "q_max": 0.6096435077216193,
    "n_bins": 512,
    "binning_mode": "Logarithmic",
    "smearing_x": 3.0,
    "smearing_y": 15.0,
    "flux_pre": 1.0,
    "flux_exp": 5,
    "optimal_flux": False,
    "add_noise": True,
    "radius_samples": 50,
    "q_samples": 1000,
    "form_factor_model": "Exact Sphere",
    "phi2": -1.0 / 63.0,
    "phi3": 0.0,
    "ensemble_sampling": "Discrete",
    "ensemble_members": 41,
    "tenor_radial_bins": 18,
    "tenor_qrg_limit": 0.85,
    "tenor_psf_count": 5,
    "tenor_psf_truncate": 4.0,
    "tenor_psf_sigma_x_start": 1.2,
    "tenor_psf_sigma_y_start": 0.6,
    "tenor_psf_sigma_step": 0.4,
    "tenor_psf_secondary_ratio": 0.5,
    "tenor_use_g3": False,
    "tenor_use_m3": False,
    "tenor_reconstruction_trials": 5,
    "tenor_calibration_p_min": 0.01,
    "tenor_calibration_p_max": 1.0,
    "tenor_calibration_p_count": 15,
    "tomchuk_target_abs_error": 0.001,
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
    "pixel_size_um": "Physical detector pixel dimension in micrometers. The total detector width is pixels multiplied by this value.",
    "sample_detector_distance_cm": "Sample-to-detector distance used to convert detector position into q, in centimeters.",
    "wavelength_nm": "Beam wavelength used to convert detector position into q, in nanometers.",
    "q_min": "Minimum q value included in the generated or analyzed 1D profile.",
    "q_max": "Maximum q value used by the current analysis target. In the single-run GUI this is derived automatically from the actual data: detector-limited q_max for simulated data, or the last q point for uploaded 1D data.",
    "n_bins": "Number of q bins used when reducing the detector image to a 1D intensity profile.",
    "binning_mode": "Whether the 1D q bins are spaced linearly or logarithmically. Allowed values: 'Linear', 'Logarithmic'.",
    "smearing_x": "Gaussian detector smearing width along the horizontal detector axis, in pixels.",
    "smearing_y": "Gaussian detector smearing width along the vertical detector axis, in pixels.",
    "flux_pre": "Mantissa used in scientific notation for the central-pixel flux target.",
    "flux_exp": "Base-10 exponent used in scientific notation for the central-pixel flux target.",
    "optimal_flux": "Whether the app should override the user flux and choose an internally optimized flux value.",
    "add_noise": "Whether Poisson counting noise is added to the simulated detector data.",
    "radius_samples": "Number of radius or Rg support points used to build the size grid before simulation. This matters mainly for Continuous ensemble sampling, where the distribution is integrated across the full grid. In Discrete sampling, this still defines the candidate grid, but the simulator keeps only ensemble_members representative points.",
    "q_samples": "Number of internal q samples used when evaluating the forward model before interpolation onto the detector. Increase this when the form factor has structure or oscillations, or when you want smoother detector interpolation. It matters much less for low-q Guinier-only work than for broad-q exact form-factor simulations.",
    "form_factor_model": "Forward model used for the particle form factor. Allowed values: 'Exact Sphere', 'Guinier Curvature', 'Exact Gaussian Chain', 'Exact Shell', 'Exact Thin Rod', 'Exact Thin Disk'.",
    "phi2": "Second-order curvature coefficient used only by the Guinier Curvature forward model. It is ignored by Exact Sphere, Exact Gaussian Chain, Exact Shell, Exact Thin Rod, and Exact Thin Disk, because those models already define their own full q dependence.",
    "phi3": "Third-order curvature coefficient used only by the Guinier Curvature forward model. It is ignored by the exact form-factor models and can usually stay at zero unless you are intentionally using the truncated Guinier-curvature expansion.",
    "ensemble_sampling": "Whether the ensemble is integrated continuously or approximated with a finite number of members. Allowed values: 'Continuous', 'Discrete'.",
    "ensemble_members": "Number of representative sizes kept when Ensemble Sampling is set to Discrete. This value is ignored in Continuous mode, where the simulator integrates across the full radius_samples grid instead of selecting a finite member list.",
    "tenor_radial_bins": "Number of radial sectors used in the Tenor-SAXS observable extraction workflow.",
    "tenor_qrg_limit": "Upper qRg cutoff used when selecting the apparent Guinier fitting region for Tenor-SAXS.",
    "tenor_psf_count": "Number of PSF quartets scanned when searching for a stable Tenor-SAXS observable extraction.",
    "tenor_psf_truncate": "Gaussian kernel truncation radius, in sigma units, for digitally applied Tenor PSFs.",
    "tenor_psf_sigma_x_start": "Starting horizontal PSF sigma for the Tenor-SAXS PSF scan, in pixels.",
    "tenor_psf_sigma_y_start": "Starting vertical PSF sigma for the Tenor-SAXS PSF scan, in pixels.",
    "tenor_psf_sigma_step": "Increment applied between successive Tenor-SAXS PSF scan candidates, in pixels.",
    "tenor_psf_secondary_ratio": "Relative width factor used to define the secondary PSF in each Tenor-SAXS quartet.",
    "tenor_use_g3": "Whether the Tenor-SAXS G(Q) fit includes a cubic-in-Q term. The MATLAB working3_26 example keeps this off, so the default is False.",
    "tenor_use_m3": "Whether the Tenor-SAXS M(Q) fit includes a cubic-in-Q term. The MATLAB working3_26 example keeps this off, so the default is False.",
    "tenor_reconstruction_trials": "Number of candidate TENOR quartets that are carried into the reconstruction-validation stage. The code ranks quartets by physical plausibility and fit quality, then re-simulates these candidates and keeps the one whose reconstructed 2D data best matches the raw detector image.",
    "tenor_calibration_p_min": "Minimum p value included in the Tenor-SAXS simulation calibration grid.",
    "tenor_calibration_p_max": "Maximum p value included in the Tenor-SAXS simulation calibration grid.",
    "tenor_calibration_p_count": "Number of p values sampled in the Tenor-SAXS simulation calibration grid.",
    "tomchuk_target_abs_error": "Target absolute fitting tolerance used by the Tomchuk recovery procedure.",
}


def _is_persistable_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return True
    if isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_persistable_value(item) for item in value)
    return False


def _normalize_persistable_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_normalize_persistable_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_persistable_value(item) for item in value]
    return value


def _iter_persisted_keys(settings_map):
    keys = list(DEFAULT_APP_SETTINGS.keys())
    extra_keys = []
    for key in settings_map.keys():
        if key in DEFAULT_APP_SETTINGS:
            continue
        if key in EXCLUDED_SESSION_KEYS or key in TRANSIENT_WIDGET_KEYS or str(key).startswith("_"):
            continue
        value = settings_map.get(key)
        if _is_persistable_value(value):
            extra_keys.append(str(key))
    keys.extend(sorted(extra_keys))
    return keys


def _build_settings_payload(settings_map):
    settings_block = {}
    for key in _iter_persisted_keys(settings_map):
        default_value = DEFAULT_APP_SETTINGS.get(key)
        value = settings_map.get(key, default_value)
        value = _normalize_persistable_value(value)
        settings_block[key] = {
            "value": value,
            "_comment": PARAMETER_DESCRIPTIONS.get(key, "Persisted Streamlit widget value."),
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
        for key in settings_block:
            entry = settings_block.get(key)
            if isinstance(entry, dict) and "value" in entry:
                parsed[key] = entry["value"]
            elif key in settings_block:
                parsed[key] = settings_block[key]
        if "pixel_size_um" not in parsed and "detector_side_cm" in settings_block:
            detector_entry = settings_block.get("detector_side_cm")
            detector_side_cm = detector_entry.get("value") if isinstance(detector_entry, dict) else detector_entry
            pixels = parsed.get("pixels", DEFAULT_APP_SETTINGS["pixels"])
            try:
                parsed["pixel_size_um"] = float(detector_side_cm) * 1.0e4 / max(float(pixels), 1.0)
            except Exception:
                pass
        return parsed
    parsed = {key: payload[key] for key in payload if key not in {"_format", "_comment"}}
    if "pixel_size_um" not in parsed and "detector_side_cm" in payload:
        pixels = parsed.get("pixels", DEFAULT_APP_SETTINGS["pixels"])
        try:
            parsed["pixel_size_um"] = float(payload["detector_side_cm"]) * 1.0e4 / max(float(pixels), 1.0)
        except Exception:
            pass
    return parsed


def hydrate_session_state_from_disk(session_state):
    persisted = load_persisted_settings()
    for transient_key in TRANSIENT_WIDGET_KEYS:
        persisted.pop(transient_key, None)
        session_state.pop(transient_key, None)
    for key, value in DEFAULT_APP_SETTINGS.items():
        session_state[key] = persisted.get(key, value)
    for key, value in persisted.items():
        if key not in EXCLUDED_SESSION_KEYS and key not in TRANSIENT_WIDGET_KEYS and not str(key).startswith("_"):
            session_state[key] = value
    session_state["_settings_initialized_from_disk"] = True


def ensure_session_state_defaults(session_state):
    if not session_state.get("_settings_initialized_from_disk", False):
        hydrate_session_state_from_disk(session_state)
        return

    persisted = load_persisted_settings()
    for transient_key in TRANSIENT_WIDGET_KEYS:
        persisted.pop(transient_key, None)
        session_state.pop(transient_key, None)
    for key, value in DEFAULT_APP_SETTINGS.items():
        if key not in session_state:
            session_state[key] = persisted.get(key, value)
    for key, value in persisted.items():
        if key not in session_state and key not in EXCLUDED_SESSION_KEYS and key not in TRANSIENT_WIDGET_KEYS and not str(key).startswith("_"):
            session_state[key] = value


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
