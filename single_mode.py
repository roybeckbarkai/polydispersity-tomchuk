# File: single_mode.py
# Last Updated: Monday, March 30, 2026
# Description: Handling single, interactive simulation runs.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_settings import ensure_session_state_defaults, get_forward_flux
from analysis_utils import (
    build_reconstruction_quality_summary,
    build_sanity_summary_row,
    calculate_sphere_input_theoretical_parameters,
    create_distribution_csv,
    create_intensity_csv,
    get_header_string,
    normalize_simulated_sphere_intensity,
    parse_saxs_file,
    perform_saxs_analysis,
    recommend_tomchuk_settings,
)
from sim_utils import build_detector_q_grid, get_detector_q_max, get_distribution, run_simulation_core


PARAM_HELP = {
    "sim_mode": "Choose the physical sample family. The left panel defines the sample you are simulating or analyzing. 'Polydisperse Spheres' uses particle radius statistics, while 'Fixed-Length Polymers (IDP)' uses polymer Rg statistics.",
    "analysis_method": "Choose which recovery method to run on the current curve. Tomchuk and Tenor are sphere-focused methods, while NNLS performs a non-negative basis reconstruction.",
    "input_r": "Arithmetic mean particle radius used as the sphere sample input. The app converts this to the corresponding mean radius of gyration internally.",
    "input_rg": "Arithmetic mean radius of gyration used as the sample input for polymer-like models.",
    "p_val": "Relative width of the chosen distribution. Larger p means a broader ensemble and stronger polydispersity.",
    "dist_type": "Probability distribution family used for the input ensemble. This controls how particle sizes or conformations are distributed around the mean.",
    "flux_pre": "Scientific-notation mantissa for the photon flux target. Together with the exponent, this sets the expected counts in the nearest-to-center detector pixel after smearing.",
    "flux_exp": "Scientific-notation exponent for the photon flux target.",
    "pixels": "Number of detector pixels along one detector side. The detector is simulated as a square pixels x pixels image.",
    "pixel_size_um": "Physical size of one detector pixel in micrometers. Together with detector distance and wavelength, this determines the q sampling on the detector.",
    "sample_detector_distance_cm": "Distance from sample to detector in centimeters. Larger distance reduces the q range covered by a fixed detector.",
    "wavelength_nm": "Beam wavelength in nanometers. This sets how detector angles convert into scattering vector q.",
    "q_min": "Lowest q kept when reducing the 2D detector image to a 1D curve.",
    "q_max": "Highest q kept for the 1D profile and analysis. It cannot exceed the instrument-derived detector limit.",
    "n_bins": "Number of bins used to radially average the 2D detector into a 1D profile.",
    "binning_mode": "Choose whether 1D q bins are spaced linearly or logarithmically.",
    "smearing_x": "Gaussian instrumental smearing width along the detector x axis, in pixels.",
    "smearing_y": "Gaussian instrumental smearing width along the detector y axis, in pixels.",
    "optimal_flux": "If enabled, the simulation stays deterministic and Poisson counting noise is disabled.",
    "add_noise": "If enabled, Poisson counting noise is applied after scaling the detector to the requested flux.",
    "radius_samples": "Number of support points used to represent the underlying size distribution during simulation.",
    "q_samples": "Number of internal q points used before mapping the intensity onto the detector grid.",
    "ensemble_members": "Number of discrete ensemble members used when discrete sampling is selected.",
    "form_factor_model": "Scattering kernel used to simulate the chosen sample family. The left panel defines what sample is present, and the forward model defines how that sample scatters.",
    "ensemble_sampling": "Choose whether the distribution is treated continuously or approximated by a finite number of representative members.",
    "phi2": "Second-order curvature coefficient used by the Guinier-curvature forward model.",
    "phi3": "Third-order curvature coefficient used by the Guinier-curvature forward model.",
    "nnls_basis_count": "Number of basis functions used in the NNLS inversion.",
    "nnls_smooth_sigma": "Smoothing strength applied to the NNLS recovered distribution.",
    "nnls_max_rg": "Largest Rg value included in the NNLS basis set.",
    "tomchuk_target_abs_error": "Target absolute error in recovered p used by the Tomchuk q-range evaluator.",
    "tenor_guinier_bins": "Number of radial bins used for the apparent Guinier estimate in Tenor-SAXS.",
    "tenor_radial_bins": "Number of angular/radial sectors used in Tenor-SAXS observable extraction.",
    "tenor_qrg_limit": "Upper qRg limit used when selecting the low-q region for Tenor fitting.",
    "tenor_psf_count": "Number of PSF quartets tested in the Tenor-SAXS scan.",
    "tenor_psf_sigma_x_start": "Initial sigma in x for the digital PSF scan used by Tenor-SAXS.",
    "tenor_psf_sigma_y_start": "Initial sigma in y for the digital PSF scan used by Tenor-SAXS.",
    "tenor_psf_sigma_step": "Increment between successive PSF candidates in the Tenor-SAXS scan.",
    "tenor_psf_secondary_ratio": "Ratio between the primary and secondary PSF widths inside each Tenor pair.",
    "tenor_psf_truncate": "Kernel truncation radius, in sigma units, for digital PSFs in Tenor-SAXS.",
    "tenor_calibration_p_min": "Smallest p value included in the Tenor calibration grid.",
    "tenor_calibration_p_max": "Largest p value included in the Tenor calibration grid.",
    "tenor_calibration_p_count": "Number of calibration p values simulated by Tenor-SAXS.",
    "tenor_use_cubic_both": "Enable the cubic terms in the Tenor G(Q) and M(Q) polynomial fits, matching the richer MATLAB-style extraction.",
}


def _current_reconstructed_fit(analysis_res, analysis_method):
    if analysis_method == "Tomchuk":
        fit_pdi = np.asarray(analysis_res.get("I_fit_pdi", []), dtype=float)
        fit_pdi2 = np.asarray(analysis_res.get("I_fit_pdi2", []), dtype=float)
        rrms_pdi = float(analysis_res.get("rrms_pdi", np.inf))
        rrms_pdi2 = float(analysis_res.get("rrms_pdi2", np.inf))
        if fit_pdi.size and fit_pdi2.size:
            return fit_pdi if rrms_pdi <= rrms_pdi2 else fit_pdi2
        if fit_pdi.size:
            return fit_pdi
        if fit_pdi2.size:
            return fit_pdi2
        return np.asarray(analysis_res.get("I_fit_unified", []), dtype=float)
    return np.asarray(analysis_res.get("I_fit", []), dtype=float)


def _build_analysis_settings(params):
    detector_q_max = get_detector_q_max(
        pixels=params["pixels"],
        q_max=params["q_max"],
        pixel_size_um=params["pixel_size_um"],
        sample_detector_distance_cm=params["sample_detector_distance_cm"],
        wavelength_nm=params["wavelength_nm"],
    )
    return {
        **st.session_state,
        "q_max_for_tenor": detector_q_max,
    }


def _build_simulation_params(
    mode_key,
    dist_type,
    mean_rg,
    p_val,
    pixels,
    pixel_size_um,
    sample_detector_distance_cm,
    wavelength_nm,
    q_min,
    q_max,
    n_bins,
    smearing_x,
    smearing_y,
    flux,
    noise,
    binning_mode,
):
    return {
        "mean_rg": mean_rg,
        "p_val": p_val,
        "dist_type": dist_type,
        "mode": mode_key,
        "pixels": pixels,
        "pixel_size_um": pixel_size_um,
        "sample_detector_distance_cm": sample_detector_distance_cm,
        "wavelength_nm": wavelength_nm,
        "q_min": q_min,
        "q_max": q_max,
        "n_bins": n_bins,
        "smearing": 0.5 * (smearing_x + smearing_y),
        "smearing_x": smearing_x,
        "smearing_y": smearing_y,
        "flux": flux,
        "noise": noise,
        "binning_mode": binning_mode,
        "radius_samples": int(st.session_state.radius_samples),
        "q_samples": int(st.session_state.q_samples),
        "form_factor_model": st.session_state.form_factor_model,
        "phi2": float(st.session_state.phi2),
        "phi3": float(st.session_state.phi3),
        "ensemble_sampling": st.session_state.ensemble_sampling,
        "ensemble_members": int(st.session_state.ensemble_members),
    }


def _render_extracted_table(analysis_res, analysis_method, theoretical_values):
    if analysis_method == "Tomchuk":
        param_list = [
            "Rg (Guinier)",
            "G (Guinier)",
            "Rg (Selected)",
            "G (Selected)",
            "Q",
            "lc",
            "B",
            "PDI",
            "PDI2",
        ]
        extracted_numeric = [
            analysis_res.get("Rg_guinier", analysis_res.get("Rg", 0)),
            analysis_res.get("G_guinier", analysis_res.get("G", 0)),
            analysis_res.get("Rg", 0),
            analysis_res.get("G", 0),
            analysis_res.get("Q", 0),
            analysis_res.get("lc", 0),
            analysis_res.get("B", 0),
            analysis_res.get("PDI", 0),
            analysis_res.get("PDI2", 0),
        ]
        formats = ["{:.2f} nm", "{:.2e}", "{:.2f} nm", "{:.2e}", "{:.2e}", "{:.2f} nm", "{:.2e}", "{:.4f}", "{:.4f}"]
        theory_numeric = None
        if theoretical_values is not None:
            theory_numeric = [
                theoretical_values.get("Rg", 0),
                theoretical_values.get("G", 0),
                theoretical_values.get("Rg", 0),
                theoretical_values.get("G", 0),
                theoretical_values.get("Q", 0),
                theoretical_values.get("lc", 0),
                theoretical_values.get("B", 0),
                theoretical_values.get("PDI", 0),
                theoretical_values.get("PDI2", 0),
            ]
    elif analysis_method == "Tenor":
        param_list = [
            "Rg (Apparent Guinier)",
            "Mean Rg (Recovered)",
            "Mean Radius (Recovered)",
            "Weighted Variance",
            "Raw g1/g0",
            "Dimensionless J_G",
            "Raw m1/m0",
            "Candidate PSF Pairs",
        ]
        extracted_numeric = [
            analysis_res.get("Rg_guinier", 0),
            analysis_res.get("Rg", 0),
            analysis_res.get("mean_r_rec", 0),
            analysis_res.get("weighted_v", 0),
            analysis_res.get("tenor_raw_g1_over_g0", 0),
            analysis_res.get("tenor_dimless_jg", 0),
            analysis_res.get("tenor_raw_m1_over_m0", 0),
            analysis_res.get("tenor_candidate_count", 0),
        ]
        formats = ["{:.2f} nm", "{:.2f} nm", "{:.2f} nm", "{:.4f}", "{:.2e}", "{:.2e}", "{:.2e}", "{:.0f}"]
        theory_numeric = None
    else:
        param_list = ["Rg (Guinier)", "G (Guinier)"]
        extracted_numeric = [
            analysis_res.get("Rg_guinier", analysis_res.get("Rg", 0)),
            analysis_res.get("G_guinier", analysis_res.get("G", 0)),
        ]
        formats = ["{:.2f} nm", "{:.2e}"]
        theory_numeric = None

    extracted_vals = [fmt.format(val) for fmt, val in zip(formats, extracted_numeric)]
    if theory_numeric is None:
        theory_vals = ["n/a"] * len(extracted_vals)
        rel_err_vals = ["n/a"] * len(extracted_vals)
    else:
        theory_vals = [fmt.format(val) for fmt, val in zip(formats, theory_numeric)]
        rel_err_vals = []
        for extracted_val, theory_val in zip(extracted_numeric, theory_numeric):
            if theory_val != 0:
                rel_err_vals.append(f"{((extracted_val - theory_val) / theory_val):+.2%}")
            else:
                rel_err_vals.append("n/a")

    st.dataframe(
        pd.DataFrame(
            {
                "Parameter": param_list,
                "Extracted": extracted_vals,
                "Theory": theory_vals,
                "RelErr": rel_err_vals,
            }
        ),
        hide_index=True,
        width="stretch",
        height=280,
    )


def run():
    ensure_session_state_defaults(st.session_state)

    st.sidebar.title("Navigation")
    if st.sidebar.button("Return to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.header("Single Simulation / Analysis")
    tab_settings, tab_results = st.tabs(["Settings", "Results & Visuals"])

    use_experimental = False
    q_meas = None
    i_meas = None
    recommendation = None
    with st.sidebar:
        st.subheader("Sample")
        sim_mode = st.radio("Simulation Mode", ["Polydisperse Spheres", "Fixed-Length Polymers (IDP)"], key="sim_mode", help=PARAM_HELP["sim_mode"])
        mode_key = "Sphere" if "Sphere" in sim_mode else "IDP"
        analysis_options = ["Tomchuk", "NNLS", "Tenor-SAXS"] if mode_key == "Sphere" else ["NNLS"]
        default_analysis = st.session_state.get("analysis_method", "NNLS")
        if default_analysis == "Tenor":
            default_analysis = "Tenor-SAXS"
        if default_analysis not in analysis_options:
            default_analysis = analysis_options[0]
        analysis_label = st.selectbox("Analysis Method", analysis_options, index=analysis_options.index(default_analysis), help=PARAM_HELP["analysis_method"])
        analysis_method = "Tenor" if "Tenor" in analysis_label else analysis_label
        st.session_state["analysis_method"] = analysis_method

        if mode_key == "Sphere":
            radius_default = float(st.session_state.get("mean_rg", 2.0)) * np.sqrt(5.0 / 3.0)
            input_radius = st.number_input("Input R (nm)", min_value=0.1, max_value=100.0, step=0.5, value=radius_default, help=PARAM_HELP["input_r"])
            mean_rg = input_radius * np.sqrt(3.0 / 5.0)
            st.session_state["mean_rg"] = mean_rg
            st.caption(f"Equivalent input Rg: {mean_rg:.3f} nm")
        else:
            mean_rg = st.number_input("Input Rg (nm)", min_value=0.5, max_value=50.0, step=0.5, key="mean_rg", help=PARAM_HELP["input_rg"])
            input_radius = mean_rg

        p_val = st.number_input("Polydispersity (p)", min_value=0.01, max_value=6.0, step=0.01, key="p_val", help=PARAM_HELP["p_val"])
        dist_label = "Distribution Type" if mode_key == "Sphere" else "Conformational Distribution"
        dist_type = st.selectbox(dist_label, ["Gaussian", "Lognormal", "Schulz", "Boltzmann", "Triangular", "Uniform"], key="dist_type", help=PARAM_HELP["dist_type"])

        flux_help = (
            "Flux is interpreted as I(0)/(pixel dimension): "
            "the target expected photons in the nearest-to-center detector pixel "
            "after smearing and before Poisson noise."
        )
        c_flux1, c_flux2 = st.columns(2)
        c_flux1.number_input("Flux Coeff", 0.1, 9.9, key="flux_pre", step=0.1, help=PARAM_HELP["flux_pre"])
        c_flux2.number_input("Flux Exp", 1, 15, key="flux_exp", step=1, help=PARAM_HELP["flux_exp"])
        st.metric("Photon Flux", f"{get_forward_flux(st.session_state):.3e}")

        uploaded_file = st.file_uploader("Load 1D Profile", type=["dat", "out", "txt", "csv"], help="Load an experimental 1D SAXS curve. If enabled below, the analysis will run on the uploaded 1D data instead of the simulated 1D reduction.")
        if uploaded_file is not None:
            q_load, i_load, err = parse_saxs_file(uploaded_file)
            if err:
                st.error(err)
            else:
                st.success(f"Loaded {len(q_load)} points.")
                use_experimental = st.checkbox("Use loaded 1D data", value=True, key="use_loaded_data")
                q_meas, i_meas = q_load, i_load

        if mode_key == "Sphere" and analysis_method == "Tomchuk" and p_val < 0.25:
            st.warning("Tomchuk analysis is intended for highly polydisperse spheres; results below p = 0.25 may be unreliable.")

    with tab_settings:
        st.subheader("Instrument Parameters")
        c1, c2, c3, c4 = st.columns(4)
        pixels = int(c1.number_input("Detector Size (NxN)", min_value=64, step=64, key="pixels", help=PARAM_HELP["pixels"]))
        pixel_size_um = c2.number_input("Pixel Dimension (um x um)", min_value=1.0, step=1.0, key="pixel_size_um", help=PARAM_HELP["pixel_size_um"])
        sample_detector_distance_cm = c3.number_input("Sample-Detector Dist. (cm)", min_value=1.0, step=1.0, key="sample_detector_distance_cm", help=PARAM_HELP["sample_detector_distance_cm"])
        wavelength_nm = c4.number_input("Wavelength (nm)", min_value=0.001, step=0.01, key="wavelength_nm", help=PARAM_HELP["wavelength_nm"])
        detector_q_max = get_detector_q_max(
            pixels=pixels,
            q_max=st.session_state.get("q_max", 1.0),
            pixel_size_um=pixel_size_um,
            sample_detector_distance_cm=sample_detector_distance_cm,
            wavelength_nm=wavelength_nm,
        )
        if float(st.session_state.get("q_max", detector_q_max)) > detector_q_max:
            st.session_state["q_max"] = float(detector_q_max)
        st.caption(
            f"Instrument-derived detector q range: 0 to {detector_q_max:.3f} nm^-1 "
            f"(detector width {(pixels * pixel_size_um / 1.0e4):.3f} cm)"
        )
        c5, c6, c7, c8 = st.columns(4)
        q_min = c5.number_input("Min q", min_value=0.0, step=0.01, key="q_min", help=PARAM_HELP["q_min"])
        q_max = c6.number_input("Analysis q max", min_value=0.01, max_value=max(float(detector_q_max), 0.01), step=0.1, key="q_max", help=PARAM_HELP["q_max"])
        n_bins = int(c7.number_input("1D Bins", min_value=10, step=10, key="n_bins", help=PARAM_HELP["n_bins"]))
        binning_mode = c8.selectbox("Binning Mode", ["Logarithmic", "Linear"], key="binning_mode", help=PARAM_HELP["binning_mode"])
        c9, c10, c11 = st.columns(3)
        smearing_x = c9.number_input("Smearing X (px)", min_value=0.0, step=0.5, key="smearing_x", help=PARAM_HELP["smearing_x"])
        smearing_y = c10.number_input("Smearing Y (px)", min_value=0.0, step=0.5, key="smearing_y", help=PARAM_HELP["smearing_y"])
        optimal_flux = c11.checkbox("Deterministic Counts", key="optimal_flux", help=PARAM_HELP["optimal_flux"])
        add_noise = st.checkbox("Simulate Poisson Noise", value=st.session_state.add_noise, disabled=optimal_flux, key="add_noise", help=PARAM_HELP["add_noise"])

        st.subheader("Simulation")
        c12, c13, c14 = st.columns(3)
        c12.number_input("Radius Samples", min_value=50, step=50, key="radius_samples", help=PARAM_HELP["radius_samples"])
        c13.number_input("q Samples", min_value=50, step=50, key="q_samples", help=PARAM_HELP["q_samples"])
        c14.number_input("Ensemble Members", min_value=3, step=1, key="ensemble_members", help=PARAM_HELP["ensemble_members"])
        c15, c16, c17, c18 = st.columns(4)
        model_options = ["Exact Sphere", "Guinier Curvature", "Exact Gaussian Chain", "Exact Shell", "Exact Thin Rod", "Exact Thin Disk"] if mode_key == "Sphere" else ["Exact Gaussian Chain", "Guinier Curvature"]
        current_model = st.session_state.get("form_factor_model", "Exact Sphere")
        if current_model not in model_options:
            current_model = model_options[0]
            st.session_state["form_factor_model"] = current_model
        c15.selectbox("Forward Model", model_options, index=model_options.index(current_model), key="form_factor_model", help=PARAM_HELP["form_factor_model"])
        c16.selectbox("Ensemble Sampling", ["Continuous", "Discrete"], key="ensemble_sampling", help=PARAM_HELP["ensemble_sampling"])
        c17.number_input("phi2", step=0.001, key="phi2", help=PARAM_HELP["phi2"])
        c18.number_input("phi3", step=0.001, key="phi3", help=PARAM_HELP["phi3"])
        st.caption("Sample choice on the left sets the number-density distribution being simulated. For spheres, the exact-sphere forward model already applies the physical R^6 scattering weighting internally.")
        if analysis_method in {"Tomchuk", "Tenor"} and st.session_state.form_factor_model != "Exact Sphere":
            st.warning("Tomchuk and Tenor are calibrated primarily for sphere forward models; non-sphere forward models are best treated as simulation-only for now.")

        st.subheader("NNLS")
        c19, c20 = st.columns(2)
        c19.number_input("NNLS Basis Count", min_value=20, step=10, key="nnls_basis_count", help=PARAM_HELP["nnls_basis_count"])
        c20.number_input("NNLS Smooth Sigma", min_value=0.0, step=0.1, key="nnls_smooth_sigma", help=PARAM_HELP["nnls_smooth_sigma"])
        nnls_max_rg = st.number_input("NNLS Max Rg (Basis Set)", min_value=1.0, max_value=500.0, step=1.0, key="nnls_max_rg", help=PARAM_HELP["nnls_max_rg"])

        st.subheader("Tomchuk")
        target_abs_error = st.number_input("Target abs. p error", min_value=0.001, max_value=1.0, step=0.005, key="tomchuk_target_abs_error", help=PARAM_HELP["tomchuk_target_abs_error"])
        if mode_key == "Sphere" and analysis_method == "Tomchuk":
            if st.button("Evaluate q-range / bins", key="eval_tomchuk_grid"):
                with st.spinner("Sweeping q-range and 1D bins..."):
                    recommendation = recommend_tomchuk_settings(
                        mean_rg=mean_rg,
                        p_val=p_val,
                        dist_type=dist_type,
                        pixels=pixels,
                        smearing_x=smearing_x,
                        smearing_y=smearing_y,
                        flux=get_forward_flux(st.session_state),
                        noise=(not optimal_flux and add_noise),
                        q_min=q_min,
                        binning_mode=binning_mode,
                        target_abs_error=target_abs_error,
                        radius_samples=int(st.session_state.radius_samples),
                        q_samples=int(st.session_state.q_samples),
                    )
                    st.session_state["tomchuk_recommendation"] = recommendation
        recommendation = st.session_state.get("tomchuk_recommendation")

        st.subheader("Tenor-SAXS")
        c21, c22, c23, c24 = st.columns(4)
        c21.number_input("TENOR Guinier Bins", min_value=32, step=16, key="tenor_guinier_bins", help=PARAM_HELP["tenor_guinier_bins"])
        c22.number_input("TENOR Radial Bins", min_value=6, step=1, key="tenor_radial_bins", help=PARAM_HELP["tenor_radial_bins"])
        c23.number_input("TENOR qRg Limit", min_value=0.2, max_value=2.0, step=0.05, key="tenor_qrg_limit", help=PARAM_HELP["tenor_qrg_limit"])
        c24.number_input("TENOR PSF Pair Count", min_value=1, step=1, key="tenor_psf_count", help=PARAM_HELP["tenor_psf_count"])
        c25, c26, c27, c28 = st.columns(4)
        c25.number_input("TENOR PSF SigmaX Start", min_value=0.1, step=0.1, key="tenor_psf_sigma_x_start", help=PARAM_HELP["tenor_psf_sigma_x_start"])
        c26.number_input("TENOR PSF SigmaY Start", min_value=0.1, step=0.1, key="tenor_psf_sigma_y_start", help=PARAM_HELP["tenor_psf_sigma_y_start"])
        c27.number_input("TENOR PSF Sigma Step", min_value=0.05, step=0.05, key="tenor_psf_sigma_step", help=PARAM_HELP["tenor_psf_sigma_step"])
        c28.number_input("TENOR Secondary Ratio", min_value=0.1, max_value=1.0, step=0.05, key="tenor_psf_secondary_ratio", help=PARAM_HELP["tenor_psf_secondary_ratio"])
        c29, c30, c31, c32 = st.columns(4)
        c29.number_input("TENOR PSF Truncate", min_value=1.0, step=0.5, key="tenor_psf_truncate", help=PARAM_HELP["tenor_psf_truncate"])
        c30.number_input("TENOR p Min", min_value=0.001, step=0.01, key="tenor_calibration_p_min", help=PARAM_HELP["tenor_calibration_p_min"])
        c31.number_input("TENOR p Max", min_value=0.01, step=0.01, key="tenor_calibration_p_max", help=PARAM_HELP["tenor_calibration_p_max"])
        c32.number_input("TENOR Calibration Points", min_value=4, step=1, key="tenor_calibration_p_count", help=PARAM_HELP["tenor_calibration_p_count"])
        st.checkbox("TENOR Use Cubic G/M", value=st.session_state.tenor_use_g3 and st.session_state.tenor_use_m3, key="tenor_use_cubic_both", help=PARAM_HELP["tenor_use_cubic_both"])
        st.session_state["tenor_use_g3"] = bool(st.session_state.get("tenor_use_cubic_both", True))
        st.session_state["tenor_use_m3"] = bool(st.session_state.get("tenor_use_cubic_both", True))

    if analysis_method == "Tenor" and use_experimental:
        use_experimental = False
        st.warning("TENOR-SAXS currently requires a 2D detector image, so uploaded 1D profiles are not used in this mode.")

    params = _build_simulation_params(
        mode_key=mode_key,
        dist_type=dist_type,
        mean_rg=mean_rg,
        p_val=p_val,
        pixels=pixels,
        pixel_size_um=pixel_size_um,
        sample_detector_distance_cm=sample_detector_distance_cm,
        wavelength_nm=wavelength_nm,
        q_min=q_min,
        q_max=q_max,
        n_bins=n_bins,
        smearing_x=smearing_x,
        smearing_y=smearing_y,
        flux=get_forward_flux(st.session_state),
        noise=(not optimal_flux and add_noise),
        binning_mode=binning_mode,
    )
    q_sim, i_sim, i_2d_final, r_vals, pdf_vals = run_simulation_core(params)

    if use_experimental and q_meas is not None:
        mask = (q_meas >= q_min) & (q_meas <= q_max)
        q_target = q_meas[mask]
        i_target = i_meas[mask]
        normalization_scale = 1.0
    else:
        q_target = q_sim
        i_target = i_sim
        normalization_scale = 1.0
        if mode_key == "Sphere" and analysis_method == "Tomchuk" and st.session_state.form_factor_model == "Exact Sphere":
            i_target, normalization_scale = normalize_simulated_sphere_intensity(q_target, i_target, r_vals, pdf_vals)

    analysis_res = perform_saxs_analysis(
        q_target,
        i_target,
        dist_type,
        mean_rg,
        mode_key,
        analysis_method,
        nnls_max_rg,
        i_2d=i_2d_final,
        analysis_settings=_build_analysis_settings(params),
    )
    analysis_res["simulation_normalization_scale"] = normalization_scale

    reconstruction_summary = build_reconstruction_quality_summary(analysis_res) if analysis_method == "Tomchuk" else None
    theoretical_values = None
    sanity_summary = None
    if (not use_experimental) and mode_key == "Sphere" and analysis_method == "Tomchuk" and st.session_state.form_factor_model == "Exact Sphere":
        sanity_summary = build_sanity_summary_row(q_target, i_target, r_vals, pdf_vals, analysis_res)
        theoretical_values = calculate_sphere_input_theoretical_parameters(mean_rg, p_val, dist_type)

    with tab_results:
        viz1, viz2 = st.columns(2)
        with viz1:
            if use_experimental:
                st.info("Analyzing uploaded 1D data. 2D detector view is disabled.")
            else:
                q_axis = build_detector_q_grid(
                    pixels=pixels,
                    q_max=q_max,
                    pixel_size_um=pixel_size_um,
                    sample_detector_distance_cm=sample_detector_distance_cm,
                    wavelength_nm=wavelength_nm,
                )[0]
                fig_2d = go.Figure(
                    data=go.Heatmap(
                        z=np.log10(np.maximum(i_2d_final, 1.0)),
                        x=q_axis,
                        y=q_axis,
                        colorscale="Jet",
                        colorbar=dict(title="log10(I)"),
                    )
                )
                fig_2d.update_layout(
                    title="2D Detector",
                    xaxis_title="qx",
                    yaxis_title="qy",
                    width=500,
                    height=400,
                    margin=dict(l=30, r=30, t=40, b=30),
                    yaxis=dict(scaleanchor="x", scaleratio=1),
                )
                st.plotly_chart(fig_2d, width="stretch")

        with viz2:
            plot_opts = ["Log-Log", "Lin-Lin", "Guinier", "Kratky"]
            if mode_key == "Sphere":
                plot_opts.append("Porod")
            plot_type = st.selectbox("Plot Type", plot_opts)
            fig_1d = go.Figure()
            plot_x, plot_y = q_target, i_target
            x_type, y_type = "linear", "linear"
            x_label, y_label = "q (nm⁻¹)", "I(q)"
            current_fit = _current_reconstructed_fit(analysis_res, analysis_method)
            if plot_type == "Log-Log":
                x_type, y_type = "log", "log"
            elif plot_type == "Guinier":
                plot_x = q_target ** 2
                plot_y = np.log(np.maximum(i_target, 1e-9))
                x_label, y_label = "q²", "ln(I)"
            elif plot_type == "Porod":
                plot_y = i_target * (q_target ** 4)
                y_label = "I · q⁴"
            elif plot_type == "Kratky":
                plot_y = i_target * (q_target ** 2)
                y_label = "I · q²"

            fig_1d.add_trace(go.Scatter(x=plot_x, y=plot_y, mode="markers", name="Input", marker=dict(color="royalblue", size=4)))
            if current_fit.ndim == 1 and len(current_fit) == len(q_target):
                fit_x, fit_y = q_target, current_fit
                if plot_type == "Guinier":
                    rg_guinier = max(float(analysis_res.get("Rg_guinier", 0)), 1e-9)
                    fit_mask = q_target * rg_guinier < 1.0
                    fit_x = (q_target[fit_mask]) ** 2
                    fit_y = np.log(np.maximum(fit_y[fit_mask], 1e-9))
                elif plot_type == "Porod":
                    fit_y = fit_y * (q_target ** 4)
                elif plot_type == "Kratky":
                    fit_y = fit_y * (q_target ** 2)
                fig_1d.add_trace(go.Scatter(x=fit_x, y=fit_y, mode="lines", name="Recovered Fit", line=dict(color="orange", dash="dash", width=2)))

            if plot_type == "Guinier" and analysis_res.get("Rg_guinier", 0) > 0 and analysis_res.get("G_guinier", 0) > 0:
                rg_guinier = float(analysis_res["Rg_guinier"])
                guinier_mask = q_target * rg_guinier < 1.0
                q_sq_line = (q_target[guinier_mask]) ** 2
                guinier_line = np.log(np.maximum(analysis_res["G_guinier"], 1e-12)) - (analysis_res["Rg_guinier"] ** 2 / 3.0) * q_sq_line
                fig_1d.add_trace(
                    go.Scatter(
                        x=q_sq_line,
                        y=guinier_line,
                        mode="lines",
                        name="Guinier Fit",
                        line=dict(color="green", width=2),
                    )
                )
                q_sq_max = (min(np.max(q_target[guinier_mask]), 1.0 / rg_guinier)) ** 2 if np.any(guinier_mask) else np.max(q_target ** 2)
                fig_1d.update_xaxes(range=[0.0, q_sq_max * 1.05])

            if plot_type == "Porod" and analysis_res.get("B", 0) > 0:
                fig_1d.add_trace(
                    go.Scatter(
                        x=q_target,
                        y=np.full_like(q_target, float(analysis_res["B"])),
                        mode="lines",
                        name="Porod B Fit",
                        line=dict(color="firebrick", dash="dot", width=2),
                    )
                )

            fig_1d.update_layout(
                title="1D Profile Analysis",
                xaxis_title=x_label,
                yaxis_title=y_label,
                xaxis_type=x_type,
                yaxis_type=y_type,
                width=500,
                height=400,
                margin=dict(l=30, r=30, t=40, b=30),
            )
            st.plotly_chart(fig_1d, width="stretch")

        st.divider()
        c3, c4, c5, c6 = st.columns(4)
        with c3:
            st.markdown("**Recovered Parameters**")
            if analysis_method == "Tomchuk":
                st.metric("Rec. p (PDI)", f"{analysis_res.get('p_rec_pdi', 0):.3f}")
                st.metric("Rec. p (PDI2)", f"{analysis_res.get('p_rec_pdi2', 0):.3f}")
                st.metric("Rec. Mean Radius (PDI)", f"{analysis_res.get('mean_r_rec_pdi', 0):.2f} nm")
                st.metric("Rec. Mean Radius (PDI2)", f"{analysis_res.get('mean_r_rec_pdi2', 0):.2f} nm")
            elif analysis_method == "Tenor":
                st.metric("Rec. p", f"{analysis_res.get('p_rec', 0):.3f}")
                st.metric("Rec. Mean Rg", f"{analysis_res.get('Rg', 0):.2f} nm")
                st.metric("Rec. Mean Radius", f"{analysis_res.get('mean_r_rec', 0):.2f} nm")
                st.metric("Fit RelRMS", f"{analysis_res.get('rrms', 0):.4f}")
            else:
                rec_label = "NNLS Mean Radius" if mode_key == "Sphere" else "NNLS Mean Rg"
                rec_value = analysis_res.get("mean_r_rec", 0) if mode_key == "Sphere" else analysis_res.get("rg_num_rec", 0)
                st.metric(rec_label, f"{rec_value:.2f} nm")
                st.metric("NNLS Width (p)", f"{analysis_res.get('p_rec', 0):.3f}")
                st.metric("Fit RelRMS", f"{analysis_res.get('rrms', 0):.4f}")

        with c4:
            st.markdown("**Extracted Values**")
            if analysis_method == "Tomchuk":
                st.metric("Tomchuk Path", analysis_res.get("tomchuk_extraction", "n/a"))
            elif analysis_method == "Tenor":
                st.metric("PSF Candidates", f"{analysis_res.get('tenor_candidate_count', 0)}")
            _render_extracted_table(analysis_res, analysis_method, theoretical_values)
            if theoretical_values is not None and normalization_scale != 1.0:
                st.caption(f"Simulated 1D data was normalized by {normalization_scale:.4e} before Tomchuk analysis.")

        with c5:
            st.markdown("**Diagnostics**")
            if analysis_method == "Tomchuk" and reconstruction_summary is not None:
                st.caption(f"Best reconstructed fit: {reconstruction_summary['best_variant']}")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Variant": "PDI", "RelRMS": f"{reconstruction_summary['rrms_pdi']:.4f}", "Quality": reconstruction_summary['quality_pdi']},
                            {"Variant": "PDI2", "RelRMS": f"{reconstruction_summary['rrms_pdi2']:.4f}", "Quality": reconstruction_summary['quality_pdi2']},
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            elif analysis_method == "Tenor":
                st.metric("Weighted Variance", f"{analysis_res.get('weighted_v', 0):.4f}")
                st.metric("Raw g1/g0", f"{analysis_res.get('tenor_raw_g1_over_g0', 0):.2e}")
                st.metric("Raw g1/g0^2", f"{analysis_res.get('tenor_raw_g100_ratio', 0):.2e}")
                st.metric("Dimensionless J_G", f"{analysis_res.get('tenor_dimless_jg', 0):.2e}")
            else:
                st.caption("NNLS diagnostics are represented mainly by the fit error and recovered distribution.")

            if sanity_summary is not None:
                st.metric("Sanity", "PASS" if sanity_summary.get("Sanity_Pass") else "CHECK")
                st.caption(f"Failures: {sanity_summary.get('Sanity_Failures', 'none')}")
                st.caption(sanity_summary.get("Sanity_Suggestions", ""))
            elif use_experimental:
                st.caption("Sanity checks require simulated ground truth.")

            if recommendation:
                best = recommendation.get("best")
                safety = recommendation.get("safety_zone")
                if best:
                    st.caption(
                        f"Best q_max={best['q_max']:.2f}, bins={best['n_bins']}, "
                        f"max abs p error={best['max_abs_err']:.3f}"
                    )
                if safety:
                    st.caption(
                        f"Safety zone q_max={safety['q_max_min']:.2f}-{safety['q_max_max']:.2f}, "
                        f"bins={safety['n_bins_min']}-{safety['n_bins_max']}"
                    )

        with c6:
            st.markdown("**Recovered Distribution**")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(x=r_vals, y=pdf_vals, mode="lines", name="Input", line=dict(color="gray", dash="dash")))
            rec_dists_dl = {}
            if analysis_method == "Tomchuk":
                mean_r_pdi = analysis_res.get("mean_r_rec_pdi", 0)
                mean_r_pdi2 = analysis_res.get("mean_r_rec_pdi2", 0)
                if mean_r_pdi > 0:
                    pdf_rec_pdi = get_distribution(dist_type, r_vals, mean_r_pdi, analysis_res.get("p_rec_pdi", 0))
                    rec_dists_dl["pdi"] = pdf_rec_pdi
                    fig_dist.add_trace(go.Scatter(x=r_vals, y=pdf_rec_pdi, mode="lines", name="PDI Rec.", line=dict(color="blue")))
                if mean_r_pdi2 > 0:
                    pdf_rec_pdi2 = get_distribution(dist_type, r_vals, mean_r_pdi2, analysis_res.get("p_rec_pdi2", 0))
                    rec_dists_dl["pdi2"] = pdf_rec_pdi2
                    fig_dist.add_trace(go.Scatter(x=r_vals, y=pdf_rec_pdi2, mode="lines", name="PDI2 Rec.", line=dict(color="purple")))
            elif analysis_method == "Tenor":
                mean_r_tenor = analysis_res.get("mean_r_rec", 0)
                if mean_r_tenor > 0:
                    pdf_tenor = get_distribution(dist_type, r_vals, mean_r_tenor, analysis_res.get("p_rec", 0))
                    rec_dists_dl["tenor"] = pdf_tenor
                    fig_dist.add_trace(go.Scatter(x=r_vals, y=pdf_tenor, mode="lines", name="Tenor Rec.", line=dict(color="crimson")))
            elif "nnls_r" in analysis_res:
                rec_dists_dl["nnls_r"] = analysis_res["nnls_r"]
                rec_dists_dl["nnls_pdf"] = analysis_res.get("nnls_pdf", analysis_res.get("nnls_w", []))
                fig_dist.add_trace(
                    go.Scatter(
                        x=analysis_res["nnls_r"],
                        y=rec_dists_dl["nnls_pdf"],
                        mode="none",
                        fill="tozeroy",
                        name="NNLS Rec.",
                        fillcolor="rgba(255,165,0,0.5)",
                        line=dict(color="orange"),
                    )
                )
            fig_dist.update_layout(xaxis_title="Radius (nm)", yaxis_title="Prob", width=350, height=300, margin=dict(l=25, r=25, t=20, b=25))
            st.plotly_chart(fig_dist, width="stretch")

            params_dict = {
                "mean_rg": mean_rg,
                "p_val": p_val,
                "dist_type": dist_type,
                "mode": mode_key,
                "method": analysis_method,
            }
            st.download_button(
                "Download Intensity Data (.csv)",
                create_intensity_csv(get_header_string(params_dict, analysis_res), q_target, i_target, analysis_res, analysis_method),
                "saxs_intensity.csv",
                "text/csv",
            )
            st.download_button(
                "Download Distribution Data (.csv)",
                create_distribution_csv(get_header_string(params_dict, analysis_res), r_vals, pdf_vals, rec_dists_dl, params_dict),
                "saxs_distribution.csv",
                "text/csv",
            )
