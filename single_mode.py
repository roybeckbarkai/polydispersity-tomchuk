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
from sim_utils import get_distribution, run_simulation_core


def _build_analysis_settings(q_max):
    return {
        **st.session_state,
        "q_max_for_tenor": q_max,
    }


def _build_simulation_params(mode_key, dist_type, mean_rg, p_val, pixels, q_min, q_max, n_bins, smearing_x, smearing_y, flux, noise, binning_mode):
    return {
        "mean_rg": mean_rg,
        "p_val": p_val,
        "dist_type": dist_type,
        "mode": mode_key,
        "pixels": pixels,
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
    tab_settings, tab_visuals, tab_results = st.tabs(["Settings", "Visuals", "Results"])

    use_experimental = False
    q_meas = None
    i_meas = None
    recommendation = None

    with tab_settings:
        st.subheader("Experiment and Method")
        c0, c1, c2 = st.columns(3)
        with c0:
            sim_mode = st.radio("Simulation Mode", ["Polydisperse Spheres", "Fixed-Length Polymers (IDP)"], key="sim_mode")
        mode_key = "Sphere" if "Sphere" in sim_mode else "IDP"
        analysis_options = ["Tomchuk", "NNLS", "Tenor-SAXS"] if mode_key == "Sphere" else ["NNLS"]
        default_analysis = st.session_state.get("analysis_method", "NNLS")
        if default_analysis == "Tenor":
            default_analysis = "Tenor-SAXS"
        if default_analysis not in analysis_options:
            default_analysis = analysis_options[0]
        analysis_label = c1.selectbox("Analysis Method", analysis_options, index=analysis_options.index(default_analysis))
        analysis_method = "Tenor" if "Tenor" in analysis_label else analysis_label
        st.session_state["analysis_method"] = analysis_method
        with c2:
            uploaded_file = st.file_uploader("Load 1D Profile", type=["dat", "out", "txt", "csv"])
            if uploaded_file is not None:
                q_load, i_load, err = parse_saxs_file(uploaded_file)
                if err:
                    st.error(err)
                else:
                    st.success(f"Loaded {len(q_load)} points.")
                    use_experimental = st.checkbox("Use loaded 1D data", value=True, key="use_loaded_data")
                    q_meas, i_meas = q_load, i_load

        st.subheader("Sample")
        c3, c4, c5 = st.columns(3)
        mean_rg = c3.number_input("Mean Rg (nm)", min_value=0.5, max_value=50.0, step=0.5, key="mean_rg")
        p_val = c4.number_input("Polydispersity (p)", min_value=0.01, max_value=6.0, step=0.01, key="p_val")
        dist_label = "Distribution Type" if mode_key == "Sphere" else "Conformational Distribution"
        dist_type = c5.selectbox(dist_label, ["Gaussian", "Lognormal", "Schulz", "Boltzmann", "Triangular", "Uniform"], key="dist_type")
        if mode_key == "Sphere" and analysis_method == "Tomchuk" and p_val < 0.25:
            st.warning("Tomchuk analysis is intended for highly polydisperse spheres; results below p = 0.25 may be unreliable.")

        st.subheader("Instrument and Binning")
        c6, c7, c8, c9 = st.columns(4)
        pixels = int(c6.number_input("Detector Size (NxN)", min_value=64, step=64, key="pixels"))
        q_min = c7.number_input("Min q", min_value=0.0, step=0.01, key="q_min")
        q_max = c8.number_input("Max q", min_value=0.01, step=0.1, key="q_max")
        n_bins = int(c9.number_input("1D Bins", min_value=10, step=10, key="n_bins"))
        c10, c11, c12 = st.columns(3)
        binning_mode = c10.selectbox("Binning Mode", ["Logarithmic", "Linear"], key="binning_mode")
        smearing_x = c11.number_input("Smearing X (px)", min_value=0.0, step=0.5, key="smearing_x")
        smearing_y = c12.number_input("Smearing Y (px)", min_value=0.0, step=0.5, key="smearing_y")

        st.subheader("Photon Statistics")
        flux_help = (
            "Flux is interpreted as I(0)/(pixel dimension): "
            "the target expected photons in the nearest-to-center detector pixel "
            "after smearing and before Poisson noise."
        )
        c13, c14, c15 = st.columns(3)
        flux_pre = c13.number_input("Forward Coeff", 0.1, 9.9, key="flux_pre", step=0.1, help=flux_help)
        flux_exp = c14.number_input("Forward Exp", 1, 15, key="flux_exp", step=1, help=flux_help)
        optimal_flux = c15.checkbox("Deterministic Counts", key="optimal_flux")
        add_noise = st.checkbox("Simulate Poisson Noise", value=st.session_state.add_noise, disabled=optimal_flux, key="add_noise")

        st.subheader("Simulation and Analysis Settings")
        c16, c17, c18, c19 = st.columns(4)
        c16.number_input("Radius Samples", min_value=50, step=50, key="radius_samples")
        c17.number_input("q Samples", min_value=50, step=50, key="q_samples")
        c18.number_input("NNLS Basis Count", min_value=20, step=10, key="nnls_basis_count")
        c19.number_input("NNLS Smooth Sigma", min_value=0.0, step=0.1, key="nnls_smooth_sigma")
        nnls_max_rg = st.number_input("NNLS Max Rg (Basis Set)", min_value=1.0, max_value=500.0, step=1.0, key="nnls_max_rg")

        if mode_key == "Sphere" and analysis_method == "Tenor":
            st.subheader("Tenor-SAXS Settings")
            c20, c21, c22, c23 = st.columns(4)
            c20.number_input("TENOR Guinier Bins", min_value=32, step=16, key="tenor_guinier_bins")
            c21.number_input("TENOR Radial Bins", min_value=6, step=1, key="tenor_radial_bins")
            c22.number_input("TENOR qRg Limit", min_value=0.2, max_value=2.0, step=0.05, key="tenor_qrg_limit")
            c23.number_input("TENOR PSF Pair Count", min_value=1, step=1, key="tenor_psf_count")
            c24, c25, c26, c27 = st.columns(4)
            c24.number_input("TENOR PSF SigmaX Start", min_value=0.1, step=0.1, key="tenor_psf_sigma_x_start")
            c25.number_input("TENOR PSF SigmaY Start", min_value=0.1, step=0.1, key="tenor_psf_sigma_y_start")
            c26.number_input("TENOR PSF Sigma Step", min_value=0.05, step=0.05, key="tenor_psf_sigma_step")
            c27.number_input("TENOR Secondary Ratio", min_value=0.1, max_value=1.0, step=0.05, key="tenor_psf_secondary_ratio")
            c28, c29, c30, c31 = st.columns(4)
            c28.number_input("TENOR PSF Truncate", min_value=1.0, step=0.5, key="tenor_psf_truncate")
            c29.number_input("TENOR p Min", min_value=0.001, step=0.01, key="tenor_calibration_p_min")
            c30.number_input("TENOR p Max", min_value=0.01, step=0.01, key="tenor_calibration_p_max")
            c31.number_input("TENOR Calibration Points", min_value=4, step=1, key="tenor_calibration_p_count")
            st.checkbox("TENOR Use Cubic G/M", value=st.session_state.tenor_use_g3 and st.session_state.tenor_use_m3, key="tenor_use_cubic_both")
            st.session_state["tenor_use_g3"] = bool(st.session_state.get("tenor_use_cubic_both", True))
            st.session_state["tenor_use_m3"] = bool(st.session_state.get("tenor_use_cubic_both", True))

        if mode_key == "Sphere" and analysis_method == "Tomchuk":
            st.subheader("Tomchuk Evaluator")
            target_abs_error = st.number_input("Target abs. p error", min_value=0.001, max_value=1.0, step=0.005, key="tomchuk_target_abs_error")
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

    if analysis_method == "Tenor" and use_experimental:
        use_experimental = False
        st.warning("TENOR-SAXS currently requires a 2D detector image, so uploaded 1D profiles are not used in this mode.")

    params = _build_simulation_params(
        mode_key=mode_key,
        dist_type=dist_type,
        mean_rg=mean_rg,
        p_val=p_val,
        pixels=pixels,
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
        if mode_key == "Sphere" and analysis_method == "Tomchuk":
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
        analysis_settings=_build_analysis_settings(q_max),
    )
    analysis_res["simulation_normalization_scale"] = normalization_scale

    reconstruction_summary = build_reconstruction_quality_summary(analysis_res) if analysis_method == "Tomchuk" else None
    theoretical_values = None
    sanity_summary = None
    if (not use_experimental) and mode_key == "Sphere" and analysis_method == "Tomchuk":
        sanity_summary = build_sanity_summary_row(q_target, i_target, r_vals, pdf_vals, analysis_res)
        theoretical_values = calculate_sphere_input_theoretical_parameters(mean_rg, p_val, dist_type)

    with tab_visuals:
        c1, c2 = st.columns(2)
        with c1:
            if use_experimental:
                st.info("Analyzing uploaded 1D data. 2D detector view is disabled.")
            else:
                fig_2d = go.Figure(
                    data=go.Heatmap(
                        z=np.log10(np.maximum(i_2d_final, 1.0)),
                        x=np.linspace(-q_max, q_max, pixels),
                        y=np.linspace(-q_max, q_max, pixels),
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
                st.plotly_chart(fig_2d, use_container_width=True)

        with c2:
            plot_opts = ["Log-Log", "Lin-Lin", "Guinier", "Kratky"]
            if mode_key == "Sphere":
                plot_opts.append("Porod")
            plot_type = st.selectbox("Plot Type", plot_opts)
            fig_1d = go.Figure()
            plot_x, plot_y = q_target, i_target
            x_type, y_type = "linear", "linear"
            x_label, y_label = "q (nm⁻¹)", "I(q)"
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
            fit_arr = np.asarray(analysis_res.get("I_fit", []), dtype=float)
            if fit_arr.ndim == 1 and len(fit_arr) == len(q_target):
                fit_x, fit_y = q_target, fit_arr
                if plot_type == "Guinier":
                    fit_x = q_target ** 2
                    fit_y = np.log(np.maximum(fit_y, 1e-9))
                elif plot_type == "Porod":
                    fit_y = fit_y * (q_target ** 4)
                elif plot_type == "Kratky":
                    fit_y = fit_y * (q_target ** 2)
                fig_1d.add_trace(go.Scatter(x=fit_x, y=fit_y, mode="lines", name="Recovered Fit", line=dict(color="orange", dash="dash", width=2)))

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
            st.plotly_chart(fig_1d, use_container_width=True)

    with tab_results:
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
            st.plotly_chart(fig_dist, use_container_width=True)

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
