# File: batch_mode.py
# Last Updated: Monday, March 30, 2026
# Description: Logic for running batch parameter sweeps with short-code inputs and error analysis plotting.

import ast
import io
import itertools
import zipfile
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis_utils import (
    build_sanity_summary_row,
    build_summary_row,
    create_distribution_csv,
    create_intensity_csv,
    get_header_string,
    run_simulation_analysis_case,
)
from app_settings import DEFAULT_APP_SETTINGS


MODE_MAP = {"S": "Sphere", "P": "IDP"}
DIST_MAP = {"G": "Gaussian", "L": "Lognormal", "S": "Schulz", "B": "Boltzmann", "T": "Triangular", "U": "Uniform"}
METHOD_MAP = {"T": "Tomchuk", "N": "NNLS", "E": "Tenor"}
BIN_MAP = {"Log": "Logarithmic", "Lin": "Linear"}

REV_MODE_MAP = {v: k for k, v in MODE_MAP.items()}
REV_DIST_MAP = {v: k for k, v in DIST_MAP.items()}
REV_METHOD_MAP = {v: k for k, v in METHOD_MAP.items()}


def expand_batch_parameters(df):
    all_jobs = []
    df = df.astype(str)
    for _, row in df.iterrows():
        base_params = row.to_dict()
        list_params = {}
        single_params = {}
        for key, value in base_params.items():
            val_str = str(value).strip()
            if val_str.startswith("[") and val_str.endswith("]"):
                try:
                    parsed = ast.literal_eval(val_str)
                    if isinstance(parsed, list):
                        list_params[key] = parsed
                        continue
                except Exception:
                    pass
            single_params[key] = value
        if not list_params:
            all_jobs.append(single_params)
            continue
        keys = list(list_params.keys())
        vals = list(list_params.values())
        for combination in itertools.product(*vals):
            job = single_params.copy()
            for key, value in zip(keys, combination):
                job[key] = value
            all_jobs.append(job)
    return pd.DataFrame(all_jobs)


def _default_row():
    return {
        "mode (S/P)": REV_MODE_MAP.get("Sphere", "S"),
        "dist (G/L/S/B/T/U)": REV_DIST_MAP.get(DEFAULT_APP_SETTINGS["dist_type"], "G"),
        "method (T/N/E)": REV_METHOD_MAP.get(DEFAULT_APP_SETTINGS["analysis_method"], "N"),
        "mean_rg": str(DEFAULT_APP_SETTINGS["mean_rg"]),
        "p_val": str(DEFAULT_APP_SETTINGS["p_val"]),
        "pixels": str(DEFAULT_APP_SETTINGS["pixels"]),
        "q_min": str(DEFAULT_APP_SETTINGS["q_min"]),
        "q_max": str(DEFAULT_APP_SETTINGS["q_max"]),
        "n_bins": str(DEFAULT_APP_SETTINGS["n_bins"]),
        "binning (Log/Lin)": "Log",
        "smearing_x": str(DEFAULT_APP_SETTINGS["smearing_x"]),
        "smearing_y": str(DEFAULT_APP_SETTINGS["smearing_y"]),
        "flux": str(DEFAULT_APP_SETTINGS["flux_pre"] * (10 ** DEFAULT_APP_SETTINGS["flux_exp"])),
        "noise": "True",
        "radius_samples": str(DEFAULT_APP_SETTINGS["radius_samples"]),
        "q_samples": str(DEFAULT_APP_SETTINGS["q_samples"]),
        "nnls_max_rg": str(DEFAULT_APP_SETTINGS["nnls_max_rg"]),
        "nnls_basis_count": str(DEFAULT_APP_SETTINGS["nnls_basis_count"]),
        "nnls_smooth_sigma": str(DEFAULT_APP_SETTINGS["nnls_smooth_sigma"]),
        "tenor_guinier_bins": str(DEFAULT_APP_SETTINGS["tenor_guinier_bins"]),
        "tenor_radial_bins": str(DEFAULT_APP_SETTINGS["tenor_radial_bins"]),
        "tenor_qrg_limit": str(DEFAULT_APP_SETTINGS["tenor_qrg_limit"]),
        "tenor_psf_count": str(DEFAULT_APP_SETTINGS["tenor_psf_count"]),
        "tenor_psf_truncate": str(DEFAULT_APP_SETTINGS["tenor_psf_truncate"]),
        "tenor_psf_sigma_x_start": str(DEFAULT_APP_SETTINGS["tenor_psf_sigma_x_start"]),
        "tenor_psf_sigma_y_start": str(DEFAULT_APP_SETTINGS["tenor_psf_sigma_y_start"]),
        "tenor_psf_sigma_step": str(DEFAULT_APP_SETTINGS["tenor_psf_sigma_step"]),
        "tenor_psf_secondary_ratio": str(DEFAULT_APP_SETTINGS["tenor_psf_secondary_ratio"]),
        "tenor_use_g3": str(DEFAULT_APP_SETTINGS["tenor_use_g3"]),
        "tenor_use_m3": str(DEFAULT_APP_SETTINGS["tenor_use_m3"]),
        "tenor_calibration_p_min": str(DEFAULT_APP_SETTINGS["tenor_calibration_p_min"]),
        "tenor_calibration_p_max": str(DEFAULT_APP_SETTINGS["tenor_calibration_p_max"]),
        "tenor_calibration_p_count": str(DEFAULT_APP_SETTINGS["tenor_calibration_p_count"]),
    }


def run():
    st.header("Batch Simulation Runner")
    if st.button("Return to Home"):
        st.session_state.page = "home"
        st.rerun()

    tab_settings, tab_results = st.tabs(["Settings", "Results"])
    if "batch_df" not in st.session_state:
        st.session_state.batch_df = pd.DataFrame([_default_row()])

    with tab_settings:
        c1, c2 = st.columns(2)
        with c1:
            uploaded_batch = st.file_uploader("Upload Batch CSV", type=["csv"])
            if uploaded_batch:
                try:
                    st.session_state.batch_df = pd.read_csv(uploaded_batch, dtype=str)
                    st.success("Batch loaded.")
                except Exception:
                    st.error("Invalid CSV file.")
        with c2:
            st.download_button(
                "Download Template",
                pd.DataFrame([_default_row()]).to_csv(index=False),
                "batch_template.csv",
                "text/csv",
            )

        st.info(
            "Use codes: S=Sphere, P=Polymer | G/L/S/B/T/U distributions | "
            "T=Tomchuk, N=NNLS, E=Tenor-SAXS. Enter lists like `[0.1, 0.3]` to sweep values."
        )
        st.session_state.batch_df = st.data_editor(st.session_state.batch_df.astype(str), num_rows="dynamic")

    with tab_results:
        if st.button("Execute Batch Queue"):
            expanded_df = expand_batch_parameters(st.session_state.batch_df)
            st.write(f"Queue size: {len(expanded_df)} simulations")
            progress_bar = st.progress(0)
            zip_buffer = io.BytesIO()
            summary_results = []

            with zipfile.ZipFile(zip_buffer, "w") as zf:
                zf.writestr("batch_summary_expanded.csv", expanded_df.to_csv(index=False))
                for idx, row in expanded_df.iterrows():
                    row_dict = row.to_dict()
                    try:
                        mode = MODE_MAP.get(row_dict.get("mode (S/P)", "S"), "Sphere")
                        method = METHOD_MAP.get(row_dict.get("method (T/N/E)", "N"), "NNLS")
                        if mode == "IDP":
                            method = "NNLS"
                        params = {
                            "mode": mode,
                            "dist_type": DIST_MAP.get(row_dict.get("dist (G/L/S/B/T/U)", "G"), "Gaussian"),
                            "method": method,
                            "mean_rg": float(row_dict["mean_rg"]),
                            "p_val": float(row_dict["p_val"]),
                            "pixels": int(float(row_dict["pixels"])),
                            "q_min": float(row_dict["q_min"]),
                            "q_max": float(row_dict["q_max"]),
                            "n_bins": int(float(row_dict["n_bins"])),
                            "binning_mode": BIN_MAP.get(row_dict.get("binning (Log/Lin)", "Log"), "Logarithmic"),
                            "smearing": 0.5 * (float(row_dict["smearing_x"]) + float(row_dict["smearing_y"])),
                            "smearing_x": float(row_dict["smearing_x"]),
                            "smearing_y": float(row_dict["smearing_y"]),
                            "flux": float(row_dict["flux"]),
                            "noise": str(row_dict["noise"]).lower() in ["true", "1", "t", "yes"],
                            "radius_samples": int(float(row_dict["radius_samples"])),
                            "q_samples": int(float(row_dict["q_samples"])),
                            "nnls_max_rg": float(row_dict["nnls_max_rg"]),
                            "nnls_basis_count": int(float(row_dict["nnls_basis_count"])),
                            "nnls_smooth_sigma": float(row_dict["nnls_smooth_sigma"]),
                            "tenor_guinier_bins": int(float(row_dict["tenor_guinier_bins"])),
                            "tenor_radial_bins": int(float(row_dict["tenor_radial_bins"])),
                            "tenor_qrg_limit": float(row_dict["tenor_qrg_limit"]),
                            "tenor_psf_count": int(float(row_dict["tenor_psf_count"])),
                            "tenor_psf_truncate": float(row_dict["tenor_psf_truncate"]),
                            "tenor_psf_sigma_x_start": float(row_dict["tenor_psf_sigma_x_start"]),
                            "tenor_psf_sigma_y_start": float(row_dict["tenor_psf_sigma_y_start"]),
                            "tenor_psf_sigma_step": float(row_dict["tenor_psf_sigma_step"]),
                            "tenor_psf_secondary_ratio": float(row_dict["tenor_psf_secondary_ratio"]),
                            "tenor_use_g3": str(row_dict["tenor_use_g3"]).lower() in ["true", "1", "t", "yes"],
                            "tenor_use_m3": str(row_dict["tenor_use_m3"]).lower() in ["true", "1", "t", "yes"],
                            "tenor_calibration_p_min": float(row_dict["tenor_calibration_p_min"]),
                            "tenor_calibration_p_max": float(row_dict["tenor_calibration_p_max"]),
                            "tenor_calibration_p_count": int(float(row_dict["tenor_calibration_p_count"])),
                        }

                        q_sim, i_sim, r_vals, pdf_vals, res, rec_dists = run_simulation_analysis_case(params)
                        header = get_header_string(params, res)
                        zf.writestr(f"run_{idx}_intensity.csv", create_intensity_csv(header, q_sim, i_sim, res, params["method"]))
                        zf.writestr(f"run_{idx}_distribution.csv", create_distribution_csv(header, r_vals, pdf_vals, rec_dists, params))

                        summary_row = build_summary_row(params, res, base_row=row_dict)
                        if params["mode"] == "Sphere":
                            summary_row = build_sanity_summary_row(q_sim, i_sim, r_vals, pdf_vals, res, base_row=summary_row)
                        summary_results.append(summary_row)
                    except Exception as exc:
                        zf.writestr(f"run_{idx}_error.txt", str(exc))
                    progress_bar.progress((idx + 1) / len(expanded_df))

                if summary_results:
                    zf.writestr("batch_results_summary.csv", pd.DataFrame(summary_results).to_csv(index=False))

            if summary_results:
                res_df = pd.DataFrame(summary_results)
                for col in res_df.columns:
                    res_df[col] = pd.to_numeric(res_df[col], errors="ignore")
                st.success("Batch complete.")
                varying_cols = [col for col in res_df.columns if res_df[col].nunique() > 1 and col not in {"Recovered_p", "Recovered_Size", "Rel_Err_p", "Rel_Err_Size", "RelRMS"}]
                x_axis = st.selectbox("Select X-Axis Variable", varying_cols if varying_cols else list(res_df.columns), index=0)
                method_color_col = "method (T/N/E)" if "method (T/N/E)" in res_df.columns else None
                r1, r2 = st.tabs(["Size Recovery Error", "Polydispersity Error"])
                with r1:
                    fig_size = px.scatter(
                        res_df,
                        x=x_axis,
                        y="Rel_Err_Size",
                        color=method_color_col,
                        hover_data=["mean_rg", "p_val", "smearing_x", "smearing_y"],
                        title=f"Relative Error in Recovered Size vs {x_axis}",
                    )
                    fig_size.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_size, use_container_width=True)
                with r2:
                    fig_p = px.scatter(
                        res_df,
                        x=x_axis,
                        y="Rel_Err_p",
                        color=method_color_col,
                        hover_data=["mean_rg", "p_val", "smearing_x", "smearing_y"],
                        title=f"Relative Error in p vs {x_axis}",
                    )
                    fig_p.add_hline(y=0, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig_p, use_container_width=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("Download Batch Results (ZIP)", zip_buffer.getvalue(), f"saxs_batch_{timestamp}.zip", "application/zip")
