from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from matlab_protocol.matlab_results import load_matlab_results_table
from matlab_protocol.params import init_tenor_params
from matlab_protocol.plotting import plot_tenor_violin_closest
from matlab_protocol.tenor_analysis import load_gt_library, tenor_process_landscape


def _normalize_v_all(value, primary_v, alternatives):
    if np.isfinite(primary_v):
        vals = [float(primary_v)]
    else:
        vals = [np.nan]
    if alternatives is not None:
        vals.extend([float(v) for v in np.asarray(alternatives, dtype=float).ravel()])
    return vals


def main():
    repo_root = Path(__file__).resolve().parent
    matlab_h5_dir = Path(
        "/Users/roybeck/Library/CloudStorage/Dropbox/python code copy/tomchuk study files/matlab-files"
    )
    output_dir = repo_root / "matlab_protocol_outputs"
    output_dir.mkdir(exist_ok=True)

    matlab_table = load_matlab_results_table(repo_root / "matlab-new" / "sphere_r5_multinoise_lognormal.mat")
    gt_lib = load_gt_library(repo_root / "matlab-new" / "GroundTruth_sphere_rg5_lognormal.mat")
    instrument, simulation, ensemble, _ = init_tenor_params()

    rows = []
    for idx, matlab_row in matlab_table.iterrows():
        true_v = float(matlab_row["True_V"])
        noise = float(matlab_row["Noise"])
        filename = f"sim_{idx + 1:04d}_V{true_v:0.3f}.h5"
        file_path = matlab_h5_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Expected MATLAB file not found: {file_path}")

        with h5py.File(file_path, "r") as h5:
            qx = h5["qx"][:]
            qy = h5["qy"][:]
            i_noisy = h5["I_noisy"][:]

        primary_v, sols, winner, _, rg_in, yg100, yg210, ym210 = tenor_process_landscape(
            i_noisy,
            qx,
            qy,
            gt_lib,
            instrument,
            simulation,
            ensemble,
        )

        rows.append(
            {
                "Index": idx + 1,
                "Filename": filename,
                "Filepath": str(file_path),
                "Noise": noise,
                "True_V": true_v,
                "Primary_V": primary_v,
                "Winner": winner,
                "Rg_meas": rg_in,
                "Yg100": yg100,
                "Yg210": yg210,
                "Ym210": ym210,
                "V_all": _normalize_v_all(primary_v, primary_v, sols["Alternatives"]),
                "Instrument_SD_dist_cm": instrument.SD_dist,
                "Instrument_lambda_nm": instrument.lambda_,
                "Instrument_det_side_cm": instrument.det_side,
                "Instrument_DETpix": instrument.DETpix,
                "Simulation_signum": simulation.signum,
                "Simulation_use_r3": simulation.use_r3,
                "Simulation_use_g3": simulation.use_g3,
                "Ensemble_rg_input_nm": ensemble.rg,
                "Ensemble_nu": ensemble.nu,
                "Ensemble_dist_name": ensemble.d_nam,
                "Ensemble_dist_N": ensemble.dist_param["N"],
                "Matlab_Primary_V": matlab_row["Primary_V"],
                "Matlab_Winner": matlab_row["Winner"],
                "Matlab_Rg_meas": matlab_row["Rg_meas"],
                "Matlab_Yg100": matlab_row["Yg100"],
                "Matlab_Yg210": matlab_row["Yg210"],
                "Matlab_Ym210": matlab_row["Ym210"],
            }
        )

    results_df = pd.DataFrame(rows)
    csv_path = output_dir / "python_analysis_on_matlab_h5.csv"
    pkl_path = output_dir / "python_analysis_on_matlab_h5.pkl"
    figure_path = output_dir / "python_violin_on_matlab_h5.png"

    results_df.to_csv(csv_path, index=False)
    results_df.to_pickle(pkl_path)
    plot_tenor_violin_closest(results_df, instrument, output_path=figure_path)

    abs_err = np.abs(results_df["Primary_V"] - results_df["True_V"])
    finite = np.isfinite(abs_err)
    print(f"Saved results table to {csv_path}")
    print(f"Saved pickle table to {pkl_path}")
    print(f"Saved violin figure to {figure_path}")
    print(
        {
            "rows": len(results_df),
            "coverage": float(np.mean(np.isfinite(results_df["Primary_V"]))),
            "mae": float(np.mean(abs_err[finite])) if np.any(finite) else float("nan"),
        }
    )


if __name__ == "__main__":
    main()
