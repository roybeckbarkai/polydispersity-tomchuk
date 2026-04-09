from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd

from matlab_protocol.matlab_results import load_matlab_results_table
from matlab_protocol.params import init_tenor_params
from matlab_protocol.tenor_analysis import load_gt_library, tenor_process_landscape


def main(limit: int = 25):
    repo_root = Path(__file__).resolve().parent
    matlab_table = load_matlab_results_table(repo_root / "matlab-new" / "sphere_r5_multinoise_lognormal.mat")
    instrument, simulation, ensemble, _ = init_tenor_params()
    gt = load_gt_library(repo_root / "matlab-new" / "GroundTruth_sphere_rg5_lognormal.mat")
    h5_root = Path("/Users/roybeck/Library/CloudStorage/Dropbox/python code copy/tomchuk study files/matlab-files")

    rows = []
    for i in range(min(limit, len(matlab_table))):
        true_v = float(matlab_table.loc[i, "True_V"])
        fname = h5_root / f"sim_{i + 1:04d}_V{true_v:0.3f}.h5"
        with h5py.File(fname, "r") as f:
            qx = f["qx"][:]
            qy = f["qy"][:]
            i_noisy = f["I_noisy"][:]
        primary_v, sols, winner, _, rg_in, yg100, yg210, ym210 = tenor_process_landscape(
            i_noisy, qx, qy, gt, instrument, simulation, ensemble
        )
        rows.append(
            {
                "idx": i,
                "file": fname.name,
                "mat_primary": matlab_table.loc[i, "Primary_V"],
                "py_primary": primary_v,
                "mat_winner": matlab_table.loc[i, "Winner"],
                "py_winner": winner,
                "mat_rg": matlab_table.loc[i, "Rg_meas"],
                "py_rg": rg_in,
                "mat_y100": matlab_table.loc[i, "Yg100"],
                "py_y100": yg100,
                "mat_y210": matlab_table.loc[i, "Yg210"],
                "py_y210": yg210,
                "mat_ym210": matlab_table.loc[i, "Ym210"],
                "py_ym210": ym210,
            }
        )

    comp = pd.DataFrame(rows)
    out_csv = repo_root / "matlab_protocol_comparison.csv"
    comp.to_csv(out_csv, index=False)
    print(comp.to_string(index=False))
    print(f"\nSaved comparison to {out_csv}")


if __name__ == "__main__":
    main()
