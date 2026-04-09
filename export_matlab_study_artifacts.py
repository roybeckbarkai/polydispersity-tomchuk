from __future__ import annotations

from pathlib import Path

from matlab_protocol.matlab_results import load_matlab_results_table
from matlab_protocol.params import init_tenor_params
from matlab_protocol.plotting import plot_tenor_violin_closest


def main():
    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / "matlab_protocol_outputs"
    output_dir.mkdir(exist_ok=True)

    matlab_table = load_matlab_results_table(repo_root / "matlab-new" / "sphere_r5_multinoise_lognormal.mat")
    instrument, _, _, _ = init_tenor_params()

    csv_path = output_dir / "matlab_saved_results_table.csv"
    pkl_path = output_dir / "matlab_saved_results_table.pkl"
    figure_path = output_dir / "matlab_saved_violin.png"

    matlab_table.to_csv(csv_path, index=False)
    matlab_table.to_pickle(pkl_path)
    plot_tenor_violin_closest(matlab_table, instrument, output_path=figure_path)

    print(f"Saved MATLAB table CSV to {csv_path}")
    print(f"Saved MATLAB table pickle to {pkl_path}")
    print(f"Saved MATLAB violin figure to {figure_path}")


if __name__ == "__main__":
    main()
