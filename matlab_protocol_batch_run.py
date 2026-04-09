from __future__ import annotations

from pathlib import Path

from matlab_protocol.tenor_analysis import batch_run_protocol


def main():
    repo_root = Path(__file__).resolve().parent
    matlab_output_dir = Path("/Users/roybeck/Library/CloudStorage/Dropbox/python code copy/tomchuk study files/matlab-files-python")
    violin_path = repo_root / "matlab_protocol_violin.png"
    result = batch_run_protocol(
        output_dir=matlab_output_dir,
        gt_library_path=repo_root / "matlab-new" / "GroundTruth_sphere_rg5_lognormal.mat",
        plot_output_path=violin_path,
    )
    print(result["Results_Table"].head())
    print(f"Saved violin plot to {violin_path}")


if __name__ == "__main__":
    main()
