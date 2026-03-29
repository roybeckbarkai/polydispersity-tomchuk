SAXS Simulator and Tomchuk Polydispersity App

A Streamlit app for simulating and analyzing small-angle X-ray scattering (SAXS) data for polydisperse spheres and fixed-length polymers.

## Features

- Simulates 2D and 1D SAXS patterns for several sphere size distributions.
- Supports Gaussian, Lognormal, Schulz, Boltzmann, Triangular, and Uniform distributions.
- Includes Tomchuk-style invariant analysis for highly polydisperse spheres.
- Includes NNLS-based distribution recovery for spheres and IDP-style polymer mode.
- Exports fitted intensity curves and recovered distributions to CSV.

## Requirements

- Python 3.11+ recommended
- macOS, Linux, or another environment that can run Streamlit

## Quick Start

From the app folder:

```bash
cd "/Users/roybeck/Library/CloudStorage/Dropbox/python code copy/polydispersity app"
./run_app.sh
```

The script will:

- create a local virtual environment in `.venv` if needed
- install or update the Python requirements
- launch the Streamlit app

Then open:

```text
http://localhost:8501
```

## Manual Setup

If you prefer to run everything manually:

```bash
cd "/Users/roybeck/Library/CloudStorage/Dropbox/python code copy/polydispersity app"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

streamlit run streamlit_app.py
```

If port `8501` is already in use:

```bash
streamlit run streamlit_app.py --server.port 8502
```

## Validation

To run the Tomchuk validation script:

```bash
cd "/Users/roybeck/Library/CloudStorage/Dropbox/python code copy/polydispersity app"
source .venv/bin/activate
python validate_tomchuk.py
```

## Main Files

- `streamlit_app.py`: app entry point
- `single_mode.py`: single-run interactive UI
- `batch_mode.py`: batch processing UI
- `analysis_utils.py`: analysis and Tomchuk recovery logic
- `sim_utils.py`: simulation and scattering kernels
- `validate_tomchuk.py`: validation script for simulated sphere recovery

