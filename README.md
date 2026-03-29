SAXS Simulator and Tomchuk Polydispersity App
=============================================

This repository contains a Streamlit application for simulating and analyzing small-angle X-ray scattering (SAXS) data for:

- polydisperse spheres
- fixed-length polymer / IDP-style scattering

The app supports two analysis paths for spheres:

- `Tomchuk (Invariants)`
- `NNLS (Distribution Fit)`

For IDP mode, the app uses `NNLS` only.


What Is In The Repository
-------------------------

- `streamlit_app.py`: main entry point
- `single_mode.py`: interactive single-run UI
- `batch_mode.py`: batch queue UI
- `analysis_utils.py`: shared analysis, export, and batch/validation helpers
- `sim_utils.py`: simulation kernels and distribution models
- `validate_tomchuk.py`: command-line validation benchmark for Tomchuk recovery
- `run_app.sh`: convenience launcher for macOS/Linux shells


Quick Start
-----------

From the app folder:

```bash
cd "/Users/roybeck/Library/CloudStorage/Dropbox/python code copy/polydispersity app"
./run_app.sh
```

The script will:

- create `.venv` if it does not exist
- install the Python requirements
- launch Streamlit

Then open:

```text
http://localhost:8501
```


Manual Setup
------------

```bash
cd "/Users/roybeck/Library/CloudStorage/Dropbox/python code copy/polydispersity app"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

streamlit run streamlit_app.py
```

If port `8501` is busy:

```bash
streamlit run streamlit_app.py --server.port 8502
```


Batch Mode
----------

Batch mode is now aligned more closely with `single_mode.py`:

- it uses the same shared simulation-and-analysis path from `analysis_utils.py`
- it uses the same recovered-distribution construction logic as single mode
- it uses the same size/error bookkeeping as the validator
- it treats `IDP` rows as `NNLS` analysis only, matching the single-mode UI

Batch CSV columns:

- `mode (S/P)`: `S` for spheres, `P` for IDP
- `dist (G/L/S/B/T/U)`: Gaussian, Lognormal, Schulz, Boltzmann, Triangular, Uniform
- `mean_rg`
- `p_val`
- `pixels`
- `q_min`
- `q_max`
- `n_bins`
- `binning (Log/Lin)`
- `smearing`
- `flux`
- `noise`
- `method (T/N)`: `T` Tomchuk, `N` NNLS
- `nnls_max_rg`

Lists such as `[0.3, 0.7, 1.0]` are expanded into multiple jobs.


Validation
----------

The validator uses the same shared helper path used by batch mode:

- `run_simulation_analysis_case(...)`
- `build_summary_row(...)`

Run the default Tomchuk benchmark:

```bash
cd "/Users/roybeck/Library/CloudStorage/Dropbox/python code copy/polydispersity app"
source .venv/bin/activate
python validate_tomchuk.py
```

Run the same benchmark with Poisson noise:

```bash
python validate_tomchuk.py --noise --flux 1e12
```

Run a broader, exploratory sweep across several distributions:

```bash
python validate_tomchuk.py --distributions Gaussian Lognormal Schulz Boltzmann Triangular Uniform --p-values 0.7 --max-rel-error 1.0
```


Current Test Results
--------------------

The following results were obtained from the current code in this repository.

1. Default validation benchmark

Command:

```bash
python validate_tomchuk.py
```

Result:

- `Gaussian, p = 0.7` -> recovered `p = 0.8301` by `PDI`, `p = 0.8394` by `PDI2`
- `Gaussian, p = 1.0` -> recovered `p = 1.0463` by `PDI`, `p = 0.8438` by `PDI2`
- max relative error: `0.1858` for `PDI`, `0.1991` for `PDI2`
- validator status: `PASS` at threshold `0.20`

2. Default validation benchmark with noise

Command:

```bash
python validate_tomchuk.py --noise --flux 1e12
```

Result:

- `Gaussian, p = 0.7` -> recovered `p = 0.8301` by `PDI`, `p = 0.8394` by `PDI2`
- `Gaussian, p = 1.0` -> recovered `p = 1.0461` by `PDI`, `p = 0.8438` by `PDI2`
- max relative error: `0.1858` for `PDI`, `0.1991` for `PDI2`
- validator status: `PASS` at threshold `0.20`

3. Direct shared-helper check used by batch mode

Input:

- `mode = Sphere`
- `distribution = Gaussian`
- `mean_rg = 4.0`
- `p = 0.7`
- `q_max = 0.8`
- `flux = 1e12`
- `noise = False`

Result:

- `Recovered_p = 0.8033`
- `Recovered_p_PDI2 = 0.8191`
- `Recovered_Size = 4.6377 nm`
- `Recovered_Size_PDI2 = 4.5685 nm`

4. Exploratory multi-distribution sweep

Command:

```bash
python validate_tomchuk.py --distributions Gaussian Lognormal Schulz Boltzmann Triangular Uniform --p-values 0.7 --max-rel-error 1.0
```

Observed behavior:

- `Gaussian`: the most stable overall, but still biased high in the current extraction path
- `Boltzmann`: one of the better non-Gaussian cases, especially for `PDI2`
- `Lognormal`: `PDI2` is clearly better than `PDI`
- `Schulz`: `PDI` is acceptable, `PDI2` can overshoot badly
- `Triangular` and `Uniform`: currently unreliable

This sweep is included here to document current behavior honestly, not as a claim of full recovery accuracy across all supported families.


Sanity Check Layer
------------------

The validator now performs two checks for every synthetic sphere run:

- recovery error for `p` from `PDI` and `PDI2`
- sanity comparison of extracted Tomchuk quantities against the known simulated distribution

When a sanity check fails, the validator prints a targeted suggestion:

- `Rg` or `G` failures point to the low-q Guinier fit window
- `B` failures point to the high-q Porod tail estimate
- `Q`, `lc`, or `PDI2` failures point to the invariant extraction and tail-correction path
- `PDI` or `p_rec_pdi` failures point to the combined `G/Rg/B` extraction path
- mean-size failures point to the moment-to-size conversion for the selected distribution

Batch mode now carries the same sanity fields in the exported summary CSV for sphere jobs:

- `Sanity_Pass`
- `Sanity_Failures`
- `Sanity_Suggestions`
- `Sanity_RelErr_*` columns for the extracted quantities


5. Flux sweep at fixed geometry

Benchmark setup:

- `mean_rg = 4.0 nm`
- `pixels = 1024`
- `smearing = 2.0`
- `q_max = 0.8`
- `p = 0.7`
- `noise = True`
- `flux = 1e6, 1e8, 1e10, 1e12`

Observed behavior from the current code:

- `Gaussian`: `PDI` moves from `0.5179` at `1e6` to `0.8630` at `1e12`; `PDI2` stays around `0.863-0.871`. High count removes much of the scatter but not the systematic bias.
- `Lognormal`: `PDI` stays near `0.515` while `PDI2` stays near `0.774` almost independent of flux. This looks like extraction-path bias, not photon limitation.
- `Schulz`: `PDI` stays close to the true value near `0.72`, but `PDI2` remains badly high around `1.10-1.15`.
- `Boltzmann`: `PDI` is unstable at `1e6` and then settles around `0.85`; `PDI2` is more stable around `0.778`, but both remain biased high relative to the true `p = 0.7`.
- `Triangular`: `PDI` collapses to `0.0` for `1e8` and above, which is a hard failure in the current implementation. `PDI2` is repeatable near `0.986`, but still substantially biased.
- `Uniform`: both `PDI` and `PDI2` remain unstable and inconsistent across the sweep.
- Increasing photon count improves numerical stability, but it does not remove the dominant bias because the main limitation is the current extraction path, not counting noise alone.

Sanity-check interpretation:

- `mean_radius_pdi` and `mean_radius_pdi2` failures indicate the moment-to-size back-conversion is not matching the known simulated distribution closely enough.
- `p_rec_pdi` failures point to a mismatch in the `G/Rg/B -> PDI -> p` path.
- `p_rec_pdi2` failures point to a mismatch in the `Q/lc/B -> PDI2 -> p` path.
- When `PDI2` fails for cases such as `Schulz`, the likely issue is the current `Q` and `lc` extraction strategy from the finite simulated curve, not the detector count itself.

Practical takeaway:

- Higher photon count helps, but it is not the main fix if `PDI` or `PDI2` are consistently biased.
- If the sanity checker flags `PDI2`, inspect the `Q/lc` integration and tail correction path first.
- If the sanity checker flags `PDI`, inspect the `Rg`, `G`, and `B` extraction consistency against the simulated curve first.
- For the current code, the most defensible high-count cases are `Gaussian` and `Boltzmann`; `Triangular` and `Uniform` are the weakest.


Important Deviations From The Original Tomchuk Paper
----------------------------------------------------

The app is inspired by:

- O. V. Tomchuk et al., "Particle-size polydispersity analysis based on the unified exponential/power-law approach to small-angle scattering", J. Appl. Cryst. (2023), 56, 1099-1107.

The current implementation intentionally deviates from the paper in several ways.

1. Distribution type is user-selected

The paper proposes a workflow where `PDI` and `PDI2` can help infer the likely distribution family through a `PDI` vs `PDI2` comparison.

This app does not yet implement that unknown-distribution classification step.
Instead, the user declares the distribution type in advance, and recovery is performed within that chosen family.

2. `G`, `Rg`, `B`, `Q`, and `lc` are extracted directly from the data path

Per current project direction, the code extracts these quantities directly from the scattering data workflow rather than enforcing a strict Beaucage-fit-only pipeline for final parameter extraction.

In practice, the current Tomchuk path:

- estimates `Rg` and `G` from a Guinier-style fit
- estimates `B` from the high-q `I(q) q^4` tail
- estimates `Q` and `lc` from direct numerical integration with tail corrections

This differs from a strict implementation of the paper’s fully unified-fit-based route.

3. The app is intended for highly polydisperse sphere cases

The paper notes that the method is best suited for highly polydisperse systems, and the app warns when `p < 0.25` in Tomchuk sphere mode.

4. Exact recovery is not guaranteed across all supported distributions

The current implementation works reasonably on the benchmarked high-polydispersity Gaussian sphere cases, but broader non-Gaussian sweeps remain mixed.
That limitation is real and should be kept in mind when interpreting results.


Practical Notes
---------------

- Use `Tomchuk` primarily for highly polydisperse sphere cases.
- Use `NNLS` when you want a more general recovered distribution rather than invariant-based Tomchuk indices.
- In batch mode, `IDP` rows are analyzed with `NNLS` even if `T` is supplied in the CSV.
- The validator is the quickest way to check whether a code change improved or degraded current Tomchuk recovery behavior.
