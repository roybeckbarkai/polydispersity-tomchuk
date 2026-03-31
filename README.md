SAXS Simulator and Tomchuk Polydispersity App
=============================================

This repository contains a Streamlit application for simulating and analyzing small-angle X-ray scattering (SAXS) data for:

- polydisperse spheres
- fixed-length polymer / IDP-style scattering

The app supports three analysis paths for spheres:

- `Tomchuk (Invariants)`
- `NNLS (Distribution Fit)`
- `Tenor-SAXS (2D Observable Extraction)`

For IDP mode, the app uses `NNLS` only.


Tomchuk Extraction Path In This Code
------------------------------------

The Tomchuk branch now uses a shared, explicit extraction path in `analysis_utils.py`:

1. estimate `Rg` and `G` from a Guinier window
2. fit the unified Beaucage / Tomchuk-style intensity model
3. if that fit is valid, use fitted `G`, `Rg`, and `B`
4. compute `Q` and `lc` analytically from the fitted model
5. recover `p` from `PDI` and `PDI2`

The older hybrid path is still kept internally for diagnostics:

- `Rg` and `G` from Guinier
- `B` from the measured high-q tail
- `Q` and `lc` from finite-q integration plus tail correction

For Tomchuk analysis, the app now prefers the unified-fit path when it is available, and records the choice as:

- `tomchuk_extraction = unified_fit`
- or `tomchuk_extraction = hybrid`

This makes it easier to inspect whether failures come from the fitted model itself or from the older mixed extraction route.


Normalization Of Simulated Data
-------------------------------

For simulated sphere runs in Tomchuk mode, the app now normalizes the simulated 1D profile before analysis.

Why this is needed:

- the simulator now defines `flux` as the expected number of photons in the nearest-to-center detector pixel after smearing and before Poisson noise
- total detector counts still change with `q_max`, detector size, smearing, and the distribution shape because those settings change how much intensity is spread away from the beam center
- that makes raw amplitude-carrying quantities such as `G`, `B`, and `Q` move when detector settings are changed unless the 1D profile is normalized before Tomchuk extraction
- that behavior is acceptable for raw detector counts, but it is not what we want in the theory-comparison table

What the app does now:

- it computes the scale that maps the input distribution’s theoretical 1D intensity shape onto the simulated 1D profile
- it divides the simulated 1D profile by that scale before Tomchuk extraction
- the extracted amplitudes are then compared against input-based theoretical values on the same normalization

As a result:

- the theory column in single mode now depends only on the declared input `mean_rg`, `p`, and distribution family
- forward-pixel photon count no longer changes extracted `G`, `B`, or `Q` after normalization
- remaining changes with `q_max`, detector size, or smearing reflect extraction bias or detector-induced distortion rather than a simple scale mismatch

Forward Pixel Photon Definition
-------------------------------

The simulator now uses the following convention for `flux`:

- `flux` = expected photons in the nearest-to-center detector pixel
- this scaling is applied after detector smearing and before optional Poisson sampling
- for even detector sizes such as `1024 x 1024`, there is no exact `q = 0` pixel, so the code uses pixel index `(N//2, N//2)`, one of the four pixels nearest to the beam center

This is different from the older convention used in the first benchmark study:

- old convention: `flux = total expected photons summed over the whole detector`
- new convention: `flux = expected photons in the center-nearest detector pixel`

Because the ratio

- `center pixel photons / total detector photons`

depends strongly on distribution family, `p`, and smearing, the old benchmark cannot be converted to the new definition with one global scale factor.


New UI Diagnostics
------------------

Single mode now exposes three extra Tomchuk diagnostics:

- `Tomchuk Path`: shows whether the current analysis used `unified_fit` or `hybrid`
- `Reconstructed Fit`: shows relative RMS error for the SAXS curves reconstructed from the recovered `p` values from `PDI` and `PDI2`
- `Recommended q / bins`: an on-demand sweep that searches for a practical `q_max` / `1D bins` region for the current settings

The reconstructed-fit block is intended to answer a simple question:

- if I take the extracted scattering `Rg`, combine it with the recovered `p`, reconstruct the corresponding distribution, and calculate the SAXS curve again, does that curve still agree with the measured or simulated data?

The quality labels are:

- `strong`: `RelRMS <= 0.02`
- `usable`: `RelRMS <= 0.05`
- `weak`: `RelRMS <= 0.10`
- `poor`: `RelRMS > 0.10`

These labels are intentionally conservative.


Tenor-SAXS In This Code
-----------------------

The Tenor branch is implemented in `tenor_saxs.py` and integrated into the
shared pipeline through `analysis_utils.py`.

Current extraction flow:

1. estimate the apparent Guinier `Rg` from the same shared 2D->1D reducer used elsewhere in the app
2. scan a family of anisotropic digital PSF quartets
3. fit the log-ratio observable for each quartet using the MATLAB-style weighted centered model
4. reject quartets whose recovered `J_G` is physically implausible for the selected distribution family
5. calibrate surviving candidates against forward simulations using the same simulator settings as the app
6. optionally re-simulate several candidate quartets and choose the one whose reconstructed 2D data best matches the raw 2D data

Important practical notes:

- Tenor requires a 2D detector image
- the app derives the Tenor Guinier bin count internally from the main `1D Bins` setting
- the Tenor calibration now reuses the same section-4-style forward simulation path that is also used for its noise-free ground-truth comparison
- for spheres, the input `p` is the number-density polydispersity, while Tenor's variance observable is scattering-weighted; the code converts the recovered weighted variance back to the corresponding number-density `p`


Tenor Stability And Failure Handling
------------------------------------

The Tenor branch can become unstable in noisy or low-flux conditions.

The main failure signature is:

- no physically plausible quartet survives the `J_G` plausibility filter

When that happens, the app now:

- marks the run as unstable
- shows a warning in the UI
- reports the main recovered Tenor quantities as `n/a` instead of presenting unreliable values as normal results

The most useful settings to try first are:

- increase photon statistics / flux
- increase `TENOR Recon Trials` from `1` to about `3-5`

The settings below usually have smaller effect by themselves:

- `TENOR Calibration Points`
- `TENOR PSF Pair Count`

In the current implementation, random quartet selection is not used by default,
because randomizing among bad candidates does not solve the core failure mode.
The main bottleneck is whether the data support any physically plausible
observable extraction at all.

For a focused discussion of how this implementation differs from the original
text protocol, see:

- `TENOR_STABILITY_NOTES.md`


Persistent GUI Settings
-----------------------

The app now stores its current defaults in `app_settings.json`.

This means:

- changing a widget updates the saved settings
- leaving and returning to Home preserves the latest values
- restarting the app reloads the last saved values

The JSON file stores:

- the current value for each parameter
- a short `_comment` describing what that parameter controls


Instrument Geometry In The Current Simulator
--------------------------------------------

The simulator is now geometry-driven instead of using a fixed detector-size
shortcut.

The relevant inputs are:

- detector pixels
- pixel size in micrometers
- sample-to-detector distance
- wavelength

From these values the code derives the detector-limited `q_max`.

In single mode:

- `Analysis q max` is derived automatically from the actual data
- for simulated data it is the instrument-limited detector maximum
- for uploaded 1D data it is the largest measured `q`


Simulation Controls
-------------------

Several simulation controls are now explicit in the GUI instead of being hidden
"magic numbers".

The most important ones are:

- `radius_samples`
- `q_samples`
- `ensemble_sampling`
- `ensemble_members`
- forward-model choice
- anisotropic smearing in `x` and `y`

Interpretation:

- `radius_samples`: support size grid used to represent the ensemble before simulation
- `q_samples`: internal q grid used before interpolation onto the detector
- `ensemble_members`: only matters in `Discrete` ensemble mode
- `phi2` and `phi3`: only matter when `Forward Model = Guinier Curvature`

For exact sphere simulations:

- you do not need to supply `phi2` or `phi3`
- the exact-sphere form factor already defines the q dependence


What Is In The Repository
-------------------------

- `streamlit_app.py`: main entry point
- `single_mode.py`: interactive single-run UI
- `batch_mode.py`: batch queue UI
- `analysis_utils.py`: shared analysis, export, and batch/validation helpers
- `sim_utils.py`: simulation kernels and distribution models
- `tenor_saxs.py`: standalone Tenor-SAXS extraction, calibration, and truth helpers
- `app_settings.py` / `app_settings.json`: persisted GUI defaults and descriptions
- `validate_tomchuk.py`: command-line validation benchmark for Tomchuk recovery
- `validate_tenor_saxs.py`: command-line validation benchmark for Tenor-SAXS recovery
- `TENOR_STABILITY_NOTES.md`: notes on Tenor stability handling and deviations from the paper workflow
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

For better reload performance in Streamlit, `watchdog` is now included in
`requirements.txt`.


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

It can also run an automatic recommendation sweep for `q_max` and `1D bins`.

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

Run the higher-resolution benchmark discussed below:

```bash
python validate_tomchuk.py --distributions Gaussian --p-values 0.7 --mean-rg 4.0 --q-max 4.0 --n-bins 1024 --pixels 1024 --smearing 2.0 --flux 1e12 --max-rel-error 0.2
```

Run the higher-resolution multi-distribution sweep:

```bash
python validate_tomchuk.py --distributions Gaussian Lognormal Schulz Boltzmann Triangular Uniform --p-values 0.7 --mean-rg 4.0 --q-max 4.0 --n-bins 1024 --pixels 1024 --smearing 2.0 --flux 1e12 --max-rel-error 2.0
```

Run the recommendation evaluator:

```bash
python validate_tomchuk.py --distributions Gaussian --p-values 0.3 --mean-rg 2.0 --q-max 2.5 --n-bins 256 --pixels 1024 --smearing 1.0 --flux 1e12 --recommend-settings --target-abs-error 0.01 --max-rel-error 1.0
```


Large Benchmark Study
---------------------

A dedicated end-to-end study runner is available:

```bash
python run_tomchuk_benchmark_study.py --workers 6 --output-root study_outputs
```

Current benchmark assumptions in that script:

- `mean_rg = 2.0 nm`
- `q_max = 5.0 nm^-1`
- `pixels = 1024`
- `n_bins = 1024`
- logarithmic binning
- all six sphere distribution families
- `p = 0.1 .. 1.0`
- flux exponents `5 .. 9`
- smearing `1 .. 10`
- five replicates per condition

Outputs are written into a dedicated timestamped folder under `study_outputs/` and include:

- one TIFF detector image per run
- one 1D CSV per run
- a master summary CSV for all runs
- aggregated summary CSV files
- generated figures
- a LaTeX appendix and compiled PDF report


Current Test Results
--------------------

The following results were obtained from the current code in this repository.

Important note:

- `p_input` in the validator is the sidebar / command-line input width
- the deeper sanity check compares extracted quantities against the actual simulated distribution moments
- those two are not always numerically identical because the simulated distribution is represented on a finite radius grid and some families are truncated at small radius
- for that reason, the sanity layer is the better test of whether `G`, `Rg`, `B`, `Q`, `lc`, `PDI`, and `PDI2` were extracted consistently from the simulated curve

Additional note on theory comparisons:

- single mode now uses input-normalized theory for the extracted/theory table
- for simulated sphere runs, the 1D profile is normalized before Tomchuk analysis so `G`, `B`, and `Q` can be compared directly to theory

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


Recommendation Evaluator
------------------------

The recommendation evaluator uses the current Tomchuk pipeline and sweeps a grid of:

- `q_max`
- `1D bins`

For each combination it records:

- recovered `p` from `PDI`
- recovered `p` from `PDI2`
- absolute and relative error in `p`
- relative RMS error of the reconstructed SAXS fits from `PDI` and `PDI2`
- whether the recovered values pass the requested target absolute error

It then reports:

- the single best combination by worst-case absolute `p` error
- a safety zone around the best combination
- the number of combinations that hit the requested target for both `PDI` and `PDI2`

This is available in two places:

- the single-mode Streamlit UI through the `Evaluate q-range / bins` button
- the CLI validator through `--recommend-settings`


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

- raising photon count helps the extraction stabilize quickly
- after about `1e8` to `1e10` photons in the current Gaussian benchmark, additional count changes little
- remaining error is dominated by model / inversion bias, not counting noise


6. Higher-resolution Tomchuk benchmark

Benchmark setup:

- `distribution = Gaussian`
- `p = 0.7`
- `mean_rg = 4.0 nm`
- `pixels = 1024`
- `n_bins = 1024`
- `smearing = 2.0`
- `flux = 1e12`
- `q_max = 4.0`

Command:

```bash
python validate_tomchuk.py --distributions Gaussian --p-values 0.7 --mean-rg 4.0 --q-max 4.0 --n-bins 1024 --pixels 1024 --smearing 2.0 --flux 1e12 --max-rel-error 0.2
```

Observed result:

- extraction path selected: `unified_fit`
- recovered `p = 0.6177` from `PDI`
- recovered `p = 0.7070` from `PDI2`
- relative error: `-11.8%` for `PDI`, `+1.0%` for `PDI2`
- sanity errors for the extracted invariants stayed small:
  - `Rg: -2.4%`
  - `G: -1.6%`
  - `B: +4.4%`
  - `Q: +1.7%`
  - `lc: -0.6%`
  - `PDI2: +2.1%`

Interpretation:

- at this higher-resolution setting, the code is extracting the Tomchuk invariants much more cleanly than before
- `PDI2` works well in this benchmark
- `PDI` still shows noticeable bias
- this suggests the main residual weakness is now the `PDI -> p` inversion for some families, not the raw extraction of `G`, `Rg`, `B`, `Q`, and `lc`


7. Photon-count sweep at higher resolution

Benchmark setup:

- `distribution = Gaussian`
- `p = 0.7`
- `mean_rg = 4.0 nm`
- `pixels = 1024`
- `n_bins = 1024`
- `smearing = 2.0`
- `q_max = 4.0`
- `noise = True`
- `flux = 1e6, 1e8, 1e10, 1e12`

Observed behavior:

- `1e6`: `p(PDI) = 0.5891`, `p(PDI2) = 0.6700`
- `1e8`: `p(PDI) = 0.6159`, `p(PDI2) = 0.7048`
- `1e10`: `p(PDI) = 0.6175`, `p(PDI2) = 0.7068`
- `1e12`: `p(PDI) = 0.6177`, `p(PDI2) = 0.7070`

Interpretation:

- the result stabilizes quickly as flux increases
- by `1e8` to `1e10`, the current Gaussian benchmark is effectively count-limited no longer
- the remaining `PDI` bias is systematic
- the `PDI2` route is the more reliable Tomchuk recovery path in this benchmark


8. Higher-resolution multi-distribution sweep

Benchmark setup:

- `p = 0.7`
- `mean_rg = 4.0 nm`
- `pixels = 1024`
- `n_bins = 1024`
- `smearing = 2.0`
- `q_max = 4.0`
- `flux = 1e12`

Command:

```bash
python validate_tomchuk.py --distributions Gaussian Lognormal Schulz Boltzmann Triangular Uniform --p-values 0.7 --mean-rg 4.0 --q-max 4.0 --n-bins 1024 --pixels 1024 --smearing 2.0 --flux 1e12 --max-rel-error 2.0
```

Observed result summary:

- `Gaussian`: `PDI2` works well, `PDI` still runs low
- `Lognormal`: `PDI2` is acceptable, `PDI` fails badly low
- `Schulz`: both `PDI` and `PDI2` remain unreliable in this setting
- `Boltzmann`: `PDI2` is acceptable, `PDI` fails badly low
- `Triangular`: `PDI` collapses to zero, `PDI2` stays biased high
- `Uniform`: `PDI` collapses to zero, `PDI2` stays biased high

Why the failures happen:

- for `Gaussian` and often `Boltzmann`, the extracted invariants are already fairly good, but the `PDI -> p` mapping remains the weak step
- for `Lognormal`, `Schulz`, `Triangular`, and `Uniform`, the recovered mean size and `PDI` pathway remain unstable enough that higher detector resolution and higher count alone do not fix the inversion
- triangular and uniform families are especially fragile because small shifts in the extracted invariants move the recovered `p` sharply
- this means that better data helps, but it does not guarantee full recovery for every declared family under the current inversion formulas


9. Recommendation example for the default low-p Gaussian case

Benchmark setup:

- `distribution = Gaussian`
- `mean_rg = 2.0 nm`
- `p = 0.3`
- `pixels = 1024`
- `smearing = 1.0`
- `q_min = 0`
- `binning = logarithmic`
- `noise = False`
- `flux = 1e12`
- target absolute `p` error for both `PDI` and `PDI2`: `0.01`

Command:

```bash
python validate_tomchuk.py --distributions Gaussian --p-values 0.3 --mean-rg 2.0 --q-max 2.5 --n-bins 256 --pixels 1024 --smearing 1.0 --flux 1e12 --recommend-settings --target-abs-error 0.01 --max-rel-error 1.0
```

Observed result:

- best compromise found by the sweep: `q_max = 2.0`, `bins = 2048`
- recovered `p(PDI) = 0.3832`
- recovered `p(PDI2) = 0.2127`
- best-case worst absolute error in `p`: `0.0873`
- safety zone: `q_max = 2.0`, `bins = 128..2048`
- target hits at `0.01`: `0`

Interpretation:

- for this low-polydispersity default case, the current Tomchuk implementation does not produce a region where both `PDI` and `PDI2` recover `p` to within `0.01`
- the recommendation tool is therefore useful not only for finding good settings, but also for showing honestly when no truly accurate zone exists
- this is consistent with the known limitation that Tomchuk analysis is much more reliable for more strongly polydisperse sphere systems


10. Validation of the simulated-data normalization

Benchmark setup:

- `distribution = Gaussian`
- `mean_rg = 4.0 nm`
- `p = 0.7`
- `mode = Sphere`
- `method = Tomchuk`
- `noise = False`

Validation result:

After normalization of the simulated 1D profile:

- changing `flux` no longer changes extracted `G`, `B`, or `Q`
- the same case run at `flux = 1e6, 1e8, 1e10, 1e12` returns the same extracted amplitudes to numerical precision

Example:

- `flux = 1e6` -> `G = 7.446e6`, `B = 3.530e3`, `Q = 3.123e4`
- `flux = 1e12` -> `G = 7.446e6`, `B = 3.530e3`, `Q = 3.123e4`

What still changes after normalization:

- changing `q_max`
- changing detector size
- changing smearing

These still affect extracted values somewhat because they change the sampled curve and therefore the quality of the Guinier / unified / invariant extraction, not because of a raw scale mismatch.

Observed behavior for the same `Gaussian, mean_rg = 4.0, p = 0.7` case:

- `q_max` changes extracted `Rg`, `G`, `B`, `Q`, `lc`, `PDI`, and `PDI2` modestly
- detector size changes them modestly as well
- smearing also shifts them, especially the amplitude terms and the recovered `PDI`

What does not change the input-based theory:

- `q_max`
- detector size
- smearing
- flux
- noise setting
- binning choice

The input-based theoretical Tomchuk quantities depend only on:

- `mean_rg`
- `p`
- distribution family

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
