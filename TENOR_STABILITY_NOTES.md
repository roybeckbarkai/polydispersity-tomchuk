TENOR Stability Notes
=====================

Purpose
-------

This note explains how the current Python Tenor-SAXS implementation differs
from the original text protocol in `TenorSAXS.pdf`, especially in unstable
or noisy conditions.


Why This Note Exists
--------------------

In ideal, noise-free conditions the current implementation reproduces the
expected Tenor behavior well. In noisy conditions, however, the quartet
selection stage can become unstable.

The main failure mode is:

- every scanned PSF quartet gives an observable outside the physically
  plausible `J_G` range for the selected distribution family

If we still force one of those quartets to win, the recovered variance and
mean size can become badly wrong.


What The Original Text Emphasizes
---------------------------------

Section 4 of the text focuses on:

1. extracting an apparent Guinier size
2. scanning PSF quartets
3. choosing the quartet that yields the best observable fit
4. calibrating the chosen observable against forward simulations

The text also notes that:

- PSF size strongly affects performance
- larger PSFs can help with noise
- the best quartet should be evaluated by fit quality
- in practice the choice should remain inside the Guinier region

The MATLAB code goes further than a plain residual score:

- it computes confidence-interval-based grades for the extracted ratios
- this gives an uncertainty-aware ranking of candidate quartets


What The Python Code Does Differently
-------------------------------------

The Python app adds several safeguards that are not spelled out as-is in the
paper text.

1. Physical plausibility filter

- after extracting `J_G` for each quartet, the code checks whether it lies in
  the physically plausible range implied by the selected distribution family
  and the calibration p-grid
- quartets outside that range are treated as implausible

2. Shared forward calibration path

- the Tenor calibration reuses the app's current simulation settings instead of
  a disconnected set of hard-coded defaults
- this keeps the ground-truth calibration consistent with the actual simulated
  instrument geometry, smearing, and forward model

3. Multi-reconstruction validation

- instead of trusting only the first best-scoring quartet, the app can carry
  several candidates into a reconstruction stage
- each candidate is converted to a recovered `p` and mean size
- the code re-simulates the corresponding 2D pattern
- the candidate whose reconstructed 2D data best matches the observed 2D data
  is selected

4. Explicit unstable-run handling

- if no plausible quartet survives, the run is marked unstable
- the GUI warns the user
- the main recovered Tenor values are shown as `n/a`


Why We Do Not Use Pure Randomness By Default
--------------------------------------------

A random quartet choice can be useful only if there are several reasonable
candidates and the ranking among them is noisy.

It does not solve the main failure mode seen here:

- zero plausible candidates

In that situation, random selection mostly samples different bad quartets.
That can make the result less reproducible without making it more correct.

For this reason, the current code prefers:

- plausibility filtering
- uncertainty-aware ranking
- deterministic reconstruction agreement

over pure random choice.


What Parameters Actually Help
-----------------------------

In the current app, the following settings are the most useful when Tenor is
close to instability:

1. Photon statistics / flux

- better SNR helps the observable extraction stage the most
- this is usually the first thing to improve

2. `TENOR Recon Trials`

- increasing this from `1` to about `3-5` can help when some plausible
  quartets exist
- beyond that, gains are usually small unless the candidate pool is genuinely
  diverse

Parameters that often have smaller effect by themselves:

- `TENOR Calibration Points`
- `TENOR PSF Pair Count`

These mainly refine interpolation or broaden the search, but they do not fix a
case where the observable extraction itself is already dominated by noise.


Current Recommendation For Users
--------------------------------

If the app reports that no plausible Tenor quartet was found:

1. improve flux / counting statistics if possible
2. keep `TENOR Recon Trials` around `3-5`
3. do not rely on the Tenor numeric recovery for that run
4. compare with Tomchuk or NNLS, or repeat under better SNR


Summary
-------

The current Python Tenor implementation follows the section-4 workflow, but it
adds explicit safeguards to handle noisy detector data more honestly than a
plain best-residual quartet choice.

Those safeguards are:

- plausibility filtering
- simulation-consistent calibration
- multi-reconstruction selection
- unstable-run detection and warning
