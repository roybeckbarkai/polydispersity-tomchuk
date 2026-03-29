SAXS Simulator: Polydisperse Spheres

A Python Streamlit application for simulating and analyzing Small-Angle X-ray Scattering (SAXS) data.

Features

Simulation: Generates 2D and 1D SAXS patterns for spheres with various size distributions (Gaussian, Lognormal, Schulz, etc.).

Physics: Includes form factors, structure averaging, smearing, flux scaling, and Poisson noise.

Analysis: Performs a unified Beaucage fit for Tomchuk analysis, calculates scattering invariants, and estimates polydispersity indices (PDI, PDI2).

Recovery: Back-calculates the input size distribution parameters using the method described by Tomchuk et al.
How to Run
I
nstall Dependencies:
pip install -r requirements.txt

Run the App:
streamlit run streamlit_app.py
