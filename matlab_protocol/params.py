from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class InstrumentParams:
    SD_dist: float
    lambda_: float
    det_side: float
    DETpix: int
    PSF0: np.ndarray


@dataclass(slots=True)
class SimulationParams:
    ndiv: int
    Pxn: np.ndarray
    signum: int
    use_r3: int
    use_g3: int


@dataclass(slots=True)
class EnsembleParams:
    rg: float
    V: float
    nu: float
    Scatter_R_g_weight: float
    d_nam: str
    dist_param: dict


def tri1d(n: int, mode: str = "raised") -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=float)
    if n == 1:
        return np.ones((1,), dtype=float)
    idx = np.arange(n, dtype=float)
    c = (n - 1) / 2.0
    if mode.lower() == "zero":
        if n == 2:
            w = np.array([1.0, 1.0], dtype=float)
        else:
            w = 1.0 - np.abs(idx - c) / c
    else:
        d = (n + 1) / 2.0
        w = 1.0 - np.abs(idx - c) / d
        w = np.clip(w, 0.0, None)
    return w


def bartlett2d(n: int, m: int | None = None, mode: str = "raised") -> np.ndarray:
    if m is None:
        m = n
    wy = tri1d(n, mode)[:, None]
    wx = tri1d(m, mode)[None, :]
    kernel = wy * wx
    total = kernel.sum()
    if total > 0:
        kernel = kernel / total
    return kernel


def init_tenor_params() -> tuple[InstrumentParams, SimulationParams, EnsembleParams, list[str]]:
    dnames = [
        "normal",
        "lognormal",
        "schulz",
        "exponential",
        "boltzmann",
        "triangular",
        "uniform",
    ]

    ndiv = 2
    instrument = InstrumentParams(
        SD_dist=360.0,
        lambda_=0.1,
        det_side=7.0 / ndiv,
        DETpix=round(1000 / ndiv),
        PSF0=bartlett2d(3, 15),
    )
    simulation = SimulationParams(
        ndiv=ndiv,
        Pxn=np.array([87, 85, 125, 123], dtype=int),
        signum=4,
        use_r3=0,
        use_g3=0,
    )
    ensemble = EnsembleParams(
        rg=5.0,
        V=0.1,
        nu=-1.0 / 63.0,
        Scatter_R_g_weight=6.0,
        d_nam=dnames[1],
        dist_param={"N": 41},
    )
    return instrument, simulation, ensemble, dnames
