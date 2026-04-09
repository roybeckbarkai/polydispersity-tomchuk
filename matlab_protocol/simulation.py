from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve2d
from scipy.special import exp1, j1


@dataclass(slots=True)
class DistributionResult:
    r: np.ndarray
    p: np.ndarray


def make_kernel_odd_centered(kernel_in: np.ndarray) -> np.ndarray:
    kernel_out = np.asarray(kernel_in, dtype=float)
    rows, cols = kernel_out.shape
    if rows % 2 == 0:
        tmp = np.zeros((rows + 1, cols), dtype=float)
        for i in range(rows):
            tmp[i, :] += 0.5 * kernel_out[i, :]
            tmp[i + 1, :] += 0.5 * kernel_out[i, :]
        kernel_out = tmp
    rows, cols = kernel_out.shape
    if cols % 2 == 0:
        tmp = np.zeros((rows, cols + 1), dtype=float)
        for j in range(cols):
            tmp[:, j] += 0.5 * kernel_out[:, j]
            tmp[:, j + 1] += 0.5 * kernel_out[:, j]
        kernel_out = tmp
    return kernel_out


def _distribution_pdf(dist_type: str, grid: np.ndarray, vrel: float) -> np.ndarray:
    mean = 1.0
    sigma = np.sqrt(vrel) * mean
    g = np.asarray(grid, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if dist_type == "normal":
            pdf = np.exp(-((g - mean) ** 2) / (2.0 * sigma**2))
        elif dist_type == "lognormal":
            s = np.sqrt(np.log(1.0 + vrel))
            m = np.log(mean) - 0.5 * s**2
            pdf = (1.0 / g) * np.exp(-((np.log(g) - m) ** 2) / (2.0 * s**2))
        elif dist_type == "schulz":
            z = 1.0 / vrel - 1.0
            pdf = g**z * np.exp(-(z + 1.0) * g / mean)
        elif dist_type == "boltzmann":
            pdf = np.exp(-np.sqrt(2.0) * np.abs(g - mean) / sigma)
        elif dist_type == "triangular":
            L = sigma * np.sqrt(6.0)
            pdf = np.maximum(0.0, 1.0 - np.abs(g - mean) / L)
        elif dist_type == "uniform":
            L = sigma * np.sqrt(3.0)
            pdf = ((g >= mean - L) & (g <= mean + L)).astype(float)
        elif dist_type == "exponential":
            pdf = np.exp(-g / mean)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    pdf[~np.isfinite(pdf)] = 0.0
    pdf = np.clip(pdf, 0.0, None)
    return pdf


def size_distribution_discrete_no_max(
    N: int,
    Vrel: float,
    dist_type: str = "normal",
    threshold: float = 0.999,
    xmin: float = 0.0,
    weight_power: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    mean = 1.0
    sigma = np.sqrt(Vrel) * mean
    temp_xmax = mean + 20.0 * sigma
    u_fine = np.linspace(xmin, temp_xmax, 10000, dtype=float)
    w_fine = _distribution_pdf(dist_type.lower(), u_fine, Vrel)
    total = w_fine.sum()
    if total <= 0:
        raise ValueError("Distribution PDF integrates to zero.")
    w_fine = w_fine / total
    w_weighted = w_fine * (u_fine**weight_power)
    C_weight = np.cumsum(w_weighted)
    C_weight = C_weight / C_weight[-1]
    C_number = np.cumsum(w_fine)
    C_number = C_number / C_number[-1]
    C_weight = C_weight + np.linspace(0.0, 10e-12, len(C_weight))
    edges_prob = np.linspace(0.0, threshold, N + 1)
    r_edges = np.interp(edges_prob, C_weight, u_fine)

    x = np.zeros(N, dtype=float)
    p = np.zeros(N, dtype=float)
    C_moment1 = np.cumsum(w_fine * u_fine)
    for i in range(N):
        p_high = np.interp(r_edges[i + 1], u_fine, C_number)
        p_low = np.interp(r_edges[i], u_fine, C_number)
        p[i] = p_high - p_low
        m1_high = np.interp(r_edges[i + 1], u_fine, C_moment1)
        m1_low = np.interp(r_edges[i], u_fine, C_moment1)
        x[i] = (m1_high - m1_low) / max(p[i], np.finfo(float).eps)

    curr_mu = np.sum(p * x)
    curr_var = np.sum(p * (x - curr_mu) ** 2)
    scale_factor = np.sqrt(Vrel / max(curr_var, np.finfo(float).eps))
    x = scale_factor * (x - curr_mu) + mean
    x = np.maximum(x, xmin)
    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    return x, p


def _rod_sine_integral(u: np.ndarray) -> np.ndarray:
    return np.imag(exp1(-1j * u)) + np.pi / 2.0


def scatter2d(
    rg: float,
    nois: float,
    V: float,
    Nu: float,
    DETpix: int,
    SD_dist: float,
    lambda_: float,
    det_side: float,
    PSF0: np.ndarray | float,
    dist_type: str,
    dist_param: dict | None,
    weight_power: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DistributionResult]:
    if rng is None:
        rng = np.random.default_rng()
    psf = np.asarray(PSF0 if np.ndim(PSF0) else [[float(PSF0)]], dtype=float)
    psf = psf / psf.sum()
    psf = make_kernel_odd_centered(psf)
    rg = abs(float(rg))
    V = abs(float(V))
    Nu = float(Nu)
    noisemodel = np.sign(nois) if nois != 0 else 1
    det_hside = det_side / 2.0
    maxq = 4.0 * np.pi / lambda_ * det_hside / SD_dist
    npix = round(DETpix / 2)
    qv = np.linspace(-maxq, maxq, 2 * npix + 1, dtype=float)
    qvx, qvy = np.meshgrid(qv, qv, indexing="xy")
    qvx = qvx.T
    qvy = qvy.T
    qvr2 = qvx**2 + qvy**2
    qvr4 = qvr2**2
    qvr = np.sqrt(qvr2)

    n_radii = int((dist_param or {}).get("N", 11))
    r_vect, p = size_distribution_discrete_no_max(n_radii, V, dist_type, 0.995, 0.0, 0.0)
    distrp = DistributionResult(r=r_vect * rg, p=p)

    F = np.zeros_like(qvr2, dtype=float)
    for ii in range(n_radii):
        s = rg * r_vect[ii]
        s2 = s * s
        s4 = s2 * s2
        exponent = -(s2 / 3.0) * qvr2 + (0.5 * Nu * s4) * qvr4

        if abs((1.0 / Nu - round(1.0 / Nu)) - 0.000666) < 1e-6:
            ordersix = 1e8 * (abs(1.0 / Nu - round(1.0 / Nu)) - 0.000666)
            exponent = exponent + (ordersix / 6.0 * s4 * s2) * qvr4 * qvr2
        if abs(Nu - 0.000666) < 1e-6:
            ordersix = 1e8 * abs(Nu - 0.000666)
            exponent = exponent + (ordersix / 6.0 * s4 * s2) * qvr4 * qvr2

        invalid = np.abs(-(s2 / 3.0) * qvr2) < np.abs((0.5 * Nu * s4) * qvr4)
        exponent[invalid] = np.nan

        if Nu == -1.0 / 63.0:
            GF = np.sqrt(5.0 / 3.0)
            u = qvr * s * GF
            numer = 3.0 * (np.sin(u) - GF * s * qvr * np.cos(u))
            denom = qvr * qvr2 * s * s2 * GF**3
            with np.errstate(divide="ignore", invalid="ignore"):
                expo = np.log((numer / denom) ** 2)
            expo[np.abs(u) <= np.finfo(float).eps] = 0.0
            exponent = np.where(np.isfinite(np.real(expo)) & (np.imag(expo) == 0), np.real(expo), np.nan)
            weight_power = 6.0
        elif Nu == 1.0 / 18.0:
            u = qvr * s
            with np.errstate(divide="ignore", invalid="ignore"):
                expo = np.log(2.0 * (np.exp(-(u**2)) + u**2 - 1.0) / (u**4))
            expo[np.abs(u) <= np.finfo(float).eps] = 0.0
            exponent = np.where(np.isfinite(np.real(expo)) & (np.imag(expo) == 0), np.real(expo), np.nan)
            weight_power = 0.0
        elif Nu == -1.0 / 45.0:
            u = qvr * s
            with np.errstate(divide="ignore", invalid="ignore"):
                expo = 2.0 * np.log(np.sin(u) / u)
            expo[np.abs(u) <= np.finfo(float).eps] = 0.0
            exponent = np.where(np.isfinite(np.real(expo)) & (np.imag(expo) == 0), np.real(expo), np.nan)
            weight_power = 4.0
        elif Nu == 11.0 / 225.0:
            GF = np.sqrt(12.0)
            u = qvr * s * GF + 10.0 * np.finfo(float).eps
            Si = _rod_sine_integral(u)
            with np.errstate(divide="ignore", invalid="ignore"):
                expo = np.log(2.0 * Si / u - 4.0 * np.sin(u / 2.0) ** 2 / (u**2))
            expo[np.abs(u) <= np.finfo(float).eps] = 0.0
            exponent = np.where(np.isfinite(np.real(expo)) & (np.imag(expo) == 0), np.real(expo), np.nan)
            weight_power = 2.0
        elif Nu == 11.0 / 225.0 + 0.000666:
            nu = 11.0 / 225.0
            exponent = -(s2 / 3.0) * qvr2 + (0.5 * nu * s4) * qvr4
            ordersix = 412.0 / 33075.0
            exponent = exponent + (ordersix / 6.0 * s4 * s2) * qvr4 * qvr2
            invalid = np.abs(-(s2 / 3.0) * qvr2) < np.abs((0.5 * nu * s4) * qvr4)
            exponent[invalid] = np.nan
        elif Nu == 0.000666:
            GF = np.sqrt(2.0)
            u = qvr * s * GF
            with np.errstate(divide="ignore", invalid="ignore"):
                expo = np.log(2.0 / (u**2) * (1.0 - j1(2.0 * u) / u))
            expo[np.abs(u) <= np.finfo(float).eps] = 0.0
            exponent = np.where(np.isfinite(np.real(expo)) & (np.imag(expo) == 0), np.real(expo), np.nan)
            weight_power = 4.0

        F = F + (r_vect[ii] ** weight_power) * p[ii] * np.exp(exponent)

    F = convolve2d(F, psf, mode="same", boundary="fill", fillvalue=0.0)
    if noisemodel < 0:
        F = F * abs(nois)
        F = F + np.sqrt(np.clip(F, 0.0, None)) * rng.standard_normal(size=F.shape)
        F = F / abs(nois)
    else:
        F = F + nois * rng.standard_normal(size=F.shape)
    F[F < 0] = 0.0
    return qvx, qvy, F, distrp
