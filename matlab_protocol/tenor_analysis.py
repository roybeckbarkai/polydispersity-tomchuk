from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from scipy.io import loadmat
from scipy.linalg import qr, solve_triangular, svd
from scipy.signal import convolve2d
from scipy.spatial import Delaunay, Voronoi
from shapely.geometry import Polygon, box

from .io_utils import save_simulation_h5
from .params import EnsembleParams, InstrumentParams, SimulationParams, init_tenor_params
from .plotting import plot_tenor_violin_closest
from .simulation import scatter2d


@dataclass(slots=True)
class GTLibrary:
    V_list: np.ndarray
    RgMeas_list: np.ndarray
    Yg100_list: np.ndarray
    Yg210_list: np.ndarray
    Ym210_list: np.ndarray
    RgTrue_covered: np.ndarray
    instrument_par: InstrumentParams | None = None
    simulation_par: SimulationParams | None = None
    ensemble_par: EnsembleParams | None = None


_SCATTER_CACHE: dict[tuple, dict] = {}


def _mat_scalar(x):
    arr = np.asarray(x)
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def load_gt_library(mat_path: str | Path) -> GTLibrary:
    data = loadmat(mat_path)
    gt = data["GT_lib"]
    root = gt[0, 0]
    names = root.dtype.names
    payload = {name: root[name] for name in names}
    inst_raw = payload["instrument_par"][0, 0]
    sim_raw = payload["simulation_par"][0, 0]
    ens_raw = payload["ensemble_par"][0, 0]
    instrument = InstrumentParams(
        SD_dist=float(_mat_scalar(inst_raw["SD_dist"])),
        lambda_=float(_mat_scalar(inst_raw["lambda"])),
        det_side=float(_mat_scalar(inst_raw["det_side"])),
        DETpix=int(_mat_scalar(inst_raw["DETpix"])),
        PSF0=np.asarray(inst_raw["PSF0"], dtype=float),
    )
    simulation = SimulationParams(
        ndiv=int(_mat_scalar(sim_raw["ndiv"])),
        Pxn=np.asarray(sim_raw["Pxn"], dtype=int).ravel(),
        signum=int(_mat_scalar(sim_raw["signum"])),
        use_r3=int(_mat_scalar(sim_raw["use_r3"])),
        use_g3=int(_mat_scalar(sim_raw["use_g3"])),
    )
    dist_param_raw = ens_raw["dist_param"][0, 0]
    ensemble = EnsembleParams(
        rg=float(_mat_scalar(ens_raw["rg"])),
        V=float(_mat_scalar(ens_raw["V"])),
        nu=float(_mat_scalar(ens_raw["nu"])),
        Scatter_R_g_weight=float(_mat_scalar(ens_raw["Scatter_R_g_weight"])),
        d_nam=str(np.asarray(ens_raw["d_nam"]).reshape(-1)[0]),
        dist_param={"N": int(_mat_scalar(dist_param_raw["N"]))},
    )
    y100 = np.asarray(payload["Yg100_list"], dtype=float).ravel()
    y210 = np.asarray(payload["Yg210_list"], dtype=float).ravel()
    if y100.size == 0 and "Y100_list" in payload:
        y100 = np.asarray(payload["Y100_list"], dtype=float).ravel()
    if y210.size == 0 and "Y210_list" in payload:
        y210 = np.asarray(payload["Y210_list"], dtype=float).ravel()
    return GTLibrary(
        V_list=np.asarray(payload["V_list"], dtype=float).ravel(),
        RgMeas_list=np.asarray(payload["RgMeas_list"], dtype=float).ravel(),
        Yg100_list=y100,
        Yg210_list=y210,
        Ym210_list=np.asarray(payload["Ym210_list"], dtype=float).ravel(),
        RgTrue_covered=np.asarray(payload["RgTrue_covered"], dtype=float).ravel(),
        instrument_par=instrument,
        simulation_par=simulation,
        ensemble_par=ensemble,
    )


def best_origin_quad_b_faster_bins(x, y, nBins=200, B=100, alpha=0.05, min_pts=10, tol_pct=5, abs_tol=1e-3, stop_on_tol=True, threshold=-1.0 / 6.0):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = (x >= 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    s1 = np.cumsum(np.ones_like(x))
    sx = np.cumsum(x)
    sx2 = np.cumsum(x**2)
    sx3 = np.cumsum(x**3)
    sx4 = np.cumsum(x**4)
    sy = np.cumsum(y)
    sxy = np.cumsum(x * y)
    sx2y = np.cumsum((x**2) * y)
    sy2 = np.cumsum(y**2)

    edges = np.quantile(x, np.linspace(0, 1, nBins + 1))
    bin_idx = np.searchsorted(x, edges[1:], side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(x) - 1)
    z_val = 1.96
    best_k = 0
    stopped_early = False
    for k, idx in enumerate(bin_idx):
        idx = int(idx)
        if idx + 1 < min_pts:
            continue
        XtX = np.array(
            [
                [s1[idx], sx[idx], sx2[idx]],
                [sx[idx], sx2[idx], sx3[idx]],
                [sx2[idx], sx3[idx], sx4[idx]],
            ],
            dtype=float,
        )
        XtY = np.array([sy[idx], sxy[idx], sx2y[idx]], dtype=float)
        beta = np.linalg.solve(XtX, XtY)
        mse = (sy2[idx] - beta @ XtY) / max(idx + 1 - 3, 1)
        mse = max(mse, 0.0)
        invXtX = np.linalg.solve(XtX, np.eye(3))
        se = np.sqrt(np.diag(invXtX) * mse)
        xk = x[idx]
        xvec = np.array([1.0, xk, xk**2], dtype=float)
        y_val = xvec @ beta
        y_se = np.sqrt((xvec @ invXtX @ xvec) * mse)
        y_low = y_val - z_val * y_se
        b_est = beta[1]
        rel_err = (z_val * se[1] / max(abs(b_est), abs_tol)) * 100.0
        if stop_on_tol and (y_low < threshold) and (rel_err <= tol_pct):
            best_k = k
            stopped_early = True
            break
        best_k = k

    final_idx = int(bin_idx[best_k]) + 1
    xn = x[:final_idx]
    yn = y[:final_idx]
    X = np.column_stack([np.ones(final_idx), xn, xn**2])
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ yn)
    M = np.linalg.solve(XtX, X.T)
    resid = yn - X @ beta
    rng = np.random.default_rng(12345)
    R = rng.integers(0, final_idx, size=(final_idx, B))
    beta_b = beta[:, None] + M @ (resid[R] - np.mean(resid))
    lower_p = 100 * (alpha / 2)
    upper_p = 100 * (1 - alpha / 2)
    best_xmax = x[final_idx - 1]
    best_coef = beta
    best_coef_CI = np.vstack(
        [
            np.percentile(beta_b[0], [lower_p, upper_p]),
            np.percentile(beta_b[1], [lower_p, upper_p]),
            np.percentile(beta_b[2], [lower_p, upper_p]),
        ]
    )
    y_at_b = np.array([1.0, best_xmax, best_xmax**2]) @ beta_b
    best_y_at_xmax_CI = np.percentile(y_at_b, [lower_p, upper_p])
    info = {"stopped_early": stopped_early, "best_idx": final_idx}
    return beta[1], best_coef_CI[1], best_xmax, best_y_at_xmax_CI, best_coef, best_coef_CI, info


def fit_I_r_theta_ratios_weighted_centered(r, th, I, weight, use_r3=True, use_g3=False):
    r = np.asarray(r, dtype=float).ravel()
    th = np.asarray(th, dtype=float).ravel()
    I = np.asarray(I, dtype=float).ravel()
    weight = np.asarray(weight, dtype=float).ravel()
    c2 = np.cos(2.0 * th)
    valid = np.isfinite(I) & np.isfinite(r) & np.isfinite(th) & np.isfinite(weight)
    if not np.any(valid):
        raise ValueError("No valid samples.")
    r = r[valid]
    I = I[valid]
    c2 = c2[valid]
    w = weight[valid]

    mu_r = float(np.sum(w * r) / np.sum(w))
    rc = r - mu_r
    if use_g3:
        Gcols = [np.ones_like(r), rc, rc**2, rc**3]
    else:
        Gcols = [np.ones_like(r), rc, rc**2]
    if use_r3:
        Mcols = [r * c2, (r**2) * c2, (r**3) * c2]
    else:
        Mcols = [r * c2, (r**2) * c2]
    X = np.column_stack(Gcols + Mcols)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = I * sw
    Q, Rq = qr(Xw, mode="economic")
    rankX = np.linalg.matrix_rank(Rq)
    k = X.shape[1]
    if rankX < k:
        U, S, Vh = svd(Xw, full_matrices=False)
        tol = max(Xw.shape) * np.finfo(float).eps * S[0]
        s_inv = np.where(S > tol, 1.0 / S, 0.0)
        p_c = Vh.T @ (s_inv * (U.T @ yw))
    else:
        p_c = solve_triangular(Rq, Q.T @ yw)
    yhat = X @ p_c
    res = I - yhat
    SSE = np.sum(w * (res**2))
    dof = max(len(I) - k, 1)
    s2 = SSE / dof
    Rinv = solve_triangular(Rq, np.eye(k))
    covPc = s2 * (Rinv @ Rinv.T)

    Tfull = np.eye(7)
    Tfull[0, :4] = [1.0, -mu_r, mu_r**2, -mu_r**3]
    Tfull[1, :4] = [0.0, 1.0, -2.0 * mu_r, 3.0 * mu_r**2]
    Tfull[2, :4] = [0.0, 0.0, 1.0, -3.0 * mu_r]
    Tfull[3, :4] = [0.0, 0.0, 0.0, 1.0]
    kG = len(Gcols)
    kM = len(Mcols)
    T = np.zeros((k, k), dtype=float)
    T[:kG, :kG] = Tfull[:kG, :kG]
    T[kG:, kG:] = np.eye(kM)
    p = T @ p_c
    covP = T @ covPc @ T.T

    z95 = 1.95996398454005
    g0, g1, g2 = p[:3]
    result = {
        "p": p,
        "Rq": Rq,
        "covP": covP,
    }
    dg = np.zeros(k, dtype=float)
    dg[0] = -g1 / (g0**2)
    dg[1] = 1.0 / g0
    g_ratio = g1 / g0
    se_g = np.sqrt(max(float(dg @ covP @ dg), 0.0))
    result["g_ratio"] = g_ratio
    result["CI95_G"] = g_ratio + z95 * se_g * np.array([-1.0, 1.0])

    if abs(g0) < np.finfo(float).eps:
        g100_ratio = np.nan
        g100_CI = np.array([np.nan, np.nan], dtype=float)
    else:
        d100 = np.zeros(k, dtype=float)
        d100[0] = -2.0 * g1 / (g0**3)
        d100[1] = 1.0 / (g0**2)
        g100_ratio = g1 / (g0**2)
        se100 = np.sqrt(max(float(d100 @ covP @ d100), 0.0))
        g100_CI = g100_ratio + z95 * se100 * np.array([-1.0, 1.0])
    result["g100_ratio"] = g100_ratio
    result["g100_CI95"] = g100_CI

    if abs(g0 * g1) < np.finfo(float).eps:
        g210_ratio = np.nan
        g210_CI = np.array([np.nan, np.nan], dtype=float)
    else:
        d210 = np.zeros(k, dtype=float)
        d210[0] = -g2 / (g1 * g0**2)
        d210[1] = -g2 / (g0 * g1**2)
        d210[2] = 1.0 / (g0 * g1)
        g210_ratio = g2 / (g1 * g0)
        se210 = np.sqrt(max(float(d210 @ covP @ d210), 0.0))
        g210_CI = g210_ratio + z95 * se210 * np.array([-1.0, 1.0])
    result["g210_ratio"] = g210_ratio
    result["g210_CI95"] = g210_CI

    m0_idx = kG
    m1_idx = kG + 1
    if abs(g0 * p[m0_idx]) < np.finfo(float).eps:
        m210_ratio = np.nan
        m210_CI = np.array([np.nan, np.nan], dtype=float)
    else:
        d210m = np.zeros(k, dtype=float)
        d210m[0] = -p[m1_idx] / (p[m0_idx] * g0**2)
        d210m[m0_idx] = -p[m1_idx] / (p[m0_idx] ** 2 * g0)
        d210m[m1_idx] = 1.0 / (p[m0_idx] * g0)
        m210_ratio = p[m1_idx] / (p[m0_idx] * g0)
        se210m = np.sqrt(max(float(d210m @ covP @ d210m), 0.0))
        m210_CI = m210_ratio + z95 * se210m * np.array([-1.0, 1.0])
    result["m210_ratio"] = m210_ratio
    result["m210_CI95"] = m210_CI
    return result


def MG_extract(Pxn, q_mat_x, q_mat_y, I_mat, signum=4, RG2=None, use_r3=True, use_g3=True):
    qvr = np.hypot(q_mat_x, q_mat_y)
    if RG2 is None or not np.isfinite(float(RG2)) or float(RG2) <= 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            RG2 = -3.0 * best_origin_quad_b_faster_bins(qvr**2, np.log(I_mat))[0]
    I_mat = np.asarray(I_mat, dtype=np.float32)
    pxn = [int(v) for v in np.asarray(Pxn).ravel()[:4]]

    # MATLAB applies the broader PSF pair first to build F2, then the narrower
    # pair to build F before taking log(F2/F).
    pxx2, pxy2 = pxn[2], pxn[3]
    H2 = np.exp(-(np.linspace(-signum, signum, pxy2) ** 2) / 2.0)[:, None] * np.exp(-(np.linspace(-signum, signum, pxx2) ** 2) / 2.0)[None, :]
    H2 = (H2 / H2.sum()).astype(np.float32)
    F2 = convolve2d(I_mat, H2, mode="same", boundary="fill", fillvalue=0.0)

    pxx, pxy = pxn[0], pxn[1]
    H = np.exp(-(np.linspace(-signum, signum, pxy) ** 2) / 2.0)[:, None] * np.exp(-(np.linspace(-signum, signum, pxx) ** 2) / 2.0)[None, :]
    H = (H / H.sum()).astype(np.float32)
    F = convolve2d(I_mat, H, mode="same", boundary="fill", fillvalue=0.0)
    q_unique = np.unique(np.asarray(q_mat_y, dtype=float))
    if q_unique.size < 2:
        q_unique = np.unique(np.asarray(q_mat_x, dtype=float))
    dqpix = float(np.abs(np.mean(np.diff(np.sort(q_unique)))))

    R_g_nofilt = float(RG2)
    deadpix = 2 * max(pxn)
    maxq = float(np.max(qvr))
    qrng = np.array([0.0 * deadpix * dqpix, min(maxq - deadpix * dqpix, 0.79 / np.sqrt(R_g_nofilt))], dtype=float)
    if np.diff(qrng)[0] < 0:
        raise ValueError("not enough pixels- consider using a smaller slit")

    qvt = np.arctan2(q_mat_y, q_mat_x)
    with np.errstate(divide="ignore", invalid="ignore"):
        G2f = np.log(F2.ravel() / F.ravel())
    rng_mask = (qvr.ravel() < qrng[1]) & (qvr.ravel() > qrng[0])
    res = fit_I_r_theta_ratios_weighted_centered(
        qvr.ravel()[rng_mask] ** 2,
        qvt.ravel()[rng_mask],
        G2f[rng_mask],
        np.sqrt(I_mat.ravel()[rng_mask]),
        bool(use_r3),
        bool(use_g3),
    )
    g_coeff = res["p"][:3]
    g_rat = np.array(
        [
            g_coeff[1] / g_coeff[0],
            res["CI95_G"][0],
            res["g100_ratio"],
            res["m210_ratio"],
            res["m210_CI95"][0],
            res["g210_ratio"],
            g_coeff[2] / g_coeff[1],
            res["g100_CI95"][0],
            np.nan,
            res["g210_CI95"][0],
        ],
        dtype=float,
    )
    return g_rat, float(R_g_nofilt), np.asarray([pxn[0], pxn[1], pxn[2], pxn[3]], dtype=int), res


def scat_interp_unscaled(x, y, v, xq, yq, method="linear"):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    dx = xmax - xmin if xmax != xmin else 1.0
    dy = ymax - ymin if ymax != ymin else 1.0
    points = np.column_stack([(x - xmin) / dx, (y - ymin) / dy])
    xi = np.column_stack([(np.asarray(xq, dtype=float).ravel() - xmin) / dx, (np.asarray(yq, dtype=float).ravel() - ymin) / dy])
    if method == "nearest":
        interp = NearestNDInterpolator(points, v)
        out = interp(xi)
    else:
        interp = LinearNDInterpolator(points, v, fill_value=np.nan)
        out = interp(xi)
    return np.asarray(out, dtype=float).reshape(np.shape(xq))


def scattered_linear_extrapolation_unscaled(x, y, v, xq, yq):
    """MATLAB-like scaled-axis scattered linear interpolation with extrapolation.

    MATLAB's `scatteredInterpolant(..., 'natural', 'linear')` can extrapolate
    outside the convex hull. SciPy's standard scattered interpolators return
    NaN there, so for fallback reconstruction we extend the affine plane from
    the closest Delaunay triangle in normalized coordinates.
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    xq_arr = np.asarray(xq, dtype=float)
    yq_arr = np.asarray(yq, dtype=float)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    dx = xmax - xmin if xmax != xmin else 1.0
    dy = ymax - ymin if ymax != ymin else 1.0
    points = np.column_stack([(x - xmin) / dx, (y - ymin) / dy])
    queries = np.column_stack([(xq_arr.ravel() - xmin) / dx, (yq_arr.ravel() - ymin) / dy])

    tri = Delaunay(points)

    def point_seg_dist(p, a, b):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 0:
            return float(np.linalg.norm(p - a))
        t = np.clip(float(np.dot(p - a, ab) / denom), 0.0, 1.0)
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    out = np.empty(len(queries), dtype=float)
    x_max_norm = 1.0
    simplex_ids = tri.find_simplex(queries)
    for qi, q in enumerate(queries):
        si = int(simplex_ids[qi])
        if si >= 0:
            natural_val = natural_neighbor_interpolate_unscaled(
                x,
                y,
                v,
                np.array([xq_arr.ravel()[qi]], dtype=float),
                np.array([yq_arr.ravel()[qi]], dtype=float),
            ).reshape(-1)[0]
            out[qi] = natural_val
            continue

        best = None
        for si2, simplex in enumerate(tri.simplices):
            verts = points[simplex]
            T = tri.transform[si2]
            delta = q - T[2]
            bary = np.dot(T[:2], delta)
            bary = np.r_[bary, 1.0 - bary.sum()]
            dists = [point_seg_dist(q, verts[i], verts[(i + 1) % 3]) for i in range(3)]
            dist = min(dists)
            val = float(np.dot(bary, v[simplex]))
            candidate = (dist, val)
            if best is None or candidate[0] < best[0]:
                best = (dist, val)
        if best is None:
            out[qi] = np.nan
            continue

        # Empirical MATLAB-compatibility guard:
        # large excursions on the high-Y edge of the landscape were left as NaN
        # in the saved MATLAB table rather than being aggressively extrapolated.
        if q[0] > x_max_norm and best[0] > 0.1:
            out[qi] = np.nan
        else:
            out[qi] = best[1]

    return out.reshape(np.shape(xq_arr))


def _voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = float(np.ptp(vor.points, axis=0).max()) * 2.0

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)
    return new_regions, np.asarray(new_vertices)


def natural_neighbor_interpolate_unscaled(x, y, v, xq, yq):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    xq_arr = np.asarray(xq, dtype=float)
    yq_arr = np.asarray(yq, dtype=float)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    dx = xmax - xmin if xmax != xmin else 1.0
    dy = ymax - ymin if ymax != ymin else 1.0
    points = np.column_stack([(x - xmin) / dx, (y - ymin) / dy])
    queries = np.column_stack([(xq_arr.ravel() - xmin) / dx, (yq_arr.ravel() - ymin) / dy])

    cache_key = (
        len(x),
        float(xmin),
        float(xmax),
        float(ymin),
        float(ymax),
        float(np.sum(x)),
        float(np.sum(y)),
    )
    cached = _SCATTER_CACHE.get(cache_key)
    if cached is None:
        tri = Delaunay(points)
        bbox_poly = box(-2.0, -2.0, 3.0, 3.0)
        vor = Voronoi(points)
        regions, vertices = _voronoi_finite_polygons_2d(vor, radius=10.0)
        original_cells = []
        for region in regions:
            poly = Polygon(vertices[region]).intersection(bbox_poly)
            original_cells.append(poly if not poly.is_empty else None)
        cached = {
            "tri": tri,
            "bbox_poly": bbox_poly,
            "original_cells": original_cells,
            "points": points,
        }
        _SCATTER_CACHE[cache_key] = cached
    else:
        tri = cached["tri"]
        bbox_poly = cached["bbox_poly"]
        original_cells = cached["original_cells"]
        points = cached["points"]

    out = np.empty(len(queries), dtype=float)
    simplex_ids = tri.find_simplex(queries)
    for qi, q in enumerate(queries):
        if int(simplex_ids[qi]) < 0:
            out[qi] = np.nan
            continue
        aug_points = np.vstack([points, q])
        vor_q = Voronoi(aug_points)
        regions_q, vertices_q = _voronoi_finite_polygons_2d(vor_q, radius=10.0)
        query_poly = Polygon(vertices_q[regions_q[len(aug_points) - 1]]).intersection(bbox_poly)
        if query_poly.is_empty or query_poly.area <= 0:
            out[qi] = np.nan
            continue
        weights = np.zeros(len(points), dtype=float)
        for pi, cell in enumerate(original_cells):
            if cell is None or cell.is_empty:
                continue
            inter = cell.intersection(query_poly)
            if not inter.is_empty:
                weights[pi] = inter.area
        total = weights.sum()
        out[qi] = np.nan if total <= 0 else float(np.dot(weights / total, v))
    return out.reshape(np.shape(xq_arr))


def update_GT_landscape(GT: GTLibrary, rg_target: float, rng: np.random.Generator | None = None) -> GTLibrary:
    if rng is None:
        rng = np.random.default_rng(12345)
    inst = GT.instrument_par
    sim = GT.simulation_par
    ens = GT.ensemble_par
    V_power = 0.5
    V_vec = np.linspace(0.005**V_power, 0.35**V_power, 15) ** (1.0 / V_power)
    jitter_amplitude = np.mean(np.diff(V_vec)) * 0.2
    V_vec = np.maximum(1e-5, V_vec + (rng.random(V_vec.shape) - 0.5) * jitter_amplitude)

    new_V = []
    new_rgm = []
    new_y100 = []
    new_y210 = []
    new_ym210 = []
    for val in V_vec:
        qx, qy, I_sim, _ = scatter2d(
            rg_target,
            0.0,
            float(val),
            ens.nu,
            inst.DETpix,
            inst.SD_dist,
            inst.lambda_,
            inst.det_side,
            inst.PSF0,
            ens.d_nam,
            ens.dist_param,
            ens.Scatter_R_g_weight,
            rng=rng,
        )
        qvr = np.hypot(qx, qy)
        rg_m = np.sqrt(-3.0 * best_origin_quad_b_faster_bins(qvr**2, np.log(I_sim))[0])
        c_rg = (rg_target**2) / rg_m
        qx, qy, I_sim, _ = scatter2d(
            c_rg,
            0.0,
            float(val),
            ens.nu,
            inst.DETpix,
            inst.SD_dist,
            inst.lambda_,
            inst.det_side,
            inst.PSF0,
            ens.d_nam,
            ens.dist_param,
            ens.Scatter_R_g_weight,
            rng=rng,
        )
        rg_m = np.sqrt(-3.0 * best_origin_quad_b_faster_bins(qvr**2, np.log(I_sim))[0])
        _, _, _, res = MG_extract(sim.Pxn, qx, qy, I_sim, sim.signum, None, sim.use_r3, sim.use_g3)
        p = res["p"]
        new_V.append(val)
        new_rgm.append(rg_m)
        new_y100.append(p[1] / (p[0] ** 2))
        new_y210.append(p[2] / (p[1] * p[0]))
        new_ym210.append(p[sim.use_g3 + 4] / (p[sim.use_g3 + 3] * p[0]))
    GT.V_list = np.r_[GT.V_list, np.asarray(new_V, dtype=float)]
    GT.RgMeas_list = np.r_[GT.RgMeas_list, np.asarray(new_rgm, dtype=float)]
    GT.Yg100_list = np.r_[GT.Yg100_list, np.asarray(new_y100, dtype=float)]
    GT.Yg210_list = np.r_[GT.Yg210_list, np.asarray(new_y210, dtype=float)]
    GT.Ym210_list = np.r_[GT.Ym210_list, np.asarray(new_ym210, dtype=float)]
    GT.RgTrue_covered = np.r_[GT.RgTrue_covered, np.asarray([rg_target], dtype=float)]
    return GT


def tenor_process_landscape(I_mat0, qx, qy, GT: GTLibrary, inst, sim, ens, rng=None):
    qvr = np.hypot(qx, qy)
    with np.errstate(divide="ignore", invalid="ignore"):
        rg_in = np.sqrt(-3.0 * best_origin_quad_b_faster_bins(qvr**2, np.log(I_mat0))[0])
    _, _, _, res_in = MG_extract(sim.Pxn, qx, qy, I_mat0, sim.signum, None, sim.use_r3, sim.use_g3)
    p = res_in["p"]
    Yg100 = p[1] / (p[0] ** 2)
    Yg210 = p[2] / (p[1] * p[0])
    Ym210 = p[sim.use_g3 + 4] / (p[sim.use_g3 + 3] * p[0])

    needs_update = False
    if GT.RgTrue_covered.size == 0:
        GT.instrument_par = inst
        GT.simulation_par = sim
        GT.ensemble_par = ens
        GT = update_GT_landscape(GT, rg_in * 0.9, rng=rng)
        GT = update_GT_landscape(GT, rg_in * 1.1, rng=rng)
        needs_update = True
    else:
        rg_min_lib = float(np.min(GT.RgMeas_list))
        rg_max_lib = float(np.max(GT.RgMeas_list))
        if rg_in <= rg_min_lib:
            GT = update_GT_landscape(GT, rg_in * 0.9, rng=rng)
            needs_update = True
        elif rg_in >= rg_max_lib:
            GT = update_GT_landscape(GT, rg_in * 1.1, rng=rng)
            needs_update = True
        is_in_gap = np.min(np.abs(GT.RgTrue_covered - rg_in) / rg_in) > 0.05
        if is_in_gap:
            needs_update = True
    if needs_update:
        GT = update_GT_landscape(GT, rg_in, rng=rng)

    V_fine = np.linspace(np.min(GT.V_list), np.max(GT.V_list), 300)
    rg_query = np.full_like(V_fine, rg_in)
    Yg100_at_rg = scat_interp_unscaled(GT.V_list, GT.RgMeas_list, GT.Yg100_list, V_fine, rg_query, "linear")
    Yg210_at_rg = scat_interp_unscaled(GT.V_list, GT.RgMeas_list, GT.Yg210_list, V_fine, rg_query, "linear")
    diff_sign = np.diff(np.sign(Yg100_at_rg - Yg100))
    idx = np.where(diff_sign != 0)[0]
    v_sols = []
    for ii in idx:
        num = Yg100 - Yg100_at_rg[ii]
        den = Yg100_at_rg[ii + 1] - Yg100_at_rg[ii]
        v_sols.append(V_fine[ii] + num * (V_fine[ii + 1] - V_fine[ii]) / den)
    v_sols = np.asarray(v_sols, dtype=float)

    if v_sols.size == 0:
        V_est = float(
            scattered_linear_extrapolation_unscaled(
                GT.Yg100_list,
                GT.RgMeas_list,
                GT.V_list,
                np.array([Yg100], dtype=float),
                np.array([rg_in], dtype=float),
            ).reshape(-1)[0]
        )
        Winner = "Nearest_Fallback"
    elif v_sols.size == 1:
        V_est = float(v_sols[0])
        Winner = "Unique_Solution"
    else:
        y210_at_sols = interp1d(V_fine, Yg210_at_rg, kind="linear", bounds_error=False, fill_value=np.nan)(v_sols)
        if np.all(~np.isfinite(y210_at_sols)):
            V_est = float(
                scattered_linear_extrapolation_unscaled(
                    GT.Yg100_list,
                    GT.RgMeas_list,
                    GT.V_list,
                    np.array([Yg100], dtype=float),
                    np.array([rg_in], dtype=float),
                ).reshape(-1)[0]
            )
            Winner = "Nearest_Fallback"
        else:
            best_idx = int(np.nanargmin(np.abs(y210_at_sols - Yg210)))
            V_est = float(v_sols[best_idx])
            Winner = "Consensus_Selection"
    if V_est < -0.05:
        V_est = np.nan
    alternatives = np.setdiff1d(v_sols, np.array([V_est])) if np.isfinite(V_est) else v_sols
    alternatives = alternatives[(alternatives > -0.05) & np.isfinite(alternatives)]
    sols = {"Primary_V": V_est, "Alternatives": alternatives}
    return V_est, sols, Winner, GT, rg_in, Yg100, Yg210, Ym210


def tenor_protocol_4_26(
    PhotLevel=0.0,
    V=0.1,
    GT_lib: GTLibrary | None = None,
    save_dir: str | None = None,
    meas_source="simulate",
    instrument: InstrumentParams | None = None,
    simulation: SimulationParams | None = None,
    ensemble: EnsembleParams | None = None,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()
    if any(v is None for v in (instrument, simulation, ensemble)):
        instrument, simulation, ensemble, _ = init_tenor_params()
    if meas_source == "simulate":
        qx, qy, I_noisy, _ = scatter2d(
            ensemble.rg,
            PhotLevel,
            V,
            ensemble.nu,
            instrument.DETpix,
            instrument.SD_dist,
            instrument.lambda_,
            instrument.det_side,
            instrument.PSF0,
            ensemble.d_nam,
            ensemble.dist_param,
            ensemble.Scatter_R_g_weight,
            rng=rng,
        )
        dq = (
            4.0
            * np.pi
            / instrument.lambda_
            * instrument.det_side
            / instrument.SD_dist
            / (2 * round(instrument.DETpix / 2) + 1)
        )
        if save_dir:
            save_simulation_h5(save_dir, qx, qy, I_noisy, V, instrument, ensemble)
    else:
        raise NotImplementedError("Only meas_source='simulate' is currently ported.")

    if GT_lib is None:
        GT_lib = GTLibrary(
            V_list=np.array([], dtype=float),
            RgMeas_list=np.array([], dtype=float),
            Yg100_list=np.array([], dtype=float),
            Yg210_list=np.array([], dtype=float),
            Ym210_list=np.array([], dtype=float),
            RgTrue_covered=np.array([], dtype=float),
            instrument_par=instrument,
            simulation_par=simulation,
            ensemble_par=ensemble,
        )
    V_rec, sols, Winner, GT_lib, rg_in, Yg100, Yg210, Ym210 = tenor_process_landscape(
        I_noisy, qx, qy, GT_lib, instrument, simulation, ensemble, rng=rng
    )
    results_entry = {
        "Noise": PhotLevel,
        "True_V": V,
        "Primary_V": V_rec,
        "Winner": Winner,
        "Rg_meas": rg_in,
        "Yg100": Yg100,
        "Yg210": Yg210,
        "Ym210": Ym210,
        "V_all": np.r_[np.array([V_rec]), sols["Alternatives"]],
    }
    altern = results_entry["V_all"]
    altern = altern[altern > -0.05]
    if altern.size > 0:
        best = float(altern[np.argmin(np.abs(altern - V))])
    else:
        best = np.nan
    return {
        "qx": qx,
        "qy": qy,
        "I_noisy": I_noisy,
        "dq": dq,
        "GT_lib": GT_lib,
        "Results_entry": results_entry,
        "best": best,
        "summary": f"Phot dens= {-PhotLevel / dq**2:0.2e} ph*nm^2 | V= {V:.3f} | nearest V= {best:.3f} | V discrep= {best - V:.4f}",
    }


def batch_run_protocol(
    output_dir: str,
    gt_library_path: str | Path | None = None,
    plot_output_path: str | Path | None = None,
    max_cases: int | None = None,
    random_seed: int = 12345,
):
    instrument, simulation, ensemble, _ = init_tenor_params()
    if gt_library_path is not None:
        GT_lib = load_gt_library(gt_library_path)
    else:
        GT_lib = GTLibrary(
            V_list=np.array([], dtype=float),
            RgMeas_list=np.array([], dtype=float),
            Yg100_list=np.array([], dtype=float),
            Yg210_list=np.array([], dtype=float),
            Ym210_list=np.array([], dtype=float),
            RgTrue_covered=np.array([], dtype=float),
            instrument_par=instrument,
            simulation_par=simulation,
            ensemble_par=ensemble,
        )
    NoiseLevels = -1.0 / 1.6749 * 10.0 ** np.linspace(3, 5, 5)
    V_power = 0.5
    Vlist = np.linspace(0.004**V_power, 0.3**V_power, 100) ** (1.0 / V_power)

    rows = []
    summaries = []
    ind = 0
    for n in NoiseLevels:
        for v in Vlist:
            ind += 1
            if max_cases is not None and ind > max_cases:
                break
            rng = np.random.default_rng(random_seed + ind)
            result = tenor_protocol_4_26(
                PhotLevel=float(n),
                V=float(v),
                GT_lib=GT_lib,
                save_dir=output_dir,
                instrument=instrument,
                simulation=simulation,
                ensemble=ensemble,
                rng=rng,
            )
            GT_lib = result["GT_lib"]
            rows.append(result["Results_entry"])
            summaries.append(result["summary"])
        if max_cases is not None and ind >= max_cases:
            break
    table = pd.DataFrame(rows)
    fig = None
    if not table.empty:
        fig, _ = plot_tenor_violin_closest(table, instrument, output_path=plot_output_path)
    return {
        "instrument": instrument,
        "simulation": simulation,
        "ensemble": ensemble,
        "GT_lib": GT_lib,
        "Results_Table": table,
        "summaries": summaries,
        "figure": fig,
    }
