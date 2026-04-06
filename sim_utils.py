# File: sim_utils.py
# Last Updated: Tuesday, February 10, 2026
# Description: Physics models, distribution functions, and core simulation engine.

import numpy as np
from scipy.special import gamma, factorial
from scipy.special import j1, sici
from scipy.ndimage import gaussian_filter
from scipy.integrate import trapezoid

# --- Math & Physics Utilities ---
def double_factorial(n):
    if n <= 0: return 1
    return n * double_factorial(n - 2)

def nCr(n, r):
    try:
        return factorial(n) / (factorial(r) * factorial(n - r))
    except:
        return 1

# --- Distributions ---
def get_distribution(dist_type, r, mean_r, p):
    mean_r = max(mean_r, 1e-6)
    p = max(p, 1e-6)
    
    if dist_type == 'Lognormal':
        s = np.sqrt(np.log(1 + p**2))
        scale = mean_r / np.exp(s**2 / 2.0)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            coef = 1.0 / (r * s * np.sqrt(2 * np.pi))
            arg = (np.log(r / scale)**2) / (2 * s**2)
            pdf = coef * np.exp(-arg)
        pdf = np.nan_to_num(pdf)
        return pdf

    elif dist_type == 'Gaussian':
        sigma = p * mean_r
        coef = 1.0 / (sigma * np.sqrt(2 * np.pi))
        arg = ((r - mean_r)**2) / (2 * sigma**2)
        return coef * np.exp(-arg)

    elif dist_type == 'Schulz':
        if p < 1e-4: return np.zeros_like(r)
        z = (1.0 / (p**2)) - 1.0
        if z <= 0: return np.zeros_like(r)
        a = z + 1
        b = a / mean_r
        try:
            norm = (b**a) / gamma(a)
        except:
            norm = 1.0
        with np.errstate(over='ignore', invalid='ignore'):
             pdf = norm * (r**z) * np.exp(-b * r)
        return np.nan_to_num(pdf)

    elif dist_type == 'Boltzmann':
        sigma = p * mean_r
        coef = 1.0 / (sigma * np.sqrt(2))
        arg = (np.sqrt(2) * np.abs(r - mean_r)) / sigma
        return coef * np.exp(-arg)

    elif dist_type == 'Triangular':
        sigma = p * mean_r
        w = sigma * np.sqrt(6)
        lower = mean_r - w
        upper = mean_r + w
        h = 1.0 / w
        pdf = np.zeros_like(r)
        mask_up = (r >= lower) & (r <= mean_r)
        mask_down = (r > mean_r) & (r <= upper)
        pdf[mask_up] = h * (r[mask_up] - lower) / w
        pdf[mask_down] = h * (upper - r[mask_down]) / w
        return pdf

    elif dist_type == 'Uniform':
        sigma = p * mean_r
        w = sigma * np.sqrt(3)
        lower = mean_r - w
        upper = mean_r + w
        pdf = np.zeros_like(r)
        mask = (r >= lower) & (r <= upper)
        pdf[mask] = 1.0 / (2 * w)
        return pdf
    
    return np.zeros_like(r)


def _distribution_pdf_on_grid(dist_type, grid, mean_size, p_val):
    grid = np.asarray(grid, dtype=float)
    pdf = np.asarray(get_distribution(dist_type, grid, mean_size, p_val), dtype=float)
    pdf[~np.isfinite(pdf)] = 0.0
    pdf = np.clip(pdf, 0.0, None)
    return pdf


def _build_discrete_distribution_from_pdf(
    size_grid,
    pdf_grid,
    ensemble_members,
    threshold=0.999,
    xmin=None,
    weight_power=0.0,
):
    size_grid = np.asarray(size_grid, dtype=float)
    pdf_grid = np.asarray(pdf_grid, dtype=float)
    if xmin is None:
        xmin = float(size_grid[0]) if len(size_grid) else 0.0
    ensemble_members = max(int(ensemble_members), 3)

    valid = np.isfinite(size_grid) & np.isfinite(pdf_grid) & (size_grid >= xmin) & (pdf_grid >= 0)
    size_grid = size_grid[valid]
    pdf_grid = pdf_grid[valid]
    if size_grid.size < 4 or np.sum(pdf_grid) <= 0:
        return size_grid, pdf_grid

    area = trapezoid(pdf_grid, size_grid)
    if area <= 0:
        return size_grid, pdf_grid
    pdf_grid = pdf_grid / area

    weight_curve = pdf_grid * np.power(np.maximum(size_grid, xmin), float(weight_power))
    weight_area = trapezoid(weight_curve, size_grid)
    if weight_area <= 0:
        weight_curve = pdf_grid.copy()
        weight_area = trapezoid(weight_curve, size_grid)
    if weight_area <= 0:
        return size_grid, pdf_grid

    cumulative_weight = np.zeros_like(size_grid)
    cumulative_weight[1:] = np.cumsum(0.5 * (weight_curve[1:] + weight_curve[:-1]) * np.diff(size_grid))
    cumulative_weight /= cumulative_weight[-1]
    cumulative_weight = np.maximum.accumulate(cumulative_weight + np.linspace(0.0, 1e-12, len(cumulative_weight)))
    cumulative_weight[-1] = 1.0

    cumulative_number = np.zeros_like(size_grid)
    cumulative_number[1:] = np.cumsum(0.5 * (pdf_grid[1:] + pdf_grid[:-1]) * np.diff(size_grid))
    cumulative_number /= cumulative_number[-1]

    first_moment_density = pdf_grid * size_grid
    cumulative_first_moment = np.zeros_like(size_grid)
    cumulative_first_moment[1:] = np.cumsum(
        0.5 * (first_moment_density[1:] + first_moment_density[:-1]) * np.diff(size_grid)
    )

    threshold = float(np.clip(threshold, 1e-6, 1.0))
    prob_edges = np.linspace(0.0, threshold, ensemble_members + 1)
    radius_edges = np.interp(prob_edges, cumulative_weight, size_grid, left=size_grid[0], right=size_grid[-1])

    members = []
    weights = []
    eps = max(np.finfo(float).eps, 1e-15)
    for left_edge, right_edge in zip(radius_edges[:-1], radius_edges[1:]):
        if not np.isfinite(left_edge) or not np.isfinite(right_edge):
            continue
        if right_edge <= left_edge:
            continue
        p_low = np.interp(left_edge, size_grid, cumulative_number, left=0.0, right=1.0)
        p_high = np.interp(right_edge, size_grid, cumulative_number, left=0.0, right=1.0)
        bin_weight = max(float(p_high - p_low), 0.0)
        if bin_weight <= eps:
            continue
        m1_low = np.interp(left_edge, size_grid, cumulative_first_moment, left=0.0, right=cumulative_first_moment[-1])
        m1_high = np.interp(right_edge, size_grid, cumulative_first_moment, left=0.0, right=cumulative_first_moment[-1])
        center_mass = float((m1_high - m1_low) / bin_weight)
        if not np.isfinite(center_mass):
            continue
        members.append(max(center_mass, xmin))
        weights.append(bin_weight)

    if len(members) < 2:
        fallback_indices = np.linspace(0, len(size_grid) - 1, ensemble_members).astype(int)
        members = size_grid[np.unique(fallback_indices)]
        weights = np.interp(members, size_grid, pdf_grid, left=0.0, right=0.0)

    members = np.asarray(members, dtype=float)
    weights = np.asarray(weights, dtype=float)
    positive = np.isfinite(members) & np.isfinite(weights) & (weights > 0)
    members = members[positive]
    weights = weights[positive]
    if members.size == 0 or np.sum(weights) <= 0:
        return size_grid, pdf_grid

    order = np.argsort(members)
    members = members[order]
    weights = weights[order]
    weights = weights / np.sum(weights)
    return members, weights

# --- Form Factors ---
def sphere_form_factor(q, r):
    q_col = q[:, np.newaxis]
    r_row = r[np.newaxis, :]
    qr = q_col * r_row
    
    with np.errstate(divide='ignore', invalid='ignore'):
        amp = 3 * (np.sin(qr) - qr * np.cos(qr)) / (qr**3)
    amp = np.nan_to_num(amp, nan=1.0) 
    
    vol = (4.0/3.0) * np.pi * (r_row**3)
    return (vol**2) * (amp**2)

def debye_form_factor(q, rg):
    q_col = q[:, np.newaxis]
    r_row = rg[np.newaxis, :]
    u = (q_col * r_row)**2
    with np.errstate(divide='ignore', invalid='ignore'):
        val = 2.0 * (np.exp(-u) + u - 1.0) / (u**2)
    val = np.nan_to_num(val, nan=1.0)
    return val


def shell_form_factor(q, rg):
    q_col = q[:, np.newaxis]
    rg_row = rg[np.newaxis, :]
    qr = q_col * rg_row
    with np.errstate(divide='ignore', invalid='ignore'):
        val = (np.sin(qr) / qr) ** 2
    return np.nan_to_num(val, nan=1.0)


def thin_rod_form_factor(q, rg):
    q_col = q[:, np.newaxis]
    rg_row = rg[np.newaxis, :]
    rod_length = np.sqrt(12.0) * rg_row
    u = np.maximum(q_col * rod_length, 1e-12)
    si_vals, _ = sici(u)
    with np.errstate(divide='ignore', invalid='ignore'):
        val = (2.0 * si_vals / u) - (4.0 * np.sin(u / 2.0) ** 2) / (u ** 2)
    return np.nan_to_num(val, nan=1.0, posinf=1.0, neginf=0.0)


def thin_disk_form_factor(q, rg):
    q_col = q[:, np.newaxis]
    rg_row = rg[np.newaxis, :]
    disk_radius = np.sqrt(2.0) * rg_row
    u = np.maximum(q_col * disk_radius, 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        val = 2.0 * (1.0 - j1(2.0 * u) / u) / (u ** 2)
    return np.nan_to_num(val, nan=1.0, posinf=1.0, neginf=0.0)


def guinier_curvature_form_factor(q, rg, phi2, phi3=0.0):
    q_col = q[:, np.newaxis]
    rg_row = rg[np.newaxis, :]
    u = (q_col * rg_row) ** 2
    exponent = -(u / 3.0) + 0.5 * phi2 * (u ** 2) + (phi3 / 6.0) * (u ** 3)
    return np.exp(exponent)


def get_form_factor_kernel(form_factor_model, phi2=0.0, phi3=0.0):
    model = (form_factor_model or "").strip()
    if model == "Exact Sphere":
        return sphere_form_factor, "radius", 0.0
    if model == "Exact Gaussian Chain":
        return debye_form_factor, "rg", 0.0
    if model == "Exact Shell":
        return shell_form_factor, "rg", 4.0
    if model == "Exact Thin Rod":
        return thin_rod_form_factor, "rg", 2.0
    if model == "Exact Thin Disk":
        return thin_disk_form_factor, "rg", 4.0
    return (lambda q, size: guinier_curvature_form_factor(q, size, phi2=phi2, phi3=phi3)), "rg", 0.0


def sample_size_distribution(size_grid, dist_type, mean_size, p_val, ensemble_sampling="Continuous", ensemble_members=11):
    if str(ensemble_sampling).lower().startswith("disc"):
        sigma = max(float(p_val) * float(mean_size), 1e-12)
        fine_min = max(0.0, float(np.min(size_grid)) if len(size_grid) else 0.0)
        fine_max = max(float(mean_size) + 20.0 * sigma, float(np.max(size_grid)) if len(size_grid) else float(mean_size))
        fine_grid = np.linspace(fine_min, fine_max, max(len(size_grid), 10000))
        if dist_type == "Lognormal":
            fine_grid[0] = max(fine_grid[0], 1e-9)
        pdf_vals = _distribution_pdf_on_grid(dist_type, fine_grid, mean_size, p_val)
        return _build_discrete_distribution_from_pdf(
            size_grid=fine_grid,
            pdf_grid=pdf_vals,
            ensemble_members=ensemble_members,
            threshold=0.999,
            xmin=fine_min,
            weight_power=0.0,
        )

    pdf_vals = _distribution_pdf_on_grid(dist_type, size_grid, mean_size, p_val)
    area = trapezoid(pdf_vals, size_grid)
    if area > 0:
        pdf_vals = pdf_vals / area
    return size_grid, pdf_vals


def get_detector_q_max(
    pixels,
    q_max=None,
    pixel_size_um=None,
    detector_side_cm=None,
    sample_detector_distance_cm=None,
    wavelength_nm=None,
):
    pixels = max(int(float(pixels)), 2)
    if (
        (pixel_size_um is not None or detector_side_cm is not None)
        and sample_detector_distance_cm is not None
        and wavelength_nm is not None
    ):
        if pixel_size_um is None:
            detector_side_cm = max(float(detector_side_cm), 1e-9)
            pixel_pitch_cm = detector_side_cm / pixels
        else:
            pixel_pitch_cm = max(float(pixel_size_um), 1e-9) * 1e-4
        sample_detector_distance_cm = max(float(sample_detector_distance_cm), 1e-9)
        wavelength_nm = max(float(wavelength_nm), 1e-9)
        return (4.0 * np.pi / wavelength_nm) * pixel_pitch_cm * (pixels - 1) / (2.0 * sample_detector_distance_cm)
    if q_max is None:
        raise ValueError("Either q_max or instrument geometry must be provided.")
    return max(float(q_max), 1e-9)


def build_detector_q_grid(
    pixels,
    q_max=None,
    pixel_size_um=None,
    detector_side_cm=None,
    sample_detector_distance_cm=None,
    wavelength_nm=None,
):
    pixels = max(int(float(pixels)), 2)
    q_extent = get_detector_q_max(
        pixels=pixels,
        q_max=q_max,
        pixel_size_um=pixel_size_um,
        detector_side_cm=detector_side_cm,
        sample_detector_distance_cm=sample_detector_distance_cm,
        wavelength_nm=wavelength_nm,
    )
    q_axis = np.linspace(-q_extent, q_extent, pixels)
    qx, qy = np.meshgrid(q_axis, q_axis)
    q_r = np.sqrt(qx**2 + qy**2)
    return q_axis, qx, qy, q_r, q_extent


def radial_average_detector_image(i_2d, q_r, q_min, q_max, n_bins, binning_mode):
    """Shared 2D->1D reduction used by the simulator and Tenor-SAXS."""
    i_2d = np.asarray(i_2d, dtype=float)
    q_r = np.asarray(q_r, dtype=float)

    if binning_mode == "Linear" or binning_mode == "Lin":
        bin_width = (q_max - q_min) / n_bins
        if bin_width <= 0:
            bin_width = q_max / n_bins
        r_indices = ((q_r - q_min) / bin_width).astype(int).ravel()
        valid_mask = (r_indices >= 0) & (r_indices < n_bins) & (q_r.ravel() <= q_max)
        tbin = np.bincount(r_indices[valid_mask], weights=i_2d.ravel()[valid_mask], minlength=n_bins)
        nr = np.bincount(r_indices[valid_mask], minlength=n_bins)
        radial_prof = np.zeros(n_bins)
        nonzero = nr > 0
        radial_prof[nonzero] = tbin[nonzero] / nr[nonzero]
        q_1d = q_min + (np.arange(n_bins) + 0.5) * bin_width
    else:
        q_min_log = max(q_min, 1e-4)
        edges = np.logspace(np.log10(q_min_log), np.log10(q_max), n_bins + 1)
        r_vals_flat = q_r.ravel()
        i_vals_flat = i_2d.ravel()
        inds = np.digitize(r_vals_flat, edges)
        valid_mask = (inds >= 1) & (inds <= n_bins)
        valid_inds = inds[valid_mask] - 1
        tbin = np.bincount(valid_inds, weights=i_vals_flat[valid_mask], minlength=n_bins)
        nr = np.bincount(valid_inds, minlength=n_bins)
        radial_prof = np.zeros(n_bins)
        nonzero = nr > 0
        radial_prof[nonzero] = tbin[nonzero] / nr[nonzero]
        q_1d = np.sqrt(edges[:-1] * edges[1:])

    valid_profile = (radial_prof > 0) & np.isfinite(radial_prof)
    return q_1d[valid_profile], radial_prof[valid_profile]

# --- Core Simulation Runner ---
def run_simulation_core(params):
    mean_rg = float(params['mean_rg'])
    p_val = float(params['p_val'])
    dist_type = params['dist_type']
    mode_key = params['mode']
    pixels = int(float(params['pixels']))
    q_max = float(params['q_max'])
    q_min = float(params['q_min'])
    pixel_size_um = float(
        params.get(
            'pixel_size_um',
            float(params.get('detector_side_cm', 7.0)) * 1.0e4 / max(pixels, 1),
        )
    )
    sample_detector_distance_cm = float(params.get('sample_detector_distance_cm', 150.0))
    wavelength_nm = float(params.get('wavelength_nm', 0.15))
    n_bins = int(float(params['n_bins']))
    smearing = float(params.get('smearing', 0.0))
    smearing_x = float(params.get('smearing_x', smearing))
    smearing_y = float(params.get('smearing_y', smearing))
    flux = float(params['flux'])
    noise = bool(params['noise'])
    binning_mode = params['binning_mode']
    r_steps = int(float(params.get('radius_samples', 400)))
    q_steps = int(float(params.get('q_samples', 200)))
    form_factor_model = params.get('form_factor_model', 'Exact Sphere' if mode_key == 'Sphere' else 'Exact Gaussian Chain')
    phi2 = float(params.get('phi2', (-1.0 / 63.0) if mode_key == 'Sphere' else (1.0 / 18.0)))
    phi3 = float(params.get('phi3', 0.0))
    ensemble_sampling = params.get('ensemble_sampling', 'Continuous')
    ensemble_members = int(float(params.get('ensemble_members', 11)))
    weight_power = params.get('weight_power')

    kernel_func, size_kind, default_weight_power = get_form_factor_kernel(form_factor_model, phi2=phi2, phi3=phi3)
    if form_factor_model == "Exact Sphere":
        weight_power = 0.0
    elif weight_power is None:
        weight_power = default_weight_power
    weight_power = float(weight_power)

    mean_size = mean_rg if size_kind == "rg" else mean_rg * np.sqrt(5.0 / 3.0)
    sigma = p_val * mean_size
    size_min = max(0.1, mean_size - 5 * sigma)
    size_max = mean_size + 15 * sigma
    size_grid = np.linspace(size_min, size_max, r_steps)
    r_vals, pdf_vals = sample_size_distribution(
        size_grid=size_grid,
        dist_type=dist_type,
        mean_size=mean_size,
        p_val=p_val,
        ensemble_sampling=ensemble_sampling,
        ensemble_members=ensemble_members,
    )

    detector_q_max = get_detector_q_max(
        pixels=pixels,
        q_max=q_max,
        pixel_size_um=pixel_size_um,
        sample_detector_distance_cm=sample_detector_distance_cm,
        wavelength_nm=wavelength_nm,
    )
    q_profile_max = min(q_max, detector_q_max)
    q_1d = np.logspace(np.log10(1e-3), np.log10(detector_q_max * 1.5), q_steps)

    if str(ensemble_sampling).lower().startswith("disc"):
        i_matrix = kernel_func(q_1d, r_vals)
        if weight_power != 0:
            i_matrix = i_matrix * (r_vals[np.newaxis, :] ** weight_power)
        i_1d_ideal = np.sum(i_matrix * pdf_vals[np.newaxis, :], axis=1)
    else:
        i_matrix = kernel_func(q_1d, r_vals)
        if weight_power != 0:
            i_matrix = i_matrix * (r_vals[np.newaxis, :] ** weight_power)
        i_1d_ideal = trapezoid(i_matrix * pdf_vals, r_vals, axis=1)

    _, _, _, qv_r, _ = build_detector_q_grid(
        pixels=pixels,
        q_max=q_max,
        pixel_size_um=pixel_size_um,
        sample_detector_distance_cm=sample_detector_distance_cm,
        wavelength_nm=wavelength_nm,
    )

    i_2d_ideal = np.interp(qv_r.ravel(), q_1d, i_1d_ideal, left=i_1d_ideal[0], right=0)
    i_2d_ideal = i_2d_ideal.reshape(pixels, pixels)

    if smearing_x > 0 or smearing_y > 0:
        i_2d_smeared = gaussian_filter(i_2d_ideal, sigma=(smearing_y, smearing_x))
    else:
        i_2d_smeared = i_2d_ideal

    # The GUI and benchmark now define "flux" as the expected number of photons
    # in the nearest-to-center detector pixel after smearing and before Poisson
    # noise is applied. For even detector sizes there is no exact q = 0 pixel, so
    # we use the conventional center index (pixels // 2, pixels // 2), which is
    # one of the four pixels closest to the beam center.
    cy = pixels // 2
    cx = pixels // 2
    center_int = float(i_2d_smeared[cy, cx])
    scale_factor = flux / center_int if center_int > 0 else 1.0
    i_2d_scaled = i_2d_smeared * scale_factor

    if noise:
        i_2d_final = np.random.poisson(i_2d_scaled).astype(float)
    else:
        i_2d_final = i_2d_scaled

    q_sim, radial_prof = radial_average_detector_image(
        i_2d=i_2d_final,
        q_r=qv_r,
        q_min=q_min,
        q_max=q_profile_max,
        n_bins=n_bins,
        binning_mode=binning_mode,
    )
    return q_sim, radial_prof, i_2d_final, r_vals, pdf_vals
