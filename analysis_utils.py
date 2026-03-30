# File: analysis_utils.py
# Last Updated: Tuesday, February 10, 2026
# Description: Utilities for parsing data, performing Tomchuk analysis, and NNLS distribution recovery.

import math
import numpy as np
from scipy.optimize import bisect, least_squares, nnls
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter
from scipy.special import erf, gammaln
import pandas as pd
from sim_utils import nCr, double_factorial, sphere_form_factor, debye_form_factor, get_distribution, run_simulation_core
from app_settings import build_tenor_p_grid

# Tomchuk's analytic Q and lc expressions [equations (8a) and (8b)]
# contain two numerical integrals over the error-function crossover term.
TOMCHUK_C1 = 0.771965
TOMCHUK_C2 = 0.319739

def parse_saxs_file(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        data = []
        for line in content.splitlines():
            line = line.split('#')[0].split('!')[0].strip()
            if not line: continue
            line = line.replace(',', ' ').replace(';', ' ')
            parts = line.split()
            try:
                nums = [float(p) for p in parts]
                if len(nums) >= 2:
                    data.append([nums[0], nums[1]])
            except ValueError:
                continue

        arr = np.array(data)
        if arr.shape[0] < 5:
            return None, None, "File content too short."
        
        mask = np.isfinite(arr).all(axis=1)
        arr = arr[mask]
        
        if arr.shape[0] < 5:
            return None, None, "File contains mostly invalid (NaN/Inf) data."

        arr = arr[arr[:, 0].argsort()]
        # Filter negative q
        arr = arr[arr[:, 0] > 1e-6]
        
        return arr[:, 0], arr[:, 1], None
        
    except Exception as e:
        return None, None, f"Error parsing file: {str(e)}"

# --- Analysis / Recovery Logic ---

def get_normalized_moment(k, p, dist_type):
    p = max(p, 1e-5)
    
    if dist_type == 'Lognormal':
        return (1 + p**2)**(k*(k-1)/2.0)
    elif dist_type == 'Gaussian':
        total = 0.0
        limit = k // 2
        for j in range(limit + 1): total += nCr(k, 2*j) * double_factorial(2*j - 1) * (p**(2*j))
        return total
    elif dist_type == 'Schulz':
        z = (1.0 / p**2) - 1.0
        if z <= -1.0:
            return 1.0
        try:
            log_val = gammaln(z + k + 1.0) - gammaln(z + 1.0) - (k * np.log(z + 1.0))
            return float(np.exp(log_val))
        except Exception:
            return 1.0
    elif dist_type == 'Boltzmann':
        total = 0.0
        limit = k // 2
        for j in range(limit + 1):
            total += nCr(k, 2 * j) * math.factorial(2 * j) * (p ** (2 * j)) / (2.0 ** j)
        return total
    elif dist_type == 'Triangular':
        a = np.sqrt(6.0) * p
        if a <= 1e-8:
            return 1.0
        num = ((1.0 + a) ** (k + 2)) + ((1.0 - a) ** (k + 2)) - 2.0
        den = (a ** 2) * (k + 1) * (k + 2)
        return num / den
    elif dist_type == 'Uniform':
        a = np.sqrt(3.0) * p
        if a <= 1e-8:
            return 1.0
        num = ((1.0 + a) ** (k + 1)) - ((1.0 - a) ** (k + 1))
        den = 2.0 * a * (k + 1)
        return num / den
    return 1.0 

def calculate_indices_from_p(p, dist_type):
    # Tomchuk's two dimensionless width indices are defined from normalized
    # moments of the size distribution:
    #   PDI  = <R^2><R^8>^2 / <R^6>^3   [paper equation (4)]
    #   PDI2 = <R^2><R^4>   / <R^3>^2   [paper equation (21)]
    m2 = get_normalized_moment(2, p, dist_type)
    m3 = get_normalized_moment(3, p, dist_type)
    m4 = get_normalized_moment(4, p, dist_type)
    m6 = get_normalized_moment(6, p, dist_type)
    m8 = get_normalized_moment(8, p, dist_type)
    PDI, PDI2 = 0, 0
    if m6 > 0: PDI = (m2 * m8**2) / (m6**3)
    if m3 > 0: PDI2 = (m2 * m4) / (m3**2)
    return PDI, PDI2

def solve_p_tomchuk(target_val, index_type, dist_type):
    if target_val is None or target_val < 1.0001: return 0.0
    # The paper uses family-specific PDI-p and PDI2-p relations (Fig. 4 / Table 1).
    # Here we invert those relations numerically with a bounded scalar root solve.
    def func(p_guess):
        pdi, pdi2 = calculate_indices_from_p(p_guess, dist_type)
        return (pdi if index_type == 'PDI' else pdi2) - target_val
    try:
        f0 = func(0.001)
        f6 = func(6.0)
        if f0 * f6 > 0: return 0.0 
        return bisect(func, 0.001, 6.0, xtol=1e-4)
    except: return 0.0

def get_calculated_mean_radius(rg_scat, p, dist_type):
    if not rg_scat or rg_scat <= 0: return 0
    if p <= 0: return np.sqrt(5.0 / 3.0) * rg_scat
    # For spheres, Rg depends on <R^8>/<R^6>. This implements the same
    # moment-based back-conversion used to recover the mean size after p is known.
    m6 = get_normalized_moment(6, p, dist_type)
    m8 = get_normalized_moment(8, p, dist_type)
    if m6 <= 0 or m8 <= 0: return 0
    ratio = np.sqrt(m8 / m6)
    return np.sqrt(5.0 / 3.0) * rg_scat / ratio

def get_calculated_mean_rg_num(rg_scat, p, dist_type):
    return get_calculated_mean_radius(rg_scat, p, dist_type)

def mean_radius_to_mean_rg(mean_radius):
    if not mean_radius or mean_radius <= 0:
        return 0
    return mean_radius * np.sqrt(3.0 / 5.0)

def unified_beaucage_intensity(q, G, Rg, B):
    # Unified exponential/power-law model [Tomchuk / Beaucage equation (1)]:
    # I(q) = G exp(-q^2 Rg^2 / 3) + B q^-4 [erf(q Rg / sqrt(6))]^12
    # This is the core model used to obtain G, Rg and B before building
    # the Tomchuk indices.
    q = np.asarray(q, dtype=float)
    q_safe = np.maximum(q, 1e-12)
    erf_term = erf(q_safe * Rg / np.sqrt(6.0))
    porod = B * (q_safe ** -4.0) * (erf_term ** 12)
    return G * np.exp(-(q_safe ** 2) * (Rg ** 2) / 3.0) + porod

def fit_unified_beaucage(q_exp, i_exp, rg_init, g_init):
    if len(q_exp) < 5:
        return None

    q_fit = np.asarray(q_exp, dtype=float)
    i_fit = np.asarray(i_exp, dtype=float)
    valid = (q_fit > 0) & np.isfinite(q_fit) & np.isfinite(i_fit) & (i_fit > 0)
    q_fit = q_fit[valid]
    i_fit = i_fit[valid]
    if len(q_fit) < 5:
        return None

    # Initialize B from the high-q Porod region by sampling I(q) q^4 near the tail.
    # This is an initialization heuristic for the non-linear fit, not a final Tomchuk
    # equation from the paper.
    n_tail = max(5, min(20, len(q_fit) // 4))
    b_init = np.median(i_fit[-n_tail:] * (q_fit[-n_tail:] ** 4))
    if not np.isfinite(b_init) or b_init <= 0:
        b_init = np.max(i_fit) * max(q_fit[-1], 1e-3) ** 4

    g0 = max(float(g_init), np.max(i_fit), 1e-12)
    rg0 = max(float(rg_init), 1e-3)
    b0 = max(float(b_init), 1e-12)

    def residuals(log_params):
        g_fit, rg_fit, b_fit = np.exp(log_params)
        i_model = unified_beaucage_intensity(q_fit, g_fit, rg_fit, b_fit)
        return np.log(np.maximum(i_model, 1e-300)) - np.log(np.maximum(i_fit, 1e-300))

    best = None
    best_cost = np.inf
    lower = np.log([1e-30, max(rg0 * 0.2, 1e-6), 1e-30])
    upper = np.log([1e30, max(rg0 * 4.0, 1.0), 1e30])
    for rg_scale in (0.5, 1.0, 1.5):
        for b_scale in (0.25, 1.0, 4.0):
            p0 = np.log([g0, max(rg0 * rg_scale, 1e-3), max(b0 * b_scale, 1e-12)])
            try:
                fit = least_squares(
                    residuals,
                    p0,
                    bounds=(lower, upper),
                    loss='soft_l1',
                    max_nfev=30000,
                )
                if fit.success and fit.cost < best_cost:
                    best = np.exp(fit.x)
                    best_cost = fit.cost
            except Exception:
                continue

    if best is None:
        return None

    return {
        'G': float(best[0]),
        'Rg': float(best[1]),
        'B': float(best[2]),
        'I_fit': unified_beaucage_intensity(q_exp, *best),
    }

def estimate_guinier_parameters(q_exp, i_exp, initial_rg_guess, mode):
    """Estimate Guinier Rg and G from the low-q region.

    This is the low-q seed used before the unified fit. It corresponds to the
    dashed Guinier term shown in Tomchuk's Fig. 1, not to one of the final
    invariant equations. The later Tomchuk workflow may replace this raw
    Guinier estimate with the unified-fit radius selected from the whole curve.
    """
    if len(q_exp) < 5:
        return 0.0, 0.0, False

    rg_init = float(initial_rg_guess)
    # Keep the Guinier window conservative. For spheres we stay below qRg ~ 0.9
    # so the low-q exponential approximation remains self-consistent.
    limit = 1.0 if mode == 'IDP' else 0.9
    min_pts = 6
    try:
        valid_pts = (q_exp > 0) & (i_exp > 0)
        q_v = q_exp[valid_pts]
        i_v = i_exp[valid_pts]
        if len(q_v) > 5:
            n_init = min(15, len(q_v))
            x_init = q_v[:n_init] ** 2
            y_init = np.log(i_v[:n_init])
            slope_init, _ = np.polyfit(x_init, y_init, 1)
            if slope_init < 0:
                rg_est = np.sqrt(-3 * slope_init)
                if rg_est > 0:
                    rg_init = rg_est
    except Exception:
        pass

    rg_fit = max(rg_init, 1e-6)
    g_fit = float(i_exp[0]) if len(i_exp) > 0 else 1.0
    valid_fit = False
    for _ in range(5):
        mask = (q_exp * rg_fit) < limit
        mask = mask & (q_exp > 0) & (i_exp > 0)
        if np.sum(mask) < min_pts:
            break
        x_fit = q_exp[mask] ** 2
        y_fit = np.log(i_exp[mask])
        try:
            slope, intercept = np.polyfit(x_fit, y_fit, 1)
            if slope >= 0:
                break
            rg_candidate = np.sqrt(np.abs(-3 * slope))
            if not np.isfinite(rg_candidate) or rg_candidate <= 0:
                break
            next_mask = (q_exp * rg_candidate) < limit
            next_mask = next_mask & (q_exp > 0) & (i_exp > 0)
            if np.sum(next_mask) < min_pts:
                break
            rg_fit = rg_candidate
            g_fit = np.exp(intercept)
            valid_fit = True
        except Exception:
            break

    return float(rg_fit), float(g_fit), bool(valid_fit)

def compute_tomchuk_analytic_quantities(G, Rg, B):
    """Apply the paper's analytic invariant route.

    Uses:
      Q    from equation (8a)
      lc   from equation (8b)
      PDI  from equation (4), rewritten via G, Rg and B
      PDI2 from equation (21), rewritten via B, lc and Q

    This is the preferred implementation path because it follows the unified-fit
    logic of the paper and avoids finite-q numerical integration.
    """
    if G <= 0 or Rg <= 0 or B < 0:
        return {
            'Q': 0.0,
            'lc': 0.0,
            'PDI': 0.0,
            'PDI2': 0.0,
        }

    q_val = ((3.0 * np.sqrt(3.0 * np.pi)) * G) / (4.0 * (Rg ** 3))
    q_val += (B * Rg * TOMCHUK_C1) / np.sqrt(6.0)

    lc_val = 0.0
    if q_val > 0:
        lc_val = (3.0 * np.pi * G) / (2.0 * (Rg ** 2) * q_val)
        lc_val += (np.pi * B * (Rg ** 2) * TOMCHUK_C2) / (6.0 * q_val)

    pdi_val = (50.0 / 81.0) * (B * (Rg ** 4)) / G if G > 0 else 0.0
    pdi2_val = (2.0 * np.pi / 9.0) * (B * lc_val) / q_val if q_val > 0 and lc_val > 0 else 0.0

    return {
        'Q': float(q_val),
        'lc': float(lc_val),
        'PDI': float(pdi_val),
        'PDI2': float(pdi2_val),
    }

def compute_tomchuk_hybrid_quantities(q_exp, i_exp, G, Rg):
    """Legacy finite-q fallback for Tomchuk quantities.

    This path predates the current unified-fit-first workflow. B is estimated from
    the measured Porod tail, then Q and lc are obtained by finite-q integration plus
    explicit asymptotic tail terms:
      q_tail  ~ B / q_max
      lc_tail ~ B / (2 q_max^2)

    These tail corrections are an implementation update relative to the main paper
    concept, which recommends using analytic equations (8a) and (8b) from the
    unified fit whenever possible to reduce PDI2 error.
    """
    n_fit_b = max(8, min(40, len(q_exp) // 6))
    b_est = 0.0
    if n_fit_b > 0:
        b_region_i = i_exp[-n_fit_b:]
        b_region_q = q_exp[-n_fit_b:]
        b_est = np.median(b_region_i * (b_region_q ** 4))
        if not np.isfinite(b_est) or b_est < 0:
            b_est = 0.0

    q_val = 0.0
    lc_val = 0.0
    if G > 0 and Rg > 0 and len(q_exp) > 0:
        q_max_meas = q_exp[-1]
        integrand_q = (q_exp ** 2) * i_exp
        q_obs = trapezoid(integrand_q, q_exp)
        q_tail = b_est / q_max_meas if q_max_meas > 0 else 0.0
        q_val = q_obs + q_tail

        integrand_lc = q_exp * i_exp
        lin_obs = trapezoid(integrand_lc, q_exp)
        lin_tail = b_est / (2.0 * q_max_meas ** 2) if q_max_meas > 0 else 0.0
        lc_val = (np.pi / q_val) * (lin_obs + lin_tail) if q_val > 0 else 0.0

    pdi_val = (50.0 / 81.0) * (b_est * (Rg ** 4)) / G if G > 0 and Rg > 0 else 0.0
    pdi2_val = (2.0 * np.pi / 9.0) * (b_est * lc_val) / q_val if q_val > 0 and lc_val > 0 else 0.0

    return {
        'B': float(b_est),
        'Q': float(q_val),
        'lc': float(lc_val),
        'PDI': float(pdi_val),
        'PDI2': float(pdi2_val),
    }

def extract_tomchuk_parameters(q_exp, i_exp, initial_rg_guess):
    # Pipeline:
    #   1. low-q Guinier seed for raw Rg/G
    #   2. unified model fit [equation (1)] to get G, Rg, B over the full curve
    #   3. analytic Tomchuk invariants/indices from equations (4), (8a), (8b), (21)
    #   4. fallback to the older hybrid finite-q route only when the unified path
    #      is not numerically usable
    rg_guinier, g_guinier, valid_guinier = estimate_guinier_parameters(q_exp, i_exp, initial_rg_guess, 'Sphere')
    fit_res = fit_unified_beaucage(q_exp, i_exp, rg_guinier, g_guinier) if valid_guinier else None

    hybrid = compute_tomchuk_hybrid_quantities(q_exp, i_exp, g_guinier, rg_guinier)
    unified = None
    if fit_res:
        unified = {'G': fit_res['G'], 'Rg': fit_res['Rg'], 'B': fit_res['B']}
        unified.update(compute_tomchuk_analytic_quantities(fit_res['G'], fit_res['Rg'], fit_res['B']))

    # Keep the hybrid result available for diagnostics, but only promote it when
    # the unified-fit path fails quality checks.
    extraction_source = 'hybrid'
    selected = {
        'Rg': float(rg_guinier),
        'G': float(g_guinier),
        'B': float(hybrid['B']),
        'Q': float(hybrid['Q']),
        'lc': float(hybrid['lc']),
        'PDI': float(hybrid['PDI']),
        'PDI2': float(hybrid['PDI2']),
    }
    if unified and unified['G'] > 0 and unified['Rg'] > 0 and unified['B'] >= 0 and unified['Q'] > 0 and unified['lc'] > 0:
        extraction_source = 'unified_fit'
        selected = {
            'Rg': float(unified['Rg']),
            'G': float(unified['G']),
            'B': float(unified['B']),
            'Q': float(unified['Q']),
            'lc': float(unified['lc']),
            'PDI': float(unified['PDI']),
            'PDI2': float(unified['PDI2']),
        }

    return {
        'selected': selected,
        'source': extraction_source,
        'guinier_valid': valid_guinier,
        'guinier': {'Rg': float(rg_guinier), 'G': float(g_guinier)},
        'hybrid': hybrid,
        'unified': unified,
        'fit_res': fit_res,
    }

def recover_distribution_nnls(q_exp, i_exp, q_min, q_max, kernel_func, max_rg_basis, n_basis=150, smooth_sigma=1.0):
    if q_min <= 0: q_min_eff = 1e-3
    else: q_min_eff = q_min
    
    r_min_basis = max(0.5, 0.5 / q_max)
    r_max_basis = max_rg_basis
    n_basis = max(int(n_basis), 20)
    r_basis = np.logspace(np.log10(r_min_basis), np.log10(r_max_basis), n_basis)
    
    A = kernel_func(q_exp, r_basis)
    weights, resid = nnls(A, i_exp)
    
    i_fit = A @ weights
    
    total_w = np.sum(weights)
    if total_w > 0:
        weights = weights / total_w
    
    mean_val = np.sum(weights * r_basis)
    var = np.sum(weights * (r_basis - mean_val)**2)
    std = np.sqrt(var)
    p_rec = std / mean_val if mean_val > 0 else 0
    
    weights_smooth = gaussian_filter(weights, sigma=max(float(smooth_sigma), 0.0))
    pdf_nnls = weights_smooth.copy()
    area = trapezoid(pdf_nnls, r_basis)
    if area > 0: pdf_nnls /= area
    
    return r_basis, weights, pdf_nnls, mean_val, p_rec, i_fit

def calculate_fit_and_rrms(q_exp, i_exp, r_vals, pdf_vals, kernel_func):
    """Helper to generate I_fit and relative RMS error for a given P(r)."""
    if len(q_exp) == 0: return np.zeros_like(q_exp), 0
    
    # Normalize PDF
    area = trapezoid(pdf_vals, r_vals)
    if area > 0:
        pdf_norm = pdf_vals / area
    else:
        pdf_norm = pdf_vals

    # Integrate: I_calc = Scale * Integral( Kernel(q,r) * P(r) )
    i_mtx = kernel_func(q_exp, r_vals)
    i_calc_shape = trapezoid(i_mtx * pdf_norm, r_vals, axis=1)

    # Scale the reconstructed curve back to the data by least squares. This is
    # a validation step for the recovered distribution, not part of Tomchuk's
    # original derivation.
    num = np.sum(i_exp * i_calc_shape)
    den = np.sum(i_calc_shape**2)
    scale = num/den if den > 0 else 1.0
    i_fit = i_calc_shape * scale

    denom = np.maximum(np.abs(i_exp), np.max(np.abs(i_exp)) * 1e-12 if len(i_exp) > 0 else 1.0)
    rel_residual = (i_exp - i_fit) / denom
    rrms = np.sqrt(np.mean(rel_residual ** 2))

    return i_fit, float(rrms)

def perform_saxs_analysis(q_exp, i_exp, dist_type, initial_rg_guess, mode, method, max_rg_nnls, i_2d=None, analysis_settings=None):
    analysis_settings = {} if analysis_settings is None else analysis_settings
    results = {
        'Rg': 0, 'G': 0, 'B': 0, 'Q': 0, 'lc': 0,
        'Rg_guinier': 0, 'G_guinier': 0,
        'PDI': 0, 'PDI2': 0, 
        'p_rec_pdi': 0, 'p_rec_pdi2': 0,
        'mean_r_rec_pdi': 0, 'mean_r_rec_pdi2': 0,
        'rg_num_rec_pdi': 0, 'rg_num_rec_pdi2': 0,
        'p_rec': 0, 'rg_num_rec': 0, 'mean_r_rec': 0,
        'I_fit_pdi': [], 'I_fit_pdi2': [], 'I_fit': [], 'I_fit_unified': [],
        'rrms_pdi': 0, 'rrms_pdi2': 0, 'rrms': 0,
        'tomchuk_extraction': 'none',
        'tenor_candidate_count': 0,
    }
    
    if len(q_exp) < 5: return results

    rg_fit, g_fit, valid_fit = estimate_guinier_parameters(q_exp, i_exp, initial_rg_guess, mode)
    results['Rg'] = rg_fit
    results['G'] = g_fit
    results['Rg_guinier'] = rg_fit
    results['G_guinier'] = g_fit

    i_fit_global = np.zeros_like(i_exp)

    if mode == 'Sphere':
        if method == 'Tomchuk':
            # Tomchuk sphere workflow:
            #   - extract G, Rg, B, Q, lc
            #   - compute PDI / PDI2
            #   - invert the family-specific PDI-p or PDI2-p relation
            #   - recover the mean radius once p is known
            extraction = extract_tomchuk_parameters(q_exp, i_exp, initial_rg_guess)
            fit_res = extraction.get('fit_res')
            if fit_res:
                results['I_fit_unified'] = fit_res['I_fit']
            selected = extraction['selected']
            rg_fit = selected['Rg']
            g_fit = selected['G']
            b_est = selected['B']
            q_val = selected['Q']
            lc_val = selected['lc']
            pdi_val = selected['PDI']
            pdi2_val = selected['PDI2']

            p_rec_pdi = solve_p_tomchuk(pdi_val, 'PDI', dist_type)
            p_rec_pdi2 = solve_p_tomchuk(pdi2_val, 'PDI2', dist_type)
            
            mean_r_rec = get_calculated_mean_radius(rg_fit, p_rec_pdi, dist_type)
            mean_r_rec_2 = get_calculated_mean_radius(rg_fit, p_rec_pdi2, dist_type)

            results['Rg'] = rg_fit
            results['G'] = g_fit
            results['Q'] = q_val
            results['lc'] = lc_val
            results['B'] = b_est
            results['PDI'] = pdi_val
            results['PDI2'] = pdi2_val
            results['p_rec_pdi'] = p_rec_pdi
            results['p_rec_pdi2'] = p_rec_pdi2
            results['mean_r_rec_pdi'] = mean_r_rec
            results['mean_r_rec_pdi2'] = mean_r_rec_2
            results['rg_num_rec_pdi'] = mean_radius_to_mean_rg(mean_r_rec)
            results['rg_num_rec_pdi2'] = mean_radius_to_mean_rg(mean_r_rec_2)
            results['method'] = 'Tomchuk'
            results['tomchuk_extraction'] = extraction['source']
            results['tomchuk_hybrid'] = extraction['hybrid']
            results['tomchuk_unified'] = extraction['unified']
            
            # Reconstruct scattering curves from the recovered parameters so we can
            # compare the PDI-based and PDI2-based solutions by forward-model RRMS.
            max_rec_r = max(mean_r_rec, mean_r_rec_2, 0)
            r_sim = np.linspace(max(0.1, max_rec_r * 0.1), max(max_rec_r * 5, 1.0), 200)
            
            # PDI Fit
            pdf_sim_pdi = get_distribution(dist_type, r_sim, mean_r_rec, p_rec_pdi)
            i_fit_pdi, rrms_pdi = calculate_fit_and_rrms(q_exp, i_exp, r_sim, pdf_sim_pdi, sphere_form_factor)
            results['I_fit_pdi'] = i_fit_pdi
            results['rrms_pdi'] = rrms_pdi
            
            # PDI2 Fit
            pdf_sim_pdi2 = get_distribution(dist_type, r_sim, mean_r_rec_2, p_rec_pdi2)
            i_fit_pdi2, rrms_pdi2 = calculate_fit_and_rrms(q_exp, i_exp, r_sim, pdf_sim_pdi2, sphere_form_factor)
            results['I_fit_pdi2'] = i_fit_pdi2
            results['rrms_pdi2'] = rrms_pdi2
            
            if len(results['I_fit_unified']) != len(q_exp):
                results['I_fit'] = i_fit_pdi
            results['rrms'] = rrms_pdi
            
        elif method == 'NNLS':
            r_nnls, w_nnls, pdf_nnls, mean_r_nnls, p_nnls, i_fit_global = recover_distribution_nnls(
                q_exp,
                i_exp,
                q_exp[0],
                q_exp[-1],
                sphere_form_factor,
                max_rg_basis=max_rg_nnls,
                n_basis=analysis_settings.get('nnls_basis_count', 150),
                smooth_sigma=analysis_settings.get('nnls_smooth_sigma', 1.0),
            )
            results['rg_num_rec'] = mean_r_nnls * np.sqrt(3.0/5.0)
            results['mean_r_rec'] = mean_r_nnls
            results['p_rec'] = p_nnls
            results['nnls_r'] = r_nnls
            results['nnls_w'] = w_nnls
            results['nnls_pdf'] = pdf_nnls
            results['method'] = 'NNLS'
            results['I_fit'] = i_fit_global
            
            denom = np.maximum(np.abs(i_exp), np.max(np.abs(i_exp)) * 1e-12 if len(i_exp) > 0 else 1.0)
            rel_residual = (i_exp - i_fit_global) / denom
            results['rrms'] = float(np.sqrt(np.mean(rel_residual ** 2)))
        elif method == 'Tenor':
            if i_2d is None:
                raise ValueError("TENOR-SAXS requires a 2D detector image.")
            from tenor_saxs import analyze_tenor_saxs_2d, build_default_psf_pairs
            psf_pairs = build_default_psf_pairs(
                sigma_x_start=analysis_settings.get('tenor_psf_sigma_x_start', 1.2),
                sigma_y_start=analysis_settings.get('tenor_psf_sigma_y_start', 0.6),
                sigma_step=analysis_settings.get('tenor_psf_sigma_step', 0.4),
                pair_count=analysis_settings.get('tenor_psf_count', 5),
                secondary_ratio=analysis_settings.get('tenor_psf_secondary_ratio', 0.5),
            )
            tenor = analyze_tenor_saxs_2d(
                i_2d=i_2d,
                q_max=float(analysis_settings.get('q_max_for_tenor', q_exp[-1] if len(q_exp) > 0 else 1.0)),
                dist_type=dist_type,
                initial_rg_guess=initial_rg_guess,
                psf_pairs=psf_pairs,
                n_radial_bins=int(analysis_settings.get('tenor_radial_bins', 18)),
                qrg_limit=float(analysis_settings.get('tenor_qrg_limit', 0.85)),
                guinier_bins=int(analysis_settings.get('tenor_guinier_bins', 256)),
                calibration_p_grid=build_tenor_p_grid(analysis_settings),
                psf_truncate=float(analysis_settings.get('tenor_psf_truncate', 4.0)),
                use_m3=bool(analysis_settings.get('tenor_use_m3', True)),
                use_g3=bool(analysis_settings.get('tenor_use_g3', True)),
            )
            results['Rg'] = tenor['mean_rg_rec']
            results['G'] = tenor['g_app']
            results['Rg_guinier'] = tenor['rg_app']
            results['G_guinier'] = tenor['g_app']
            results['p_rec'] = tenor['p_rec']
            results['rg_num_rec'] = tenor['mean_rg_rec']
            results['mean_r_rec'] = tenor['mean_r_rec']
            results['weighted_v'] = tenor['weighted_v']
            results['tenor_raw_g1_over_g0'] = tenor['observable_raw_g1_over_g0']
            results['tenor_raw_g100_ratio'] = tenor['observable_raw_g100_ratio']
            results['tenor_raw_g210_ratio'] = tenor['observable_raw_g210_ratio']
            results['tenor_dimless_jg'] = tenor['observable_dimless_jg']
            results['tenor_raw_m210_ratio'] = tenor['observable_raw_m210_ratio']
            results['tenor_raw_m1_over_m0'] = tenor['observable_raw_m1_over_m0']
            results['tenor_candidate_count'] = tenor['candidate_count']
            results['tenor_best_psf_pair'] = tenor['best_psf_pair']
            results['tenor_g_rmse'] = tenor['best_g_rmse']
            results['tenor_m_rmse'] = tenor['best_m_rmse']
            results['method'] = 'Tenor'
            r_sim = np.linspace(
                max(0.1, tenor['mean_r_rec'] * 0.1),
                max(tenor['mean_r_rec'] * 5.0, 1.0),
                max(int(analysis_settings.get('radius_samples', 400)), 50),
            )
            pdf_sim = get_distribution(dist_type, r_sim, tenor['mean_r_rec'], tenor['p_rec'])
            i_fit_global, rrms = calculate_fit_and_rrms(q_exp, i_exp, r_sim, pdf_sim, sphere_form_factor)
            results['I_fit'] = i_fit_global
            results['rrms'] = rrms

    elif mode == 'IDP':
        r_nnls, w_nnls, pdf_nnls, mean_rg_nnls, p_nnls, i_fit_global = recover_distribution_nnls(
            q_exp,
            i_exp,
            q_exp[0],
            q_exp[-1],
            debye_form_factor,
            max_rg_basis=max_rg_nnls,
            n_basis=analysis_settings.get('nnls_basis_count', 150),
            smooth_sigma=analysis_settings.get('nnls_smooth_sigma', 1.0),
        )
        results['p_rec'] = p_nnls
        results['rg_num_rec'] = mean_rg_nnls
        results['nnls_r'] = r_nnls
        results['nnls_w'] = w_nnls
        results['nnls_pdf'] = pdf_nnls
        results['method'] = 'NNLS'
        results['I_fit'] = i_fit_global
        
        denom = np.maximum(np.abs(i_exp), np.max(np.abs(i_exp)) * 1e-12 if len(i_exp) > 0 else 1.0)
        rel_residual = (i_exp - i_fit_global) / denom
        results['rrms'] = float(np.sqrt(np.mean(rel_residual ** 2)))
    
    return results

def get_header_string(params, analysis_res):
    header_lines = [
        f"# SAXS Analysis Results",
        f"# Simulation Mode: {params['mode']}",
        f"# Analysis Method: {params['method']}",
        "# ==========================================",
        "# Input/Reference Parameters:",
        f"#   Mean Rg: {float(params['mean_rg']):.4f} nm",
        f"#   Mean Radius: {float(params['mean_rg']) * np.sqrt(5.0/3.0):.4f} nm" if params['mode'] == 'Sphere' else f"#   Mean Radius: n/a",
        f"#   Polydispersity (p): {float(params['p_val']):.4f}",
        f"#   Distribution Type: {params['dist_type']}",
        "#",
        "# Analysis Output:",
        f"#   Rg (Guinier Fit): {analysis_res.get('Rg_guinier', analysis_res.get('Rg', 0)):.6f} nm",
        f"#   G (Guinier Intercept): {analysis_res.get('G_guinier', analysis_res.get('G', 0)):.6e}",
    ]
    
    if params['method'] == 'Tomchuk':
        header_lines.extend([
            f"#   Tomchuk Extraction Path: {analysis_res.get('tomchuk_extraction', 'n/a')}",
            f"#   Rg (Selected Tomchuk): {analysis_res.get('Rg', 0):.6f} nm",
            f"#   G (Selected Tomchuk): {analysis_res.get('G', 0):.6e}",
            f"#   B (Porod Constant): {analysis_res.get('B', 0):.6e}",
            f"#   Porod Invariant (Q): {analysis_res.get('Q', 0):.6e}",
            f"#   PDI (Calculated): {analysis_res.get('PDI', 0):.6f}",
            f"#   PDI2 (Calculated): {analysis_res.get('PDI2', 0):.6f}",
            f"#   Recovered p (from PDI): {analysis_res.get('p_rec_pdi', 0):.6f}",
            f"#   Recovered Mean Radius (from PDI): {analysis_res.get('mean_r_rec_pdi', 0):.6f} nm",
            f"#   Fit RelRMS (PDI): {analysis_res.get('rrms_pdi', 0):.6f}",
            f"#   Recovered p (from PDI2): {analysis_res.get('p_rec_pdi2', 0):.6f}",
            f"#   Recovered Mean Radius (from PDI2): {analysis_res.get('mean_r_rec_pdi2', 0):.6f} nm",
            f"#   Fit RelRMS (PDI2): {analysis_res.get('rrms_pdi2', 0):.6f}",
        ])
    elif params['method'] == 'Tenor':
        header_lines.extend([
            f"#   Rg (Apparent Guinier): {analysis_res.get('Rg_guinier', 0):.6f} nm",
            f"#   Mean Rg (Recovered): {analysis_res.get('Rg', 0):.6f} nm",
            f"#   Mean Radius (Recovered): {analysis_res.get('mean_r_rec', 0):.6f} nm",
            f"#   Recovered p: {analysis_res.get('p_rec', 0):.6f}",
            f"#   Weighted Variance: {analysis_res.get('weighted_v', 0):.6f}",
            f"#   Raw g1/g0: {analysis_res.get('tenor_raw_g1_over_g0', 0):.6e}",
            f"#   Dimensionless J_G: {analysis_res.get('tenor_dimless_jg', 0):.6e}",
            f"#   Candidate PSF Pairs: {analysis_res.get('tenor_candidate_count', 0)}",
            f"#   Fit RelRMS: {analysis_res.get('rrms', 0):.6f}",
        ])
    else: # NNLS
        header_lines.extend([
            f"#   Recovered Mean Radius: {analysis_res.get('mean_r_rec', 0):.6f} nm" if params['mode'] == 'Sphere' else f"#   Recovered Rg (Mean): {analysis_res.get('rg_num_rec', 0):.6f} nm",
            f"#   Recovered p (Width): {analysis_res.get('p_rec', 0):.6f}",
            f"#   Fit RelRMS: {analysis_res.get('rrms', 0):.6f}"
        ])
    
    header_lines.append("# ==========================================")
    return "\n".join(header_lines) + "\n"

def create_intensity_csv(header, q, i_input, analysis_res, method):
    df = pd.DataFrame({'q': q, 'I_input': i_input})
    
    if method == 'Tomchuk':
        if len(analysis_res.get('I_fit_pdi', [])) == len(q):
            df['I_Fit_PDI'] = analysis_res['I_fit_pdi']
        if len(analysis_res.get('I_fit_pdi2', [])) == len(q):
            df['I_Fit_PDI2'] = analysis_res['I_fit_pdi2']
    else:
        if len(analysis_res.get('I_fit', [])) == len(q):
            fit_label = 'I_Fit_Tenor' if method == 'Tenor' else 'I_Fit_NNLS'
            df[fit_label] = analysis_res['I_fit']
            
    return header + df.to_csv(index=False)

def create_distribution_csv(header, r, input_dist, recovered_dists, params):
    df = pd.DataFrame({'Radius': r})
    label = "Reference_PDF"
    if input_dist is not None and len(input_dist) == len(r):
        df[label] = input_dist
        
    if params['method'] == 'Tomchuk':
        if 'pdi' in recovered_dists:
            df['PDI_Recovered_PDF'] = recovered_dists['pdi']
        if 'pdi2' in recovered_dists:
            df['PDI2_Recovered_PDF'] = recovered_dists['pdi2']
    elif params['method'] == 'Tenor':
        if 'tenor' in recovered_dists:
            df['Tenor_Recovered_PDF'] = recovered_dists['tenor']
    else: 
        if 'nnls_r' in recovered_dists and 'nnls_pdf' in recovered_dists:
             df['NNLS_Recovered_PDF'] = np.interp(r, recovered_dists['nnls_r'], recovered_dists['nnls_pdf'], left=0, right=0)
             
    return header + df.to_csv(index=False)

def get_true_size(params):
    if params['mode'] == 'Sphere':
        return float(params['mean_rg']) * np.sqrt(5.0 / 3.0)
    return float(params['mean_rg'])

def get_recovered_size(analysis_res, params, variant='primary'):
    if params['mode'] == 'IDP':
        return analysis_res.get('rg_num_rec', 0) if params['method'] == 'NNLS' else analysis_res.get('Rg', 0)

    if params['method'] == 'NNLS':
        return analysis_res.get('mean_r_rec', 0)
    if params['method'] == 'Tenor':
        return analysis_res.get('mean_r_rec', 0)
    if variant == 'pdi2':
        return analysis_res.get('mean_r_rec_pdi2', 0)
    return analysis_res.get('mean_r_rec_pdi', 0)

def build_recovered_distributions(params, analysis_res, r_vals):
    recovered_dists = {}
    if params['method'] == 'Tomchuk':
        mean_r_pdi = analysis_res.get('mean_r_rec_pdi', 0)
        if mean_r_pdi > 0:
            recovered_dists['pdi'] = get_distribution(params['dist_type'], r_vals, mean_r_pdi, analysis_res.get('p_rec_pdi', 0))

        mean_r_pdi2 = analysis_res.get('mean_r_rec_pdi2', 0)
        if mean_r_pdi2 > 0:
            recovered_dists['pdi2'] = get_distribution(params['dist_type'], r_vals, mean_r_pdi2, analysis_res.get('p_rec_pdi2', 0))
    elif params['method'] == 'Tenor':
        mean_r_tenor = analysis_res.get('mean_r_rec', 0)
        if mean_r_tenor > 0:
            recovered_dists['tenor'] = get_distribution(params['dist_type'], r_vals, mean_r_tenor, analysis_res.get('p_rec', 0))
    elif 'nnls_r' in analysis_res:
        recovered_dists['nnls_r'] = analysis_res['nnls_r']
        recovered_dists['nnls_pdf'] = analysis_res.get('nnls_pdf', analysis_res.get('nnls_w', []))
    return recovered_dists

def build_summary_row(params, analysis_res, base_row=None):
    summary_row = {} if base_row is None else base_row.copy()
    p_true = float(params['p_val'])
    true_size = get_true_size(params)

    rec_p = analysis_res.get('p_rec', 0) if params['method'] in ('NNLS', 'Tenor') else analysis_res.get('p_rec_pdi', 0)
    rec_size = get_recovered_size(analysis_res, params, variant='primary')

    summary_row.update({
        'Recovered_p': rec_p,
        'Recovered_Size': rec_size,
        'Rel_Err_p': (rec_p - p_true) / p_true if p_true != 0 else 0,
        'Rel_Err_Size': (rec_size - true_size) / true_size if true_size != 0 else 0,
        'RelRMS': analysis_res.get('rrms', 0)
    })

    if params['method'] == 'Tomchuk':
        rec_p_2 = analysis_res.get('p_rec_pdi2', 0)
        rec_size_2 = get_recovered_size(analysis_res, params, variant='pdi2')
        summary_row.update({
            'Tomchuk_Extraction': analysis_res.get('tomchuk_extraction', 'none'),
            'Recovered_p_PDI2': rec_p_2,
            'Recovered_Size_PDI2': rec_size_2,
            'Rel_Err_p_PDI2': (rec_p_2 - p_true) / p_true if p_true != 0 else 0,
            'Rel_Err_Size_PDI2': (rec_size_2 - true_size) / true_size if true_size != 0 else 0,
        })

    return summary_row

def normalize_pdf(r_vals, pdf_vals):
    r_vals = np.asarray(r_vals, dtype=float)
    pdf_vals = np.asarray(pdf_vals, dtype=float)
    if len(r_vals) == 0 or len(pdf_vals) == 0:
        return pdf_vals
    area = trapezoid(pdf_vals, r_vals)
    if area > 0:
        return pdf_vals / area
    return pdf_vals

def calculate_sphere_theoretical_parameters(q_vals, i_vals, r_vals, pdf_vals):
    q_vals = np.asarray(q_vals, dtype=float)
    i_vals = np.asarray(i_vals, dtype=float)
    r_vals = np.asarray(r_vals, dtype=float)
    pdf_norm = normalize_pdf(r_vals, pdf_vals)

    moments = {}
    for order in (2, 3, 4, 6, 8):
        moments[order] = trapezoid((r_vals ** order) * pdf_norm, r_vals)

    i_matrix = sphere_form_factor(q_vals, r_vals)
    i_shape = trapezoid(i_matrix * pdf_norm, r_vals, axis=1)
    num = np.sum(i_vals * i_shape)
    den = np.sum(i_shape ** 2)
    scale = num / den if den > 0 else 1.0

    mean_radius = moments[1] = trapezoid(r_vals * pdf_norm, r_vals)
    mean_square = trapezoid((r_vals ** 2) * pdf_norm, r_vals)
    std_radius = np.sqrt(max(mean_square - mean_radius ** 2, 0))
    p_true = std_radius / mean_radius if mean_radius > 0 else 0

    rg_true = np.sqrt((3.0 * moments[8]) / (5.0 * moments[6])) if moments[6] > 0 else 0
    g_true = scale * ((4.0 * np.pi / 3.0) ** 2) * moments[6]
    b_true = scale * 8.0 * (np.pi ** 2) * moments[2]
    q_true = scale * (8.0 * (np.pi ** 3) / 3.0) * moments[3]
    lc_true = 1.5 * moments[4] / moments[3] if moments[3] > 0 else 0
    pdi_true = (moments[2] * (moments[8] ** 2)) / (moments[6] ** 3) if moments[6] > 0 else 0
    pdi2_true = (moments[2] * moments[4]) / (moments[3] ** 2) if moments[3] > 0 else 0

    return {
        'scale': scale,
        'mean_radius': mean_radius,
        'p_true': p_true,
        'Rg': rg_true,
        'G': g_true,
        'B': b_true,
        'Q': q_true,
        'lc': lc_true,
        'PDI': pdi_true,
        'PDI2': pdi2_true,
        'I_shape': i_shape * scale,
    }

def calculate_sphere_input_theoretical_parameters(mean_rg, p_val, dist_type):
    mean_rg = float(mean_rg)
    p_val = max(float(p_val), 1e-6)
    mean_radius = mean_rg * np.sqrt(5.0 / 3.0)

    moments = {}
    for order in (2, 3, 4, 6, 8):
        moments[order] = (mean_radius ** order) * get_normalized_moment(order, p_val, dist_type)

    rg_true = np.sqrt((3.0 * moments[8]) / (5.0 * moments[6])) if moments[6] > 0 else 0
    g_true = ((4.0 * np.pi / 3.0) ** 2) * moments[6]
    b_true = 8.0 * (np.pi ** 2) * moments[2]
    q_true = (8.0 * (np.pi ** 3) / 3.0) * moments[3]
    lc_true = 1.5 * moments[4] / moments[3] if moments[3] > 0 else 0
    pdi_true = (moments[2] * (moments[8] ** 2)) / (moments[6] ** 3) if moments[6] > 0 else 0
    pdi2_true = (moments[2] * moments[4]) / (moments[3] ** 2) if moments[3] > 0 else 0

    return {
        'mean_radius': mean_radius,
        'p_true': p_val,
        'Rg': rg_true,
        'G': g_true,
        'B': b_true,
        'Q': q_true,
        'lc': lc_true,
        'PDI': pdi_true,
        'PDI2': pdi2_true,
        'scale_dependent_keys': {'G', 'B', 'Q'},
    }

def normalize_simulated_sphere_intensity(q_vals, i_vals, r_vals, pdf_vals):
    if len(q_vals) == 0 or len(i_vals) == 0:
        return np.asarray(i_vals, dtype=float), 1.0
    theory_from_data = calculate_sphere_theoretical_parameters(q_vals, i_vals, r_vals, pdf_vals)
    scale = float(theory_from_data.get('scale', 1.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.asarray(i_vals, dtype=float) / scale, scale

def evaluate_tomchuk_sanity_checks(q_vals, i_vals, r_vals, pdf_vals, analysis_res, rel_tol=0.2):
    expected = calculate_sphere_theoretical_parameters(q_vals, i_vals, r_vals, pdf_vals)
    observed = {
        'Rg': analysis_res.get('Rg', 0),
        'G': analysis_res.get('G', 0),
        'B': analysis_res.get('B', 0),
        'Q': analysis_res.get('Q', 0),
        'lc': analysis_res.get('lc', 0),
        'PDI': analysis_res.get('PDI', 0),
        'PDI2': analysis_res.get('PDI2', 0),
        'mean_radius_pdi': analysis_res.get('mean_r_rec_pdi', 0),
        'mean_radius_pdi2': analysis_res.get('mean_r_rec_pdi2', 0),
        'p_rec_pdi': analysis_res.get('p_rec_pdi', 0),
        'p_rec_pdi2': analysis_res.get('p_rec_pdi2', 0),
    }

    comparisons = {
        'Rg': ('Rg', 'Rg'),
        'G': ('G', 'G'),
        'B': ('B', 'B'),
        'Q': ('Q', 'Q'),
        'lc': ('lc', 'lc'),
        'PDI': ('PDI', 'PDI'),
        'PDI2': ('PDI2', 'PDI2'),
        'mean_radius_pdi': ('mean_radius_pdi', 'mean_radius'),
        'mean_radius_pdi2': ('mean_radius_pdi2', 'mean_radius'),
        'p_rec_pdi': ('p_rec_pdi', 'p_true'),
        'p_rec_pdi2': ('p_rec_pdi2', 'p_true'),
    }

    metrics = {}
    failures = []
    for label, (obs_key, exp_key) in comparisons.items():
        obs = observed[obs_key]
        exp = expected[exp_key]
        if exp != 0:
            rel_err = (obs - exp) / exp
        else:
            rel_err = 0 if obs == 0 else np.inf
        metrics[label] = {
            'observed': obs,
            'expected': exp,
            'rel_err': rel_err,
            'pass': np.isfinite(rel_err) and abs(rel_err) <= rel_tol,
        }
        if not metrics[label]['pass']:
            failures.append(label)

    suggestions = []
    if any(key in failures for key in ('Rg', 'G')):
        suggestions.append("Low-q extraction looks unstable: inspect the Guinier window, reduce smearing, and keep enough low-q points below qRg about 1.")
    if 'B' in failures:
        suggestions.append("High-q Porod extraction looks unstable: extend q_max, reduce smearing, and check whether the fitted unified model is capturing the Porod tail.")
    if any(key in failures for key in ('Q', 'lc', 'PDI2', 'p_rec_pdi2')):
        suggestions.append("Invariant-based extraction is drifting: prefer the unified-fit analytic Q/lc path, extend the measured q-range, and inspect any residual tail-correction dependence.")
    if any(key in failures for key in ('PDI', 'p_rec_pdi')):
        suggestions.append("PDI-based recovery is drifting: inspect the consistency of G, Rg, and B extraction against the simulated curve and compare unified-fit versus hybrid estimates.")
    if any(key in failures for key in ('mean_radius_pdi', 'mean_radius_pdi2')):
        suggestions.append("Recovered mean size is drifting: inspect moment-to-size conversion and the chosen distribution family.")
    if not suggestions:
        suggestions.append("All current sanity checks passed within tolerance.")

    return {
        'expected': expected,
        'observed': observed,
        'metrics': metrics,
        'failures': failures,
        'suggestions': suggestions,
        'rel_tol': rel_tol,
    }

def build_sanity_summary_row(q_vals, i_vals, r_vals, pdf_vals, analysis_res, rel_tol=0.2, base_row=None):
    sanity = evaluate_tomchuk_sanity_checks(q_vals, i_vals, r_vals, pdf_vals, analysis_res, rel_tol=rel_tol)
    summary_row = {} if base_row is None else base_row.copy()
    summary_row.update({
        'Sanity_Pass': len(sanity['failures']) == 0,
        'Sanity_Failures': ",".join(sanity['failures']) if sanity['failures'] else "none",
        'Sanity_Suggestions': " | ".join(sanity['suggestions']),
    })
    for key, value in sanity['metrics'].items():
        summary_row[f'Sanity_RelErr_{key}'] = value['rel_err']
    return summary_row

def classify_reconstruction_quality(rrms):
    if not np.isfinite(rrms) or rrms <= 0:
        return "n/a"
    if rrms <= 0.02:
        return "strong"
    if rrms <= 0.05:
        return "usable"
    if rrms <= 0.10:
        return "weak"
    return "poor"

def build_reconstruction_quality_summary(analysis_res):
    rrms_pdi = float(analysis_res.get('rrms_pdi', 0))
    rrms_pdi2 = float(analysis_res.get('rrms_pdi2', 0))
    summary = {
        'rrms_pdi': rrms_pdi,
        'rrms_pdi2': rrms_pdi2,
        'quality_pdi': classify_reconstruction_quality(rrms_pdi),
        'quality_pdi2': classify_reconstruction_quality(rrms_pdi2),
        'best_variant': 'n/a',
        'best_rrms': 0.0,
    }
    candidates = []
    if rrms_pdi > 0:
        candidates.append(('PDI', rrms_pdi))
    if rrms_pdi2 > 0:
        candidates.append(('PDI2', rrms_pdi2))
    if candidates:
        best_variant, best_rrms = min(candidates, key=lambda item: item[1])
        summary['best_variant'] = best_variant
        summary['best_rrms'] = best_rrms
    return summary

def recommend_tomchuk_settings(
    mean_rg,
    p_val,
    dist_type='Gaussian',
    pixels=1024,
    smearing=1.0,
    smearing_x=None,
    smearing_y=None,
    flux=1e12,
    noise=False,
    q_min=0.0,
    binning_mode='Logarithmic',
    q_max_values=None,
    n_bin_values=None,
    target_abs_error=0.01,
    safety_margin=0.01,
    mode='Sphere',
    radius_samples=400,
    q_samples=200,
):
    if smearing_x is None:
        smearing_x = smearing
    if smearing_y is None:
        smearing_y = smearing
    if q_max_values is None:
        q_max_values = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    if n_bin_values is None:
        n_bin_values = [128, 256, 512, 768, 1024, 1536, 2048]

    rows = []
    for q_max in q_max_values:
        for n_bins in n_bin_values:
            params = {
                'mean_rg': float(mean_rg),
                'p_val': float(p_val),
                'dist_type': dist_type,
                'mode': mode,
                'pixels': int(pixels),
                'q_min': float(q_min),
                'q_max': float(q_max),
                'n_bins': int(n_bins),
                'smearing': 0.5 * (float(smearing_x) + float(smearing_y)),
                'smearing_x': float(smearing_x),
                'smearing_y': float(smearing_y),
                'flux': float(flux),
                'noise': bool(noise),
                'binning_mode': binning_mode,
                'method': 'Tomchuk',
                'nnls_max_rg': float(mean_rg) * (1 + 8 * float(p_val)),
                'radius_samples': int(radius_samples),
                'q_samples': int(q_samples),
            }
            q_sim, i_sim, r_vals, pdf_vals, analysis_res, _ = run_simulation_analysis_case(params)
            summary = build_summary_row(params, analysis_res)
            sanity_summary = build_sanity_summary_row(q_sim, i_sim, r_vals, pdf_vals, analysis_res)
            reconstruction = build_reconstruction_quality_summary(analysis_res)

            p_pdi = float(analysis_res.get('p_rec_pdi', 0))
            p_pdi2 = float(analysis_res.get('p_rec_pdi2', 0))
            abs_err_pdi = abs(p_pdi - float(p_val))
            abs_err_pdi2 = abs(p_pdi2 - float(p_val))
            row = {
                'q_max': float(q_max),
                'n_bins': int(n_bins),
                'p_pdi': p_pdi,
                'p_pdi2': p_pdi2,
                'abs_err_pdi': abs_err_pdi,
                'abs_err_pdi2': abs_err_pdi2,
                'max_abs_err': max(abs_err_pdi, abs_err_pdi2),
                'sum_abs_err': abs_err_pdi + abs_err_pdi2,
                'rel_err_pdi': float(summary.get('Rel_Err_p', 0)),
                'rel_err_pdi2': float(summary.get('Rel_Err_p_PDI2', 0)),
                'rrms_pdi': reconstruction['rrms_pdi'],
                'rrms_pdi2': reconstruction['rrms_pdi2'],
                'quality_pdi': reconstruction['quality_pdi'],
                'quality_pdi2': reconstruction['quality_pdi2'],
                'best_variant': reconstruction['best_variant'],
                'best_rrms': reconstruction['best_rrms'],
                'sanity_pass': bool(sanity_summary.get('Sanity_Pass', False)),
                'sanity_failures': sanity_summary.get('Sanity_Failures', 'none'),
                'tomchuk_extraction': analysis_res.get('tomchuk_extraction', 'none'),
                'meets_target': abs_err_pdi <= target_abs_error and abs_err_pdi2 <= target_abs_error,
            }
            rows.append(row)

    if not rows:
        return {
            'rows': [],
            'best': None,
            'passing_rows': [],
            'safety_zone': None,
            'target_abs_error': target_abs_error,
        }

    rows = sorted(rows, key=lambda row: (row['max_abs_err'], row['sum_abs_err'], row['best_rrms'], row['q_max'], row['n_bins']))
    best = rows[0]
    passing_rows = [row for row in rows if row['meets_target']]
    safety_rows = [
        row for row in rows
        if row['max_abs_err'] <= best['max_abs_err'] + safety_margin
    ]
    safety_zone = None
    if safety_rows:
        safety_zone = {
            'q_max_min': min(row['q_max'] for row in safety_rows),
            'q_max_max': max(row['q_max'] for row in safety_rows),
            'n_bins_min': min(row['n_bins'] for row in safety_rows),
            'n_bins_max': max(row['n_bins'] for row in safety_rows),
            'count': len(safety_rows),
        }

    return {
        'rows': rows,
        'best': best,
        'passing_rows': passing_rows,
        'safety_zone': safety_zone,
        'target_abs_error': target_abs_error,
        'safety_margin': safety_margin,
    }

def run_simulation_analysis_case(params):
    q_sim, i_sim, i_2d, r_vals, pdf_vals = run_simulation_core(params)
    i_for_analysis = i_sim
    if params['mode'] == 'Sphere' and params.get('method') == 'Tomchuk' and params.get('normalize_simulated', True):
        i_for_analysis, norm_scale = normalize_simulated_sphere_intensity(q_sim, i_sim, r_vals, pdf_vals)
    else:
        norm_scale = 1.0
    analysis_res = perform_saxs_analysis(
        q_sim,
        i_for_analysis,
        params['dist_type'],
        params['mean_rg'],
        params['mode'],
        params['method'],
        params['nnls_max_rg'],
        i_2d=i_2d,
        analysis_settings=params,
    )
    analysis_res['simulation_normalization_scale'] = norm_scale
    recovered_dists = build_recovered_distributions(params, analysis_res, r_vals)
    return q_sim, i_for_analysis, r_vals, pdf_vals, analysis_res, recovered_dists
