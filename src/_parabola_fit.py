#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 18:16:18 2025

@author: Jon Paul Lundquist
"""

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import least_squares
from scipy.stats import binned_statistic
from numba import njit, prange

def _parabola_fit(b, y, members, stat='Mean', bin_edges=None, method='rotated'):
    y = np.asarray(y, float)
    b = np.asarray(b, float)

    m = np.isfinite(b) & np.isfinite(y)
    b = b[m]
    y = y[m]
        
    if bin_edges is None:
        percentiles = np.linspace(0, 100, 21)
        bin_edges = np.percentile(b, percentiles)

    n, _, _ = binned_statistic(b, y, statistic="count", bins=bin_edges)
    b_center, _, _ = binned_statistic(b, b, statistic="median",  bins=bin_edges)

    if stat == 'Mean':
        y_stat, _, _ = binned_statistic(b, y, statistic="mean",  bins=bin_edges)
        y_stat2, y_se, n_wedges_bin, n_ind_mean, n_ind_std = _binned_independent_se(
            lat=b, tau=y, members=members, bin_edges=bin_edges,
            stat="mean", n_trials=5000, rng=12345, min_wedges=10)

    elif stat == 'Median':
        y_stat, _, _ = binned_statistic(b, y, statistic="median",  bins=bin_edges)
        y_stat2, y_se, n_wedges_bin, n_ind_mean, n_ind_std = _binned_independent_se(
            lat=b, tau=y, members=members, bin_edges=bin_edges,
            stat="median", n_trials=5000, rng=12345, min_wedges=10)

    else:
        print('Not Implemented')

    good = np.isfinite(b_center) & np.isfinite(y_stat) & (n > 0)

    if method == 'wls':
        popt, R2_adj = _wls_fit(b_center[good], y_stat[good], sigma=y_se[good])
        fit_meta = {'method': 'wls'}

    elif method == 'lad':
        popt, R2_adj = _lad_fit(b_center[good], y_stat[good], sigma=y_se[good])
        fit_meta = {'method': 'lad'}

    elif method == 'bisquare':
        popt, R2_adj = _bisquare_fit(b_center[good], y_stat[good], sigma=y_se[good])
        fit_meta = {'method': 'bisquare'}

    elif method == 'rotated':
        popt, R2_adj, fit_meta = _rotated_fit(b_center[good], y_stat[good], sigma=y_se[good])
        
    else:
        raise ValueError(f"Method {method} not implemented")

    return popt, R2_adj, b_center, y_stat, y_se, fit_meta


# Estimate the uncertainty of the statistic using independent set resampling.
@njit(cache=True)
def _independent_stat(y, members, order, stat="mean"):
    """
    Build one maximal set of event-disjoint wedges using a given order.

    Parameters
    ----------
    y : (n_wedges,) array
    members : (n_wedges, n_events) bool array
        members[i, j] = True if event j is in wedge i
    stat : {"mean", "median"}
    order : permutation of wedge indices, or None

    Returns
    -------
    value : float
        Mean/median of the chosen independent tau values
    n_chosen : int
        Number of independent wedges selected
    """

    n_wedges, n_events = members.shape
    used = np.zeros(n_events, dtype=np.bool_)
    chosen = np.empty(n_wedges, dtype=np.int64)
    n_chosen = 0

    for idx in order:
        m = members[idx]
        overlap = False

        # Numba generally handles explicit boolean loops better than masked indexing.
        for k in range(n_events):
            if m[k] and used[k]:
                overlap = True
                break

        if not overlap:
            chosen[n_chosen] = idx
            n_chosen += 1
            for k in range(n_events):
                if m[k]:
                    used[k] = True

    vals = y[chosen[:n_chosen]]
    if stat == "mean":
        return np.mean(vals), n_chosen
    elif stat == "median":
        return np.median(vals), n_chosen
    else:
        raise ValueError("stat must be 'mean' or 'median'")


@njit(cache=True)
def _std_ddof1(x):
    # Numba version of numpy.std(ddof=1) as it is not supported by numba.
    n = x.size
    if n <= 1:
        return np.nan
    mean = np.mean(x)
    sse = 0.0
    for i in range(n):
        diff = x[i] - mean
        sse += diff * diff
    return np.sqrt(sse / (n - 1))


@njit(cache=True)
def _independent_se(y, members, stat="mean", n_trials=1000, min_wedges=5, rng=-1):
    """
    Random-order greedy independent-set resampling for one latitude bin.

    Returns
    -------
    stat_est : float
        Mean of the trial statistics
    se : float
        Standard deviation across trials
    stats : (n_trials,) array
        Trial statistics
    n_ind : (n_trials,) array
        Number of independent wedges selected in each trial
    """
    if rng >= 0:
        np.random.seed(rng)
    n = len(y)
    
    if n == 0:
        return np.nan, np.nan, np.empty(0), np.empty(0, dtype=np.int64)

    stats = np.empty(n_trials, dtype=np.float64)
    n_ind = np.empty(n_trials, dtype=np.int64)
    
    k = 0
    j = 0
    i = 0
    while i < n_trials:
        order = np.random.permutation(n)
        p_stat, p_n = _independent_stat(y, members, order, stat=stat)
        
        if p_n >= min_wedges:
            stats[i] = p_stat
            n_ind[i] = p_n
            i += 1
            k += 1
        j += 1
        if (j != 0 and j % (2 * n_trials) == 0) and (k / j < 0.003):
            min_wedges -= 1
            k = 0
            j = 0
    
    stat_est = np.mean(stats) #diagnostic only
    se = _std_ddof1(stats) # standard deviation of the statistic across trials
    return stat_est, se, stats, n_ind


@njit(cache=True, parallel=True)
def _binned_independent_se(lat, tau, members, bin_edges, stat="mean", n_trials=1000,
                          min_wedges=5, rng=12345):
    """
    Apply independent_set_se separately inside each latitude bin.

    Parameters
    ----------
    lat : (n_wedges,) array
        Latitude of each wedge/grid point, e.g. grid.sgb.deg
    tau : (n_wedges,) array
        Tau value for each wedge/grid point
    members : (n_wedges, n_events) bool array
        Event-membership matrix
    bin_edges : (n_bins+1,) array
        Latitude bin edges
    stat : {"mean", "median"}
        Statistic of the independent tau values
    n_trials : int
        Number of random orderings per bin
    rng : None, int, or Generator
    min_wedges : int
        Minimum number of wedges required in a bin per trial

    Returns
    -------
    y_stat : (n_bins,) array
        Bin statistic estimated from random independent subsets
    y_se : (n_bins,) array
        Standard deviation across trials in each bin
    n_wedges_bin : (n_bins,) int array
        Number of wedges in each latitude bin
    n_ind_mean : (n_bins,) array
        Mean number of independent wedges selected across trials
    n_ind_std : (n_bins,) array
        Std of number of independent wedges selected across trials
    """

    nb = len(bin_edges) - 1
    n_wedges = len(tau)
    n_events = members.shape[1]
    
    y_stat = np.full(nb, np.nan, dtype=np.float64)
    y_se = np.full(nb, np.nan, dtype=np.float64)
    n_wedges_bin = np.zeros(nb, dtype=np.int64)
    n_ind_mean = np.full(nb, np.nan, dtype=np.float64)
    n_ind_std = np.full(nb, np.nan, dtype=np.float64)

    good = np.isfinite(lat) & np.isfinite(tau)

    for j in prange(nb):
        lo = bin_edges[j]
        hi = bin_edges[j + 1]

        count = 0
        if j < nb - 1:
            for i in range(n_wedges):
                if good[i] and lat[i] >= lo and lat[i] < hi:
                    count += 1
        else:
            for i in range(n_wedges):
                if good[i] and lat[i] >= lo and lat[i] <= hi:
                    count += 1

        n_wedges_bin[j] = count
        if count == 0:
            continue

        tau_bin = np.empty(count, dtype=np.float64)
        members_bin = np.empty((count, n_events), dtype=np.bool_)

        k = 0
        if j < nb - 1:
            for i in range(n_wedges):
                if good[i] and lat[i] >= lo and lat[i] < hi:
                    tau_bin[k] = tau[i]
                    members_bin[k, :] = members[i, :]
                    k += 1
        else:
            for i in range(n_wedges):
                if good[i] and lat[i] >= lo and lat[i] <= hi:
                    tau_bin[k] = tau[i]
                    members_bin[k, :] = members[i, :]
                    k += 1

        bin_seed = -1
        if rng >= 0:
            bin_seed = rng + j

        stat_est, se_est, stats, n_ind = _independent_se(
            tau_bin, members_bin, stat=stat, n_trials=n_trials,
            min_wedges=min_wedges, rng=bin_seed)

        y_stat[j] = stat_est
        y_se[j] = se_est
        n_ind_mean[j] = np.mean(n_ind)
        n_ind_std[j] = _std_ddof1(n_ind)

    return y_stat, y_se, n_wedges_bin, n_ind_mean, n_ind_std


def _parabola(x, a, c):
    return a * x**2 + c

# def parabola2(x, a, b, c):
#     return a * x**2 + b * x + c

def _quad_design(x):
    x = np.asarray(x, dtype=np.float64)
    return np.column_stack((x*x, np.ones_like(x)))

# def quad_design2(x):
#     x = np.asarray(x, dtype=np.float64)
#     return np.column_stack((x*x, x, np.ones_like(x)))


def _lad_fit(x, y, sigma=None):
    """
    Exact weighted least-absolute-deviation fit (global optimum) via LP.

    Objective: minimize sum_i w_i * t_i
    s.t.  -t_i <= y_i - X_i beta <= t_i,  t_i >= 0
    with weights w_i = 1 if sigma is None else 1/sigma_i.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = _quad_design(x)
    n = y.size

    if sigma is None:
        w = np.ones(n, dtype=np.float64)
        m = np.isfinite(y) & np.isfinite(x)
    else:
        sigma = np.asarray(sigma, dtype=np.float64)
        m = np.isfinite(y) & np.isfinite(x) & np.isfinite(sigma) & (sigma > 0)
        w = 1.0 / sigma[m]

    X = X[m]
    y = y[m]
    w = w[:y.size] if w.size != y.size else w  # safety
    n = y.size

    # Variables: [a, b, c, t0..t_{n-1}]
    # Minimize: sum w_i * t_i
    c_obj = np.concatenate([np.zeros(3), w])

    # Constraints:
    #  y - Xβ <= t   ->  -Xβ - t <= -y
    # -y + Xβ <= t   ->   Xβ - t <=  y
    A_ub = np.zeros((2*n, 3+n), dtype=np.float64)
    b_ub = np.zeros(2*n, dtype=np.float64)

    # -Xβ - t <= -y
    A_ub[:n, :3] = -X
    A_ub[:n, 3:] = -np.eye(n)
    b_ub[:n] = -y

    #  Xβ - t <=  y
    A_ub[n:, :3] = X
    A_ub[n:, 3:] = -np.eye(n)
    b_ub[n:] = y

    bounds = [(None, None), (None, None), (None, None)] + [(0, None)]*n

    res = linprog(
        c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
    )
    if not res.success:
        raise RuntimeError(f"LAD linprog failed: {res.message}")

    a, c = res.x[:3]
    popt = np.array([a, c], dtype=np.float64)
    # 3) compute predictions & uniform weights
    y_fit = _parabola(x, *popt)
    w = np.ones_like(y)
    
    # 4) compute R2_adj
    R2_adj = _weighted_adj_r2(y, y_fit, w, p=len(popt))
    
    return popt, R2_adj

# Fit y = a*x^2 + b*x + c using WLS
# def wls_fit(x, y, sigma=None):
#     x = np.asarray(x, np.float64)
#     y = np.asarray(y, np.float64)

#     mask = np.isfinite(x) & np.isfinite(y)
#     if sigma is not None:
#         sigma = np.asarray(sigma, np.float64)
#         mask &= np.isfinite(sigma) & (sigma > 0)

#     x = x[mask]
#     y = y[mask]
#     if sigma is not None:
#         sigma = sigma[mask]
#         w = 1.0 / sigma**2
#     else:
#         w = np.ones_like(y)

#     x0 = np.mean(x)
#     xscale = np.max(np.abs(x - x0))
    
#     x_s = (x - x0) / xscale

#     X_s = quad_design(x_s)

#     sw = np.sqrt(w)
#     Xw = X_s * sw[:, None]
#     yw = y * sw

#     popt_s, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

#     A, B, C = popt_s

#     # Unscale back to y = a x^2 + b x + c
#     a = A / xscale**2
#     b = B / xscale - 2.0 * A * x0 / xscale**2
#     c = C - B * x0 / xscale + A * x0**2 / xscale**2

#     popt = np.array([a, b, c], dtype=np.float64)

#     y_fit = X_s @ popt_s
#     R2_adj = weighted_adj_r2(y, y_fit, w, p=3)

#     return popt, R2_adj

def _wls_fit(x, y, sigma=None):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        sigma = np.asarray(sigma, np.float64)
        mask &= np.isfinite(sigma) & (sigma > 0)

    x = x[mask]
    y = y[mask]
    if sigma is not None:
        sigma = sigma[mask]
        w = 1.0 / sigma**2
    else:
        w = np.ones_like(y)

    xscale = np.max(np.abs(x))

    x_s = x / xscale
    X_s = _quad_design(x_s)

    sw = np.sqrt(w)
    popt_s, *_ = np.linalg.lstsq(X_s * sw[:, None], y * sw, rcond=None)

    A, c = popt_s
    a = A / xscale**2

    popt = np.array([a, c], dtype=np.float64)

    y_fit = X_s @ popt_s
    R2_adj = _weighted_adj_r2(y, y_fit, w, p=2)

    return popt, R2_adj


def _weighted_adj_r2(y, y_pred, w, p):
    mask = np.isfinite(y) & np.isfinite(y_pred) & np.isfinite(w) & (w > 0)
    y, y_pred, w = y[mask], y_pred[mask], w[mask]

    if y.size == 0:
        return np.nan

    W = w.sum()
    if W <= 0:
        return np.nan

    y_bar = np.dot(w, y) / W

    ss_tot = np.dot(w, (y - y_bar)**2)
    ss_res = np.dot(w, (y - y_pred)**2)

    if np.isclose(ss_tot, 0.0):
        return np.nan   # or 0.0 / 1.0 by convention, but nan is cleaner

    R2 = 1 - ss_res / ss_tot

    n = len(y)
    if n <= p:
        return np.nan   # adjusted R^2 undefined

    R2_adj = 1 - (1 - R2) * (n - 1) / (n - p)
    return R2_adj


def _continuous_branch_mask(x, t1, t2, y1, y2):
    """Pick a smooth branch across sorted x for plotting predictions."""
    if x.size <= 1:
        return np.abs(t1) <= np.abs(t2)

    order = np.argsort(x)
    t1o, t2o = t1[order], t2[order]
    y1o, y2o = y1[order], y2[order]
    xo = x[order]

    use1o = np.zeros_like(xo, dtype=bool)
    k0 = int(np.argmin(np.abs(xo)))
    use1o[k0] = np.abs(t1o[k0]) <= np.abs(t2o[k0])
    yprev = y1o[k0] if use1o[k0] else y2o[k0]

    for k in range(k0 + 1, xo.size):
        use1o[k] = abs(y1o[k] - yprev) <= abs(y2o[k] - yprev)
        yprev = y1o[k] if use1o[k] else y2o[k]

    yprev = y1o[k0] if use1o[k0] else y2o[k0]
    for k in range(k0 - 1, -1, -1):
        use1o[k] = abs(y1o[k] - yprev) <= abs(y2o[k] - yprev)
        yprev = y1o[k] if use1o[k] else y2o[k]

    use1 = np.zeros_like(use1o)
    use1[order] = use1o
    return use1


def _rotated_predict_y(x, params, y_ref=None):
    """
    Predict y(x) for rotated parabola in raw (x, y) coordinates.

    Parametric model around vertex (0, y0):
      x = cos(theta)*t - sin(theta)*a*t^2
      y = y0 + sin(theta)*t + cos(theta)*a*t^2
    """
    a, y0, theta = params
    x = np.asarray(x, dtype=np.float64)
    ct, st = np.cos(theta), np.sin(theta)
    eps = 1e-12

    if abs(st) < eps:
        return y0 + a * x**2

    # Solve (a*sin(theta)) t^2 - cos(theta) t + x = 0
    A = a * st
    B = -ct
    C = x

    y_out = np.full_like(x, np.nan, dtype=np.float64)
    if abs(A) < eps:
        # Near-linear case in t.
        t = -C / (B if abs(B) > eps else 1.0)
        y_out[:] = y0 + st * t + ct * a * t*t
        return y_out

    disc = B*B - 4.0*A*C
    ok = disc >= 0.0
    if not np.any(ok):
        return y_out

    sq = np.sqrt(np.maximum(disc[ok], 0.0))
    den = 2.0 * A

    t1 = (-B + sq) / den
    t2 = (-B - sq) / den

    y1 = y0 + st * t1 + ct * a * t1*t1
    y2 = y0 + st * t2 + ct * a * t2*t2

    if y_ref is None:
        use1 = _continuous_branch_mask(x[ok], t1, t2, y1, y2)
    else:
        y_ref_ok = np.asarray(y_ref, dtype=np.float64)[ok]
        use1 = np.abs(y1 - y_ref_ok) <= np.abs(y2 - y_ref_ok)

    y_out[ok] = np.where(use1, y1, y2)
    return y_out


def _rotated_residuals(params, x, y, sigma):
    y_pred = _rotated_predict_y(x, params, y_ref=y)
    # Penalize unreachable x values (no real branch).
    bad = ~np.isfinite(y_pred)
    resid = (y - y_pred) / sigma
    resid[bad] = 1e6
    return resid


def _rotated_fit(x, y, sigma=None):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        sigma = np.asarray(sigma, np.float64)
        mask &= np.isfinite(sigma) & (sigma > 0)

    x = x[mask]
    y = y[mask]
    if sigma is not None:
        sigma = sigma[mask]
        w = 1.0 / sigma**2
    else:
        w = np.ones_like(y)
        sigma = np.ones_like(y)

    if x.size < 3:
        raise ValueError("Rotated fit requires at least 3 valid binned points.")

    # Initial guess from weighted centered quadratic y = a*x^2 + y0.
    X0 = np.column_stack((x*x, np.ones_like(x)))
    sw = np.sqrt(w)
    beta0, *_ = np.linalg.lstsq(X0 * sw[:, None], y * sw, rcond=None)
    a0 = float(beta0[0])
    y0_guess = float(beta0[1])
    p0 = np.array([a0, y0_guess, 0.0], dtype=np.float64)

    yspan = max(np.nanmax(y) - np.nanmin(y), 1e-3)
    xspan = max(np.nanmax(x) - np.nanmin(x), 1.0)
    a_scale = yspan / (xspan * xspan)
    alim = max(1e-6, 20.0 * max(abs(a0), a_scale))
    bounds_lo = np.array([-alim, np.nanmin(y) - yspan, -np.deg2rad(70.0)])
    bounds_hi = np.array([ alim, np.nanmax(y) + yspan,  np.deg2rad(70.0)])
    res = least_squares(
        _rotated_residuals, p0, args=(x, y, sigma),
        bounds=(bounds_lo, bounds_hi), method='trf',
        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=20000
    )
    p = res.x

    yhat = _rotated_predict_y(x, p, y_ref=y)
    R2_adj = _weighted_adj_r2(y, yhat, w, p=3)

    a = p[0]
    y0 = p[1]
    popt = np.array([a, y0], dtype=np.float64)
    fit_meta = {
        'method': 'rotated',
        'params': p,
        'theta_rad': float(p[2]),
        'theta_deg': float(np.rad2deg(p[2])),
        'y0': float(y0),
    }
    return popt, R2_adj, fit_meta


def _tukey_bisquare_weights(u):
    """Expects u = r / (c * s)"""
    w = np.zeros_like(u)
    m = np.abs(u) < 1
    w[m] = (1 - u[m]**2)**2
    return w


def _leverage_diag(X, w):
    w = np.asarray(w, float)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    Q, R = np.linalg.qr(Xw, mode='reduced')
    h = np.sum(Q**2, axis=1)
    return np.clip(h, 0.0, 1.0 - 1e-12)


def _bisquare_fit(x, y, sigma=None, tuning_c=4.685, use_leverage=True,
                 max_iter=200, tol=1e-12):

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        sigma = np.asarray(sigma, float)
        m &= np.isfinite(sigma) & (sigma > 0)

    x = x[m]
    y = y[m]
    if sigma is not None:
        sigma = sigma[m]
        w_prior = 1.0 / sigma**2
    else:
        w_prior = np.ones_like(y)

    xscale = np.max(np.abs(x))

    x_s = x / xscale
    X = _quad_design(x_s)

    sw = np.sqrt(w_prior)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)

    w = w_prior.copy()
    for _ in range(max_iter):
        r = y - (X @ beta)
        r_scaled = r / sigma if sigma is not None else r

        s = 1.4826 * np.median(np.abs(r_scaled - np.median(r_scaled)))
        s = max(s, 1e-12)

        if use_leverage:
            h = _leverage_diag(X, w)
            denom = tuning_c * s * np.sqrt(np.maximum(1.0 - h, 1e-12))
        else:
            denom = tuning_c * s

        u = r_scaled / denom
        w_rob = _tukey_bisquare_weights(u)
        w_new = w_prior * w_rob

        if np.all(w_new <= 0):
            break

        sw = np.sqrt(w_new)
        beta_new, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)

        if np.allclose(beta, beta_new, atol=tol, rtol=tol):
            beta = beta_new
            w = w_new
            break

        beta = beta_new
        w = w_new

    y_fit = X @ beta
    R2_adj = _weighted_adj_r2(y, y_fit, w_prior, p=2)

    beta[0] /= xscale**2   # a
    # beta[1] is c unchanged

    return beta, R2_adj
