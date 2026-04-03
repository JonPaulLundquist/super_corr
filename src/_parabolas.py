#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 12:21:37 2026

@author: Jon Paul Lundquist
"""

import numpy as np

from _parabola_fig import _parabola_fig
from _parabola_fit import _parabola_fit

# --- Parabola figures and Test Statistic Calculations ---

def _parabola_figures(grid, corr, members, corr2, neg_sigma, SS, result_path,
                      stat='tau', fit_method='rotated', show=True):
    stat = str(stat).strip().lower()
    if stat == 'tau':
        primary_symbol = r'\tau'
        secondary_symbol = r'\Lambda'
        primary_file = 'Tau'
        secondary_file = 'Lambda'
        secondary_name = 'lambda'
    elif stat == 'lambda':
        primary_symbol = r'\Lambda'
        secondary_symbol = r'\tau'
        primary_file = 'Lambda'
        secondary_file = 'Tau'
        secondary_name = 'tau'
    else:
        raise ValueError(f"Invalid statistic: {stat}")

    sgb = grid.sgb.deg
    # Figure of Mean tau/Lambda in equal solid angle bins of supergalactic latitude 
    # (SGB) for abs(tau/Lambda) scan. 
    # popt_mean[0] is the parameter used to test supergalactic structure significance.
    bin_edges = _default_lat_bin_edges(sgb, n_bins=20)
    
    popt_mean, R2_mean, bin_edges = _parabola_fig(
        sgb, corr, members, result_path+f'SuperCorr_Mean{primary_file}',
        ylim=(-1,1), stat='Mean', varname=rf'$\mathbf{{{primary_symbol}}}$',
        bin_edges=bin_edges, fit_method=fit_method, show=show,
        title=(rf"$\mathbf{{\langle {primary_symbol} \rangle}}$ from Max. "
               rf"$\mathbf{{\sigma}}$ Scan "
               rf"($\mathbf{{|{primary_symbol}| > 0}}$)"))
    
    # a is close to paper's test statistic of supergalactic structure of correlations
    # Fit y = a*x^2+c instead of y = a*x^2 + b*x + c to penalize <tau/Lambda> minimum 
    # off 0 SGB
    a_mean, _ = popt_mean
    # Convert to rad^-2 for a larger number for supergalactic structure test statistic.
    print(f"Supergalactic Curvature Test Statistic a={a_mean*(180/np.pi)**2:.3f}") 
    print(f"R²_adj: {R2_mean:3f}")
    
    # Figure of Median tau/Lambda in equal solid angle bins of supergalactic latitude 
    # (SGB) for abs(tau/Lambda) scan.
    popt_med, R2_median, _ = _parabola_fig(
        sgb, corr, members, result_path+f'SuperCorr_Median{primary_file}',
        ylim=(-1,1), stat='Median', varname=rf'$\mathbf{{{primary_symbol}}}$',
        bin_edges=bin_edges, fit_method=fit_method, show=show,
        title=(rf"median($\mathbf{{{primary_symbol}}}$) from Max. "
               rf"$\mathbf{{\sigma}}$ Scan ($\mathbf{{|{primary_symbol}| > 0}}$)"))

    a_med, _ = popt_med
    # Convert to rad^-2 for a larger number for supergalactic structure test statistic.
    print(f"Supergalactic Curvature Statistic "
          f"a_median={a_med*(180/np.pi)**2:.3f}") 
    print(f"R²_adj: {R2_median:3f}")

    # Mean secondary statistic in equal solid angle bins of SGB for wedges given by 
    # primary statistic abs(tau/Lambda) scan.
    popt_corr2, R2_corr2, _ = _parabola_fig(
        sgb, corr2, members, result_path+f'SuperCorr_Mean{secondary_file}',
        ylim=(-1,1), stat='Mean', varname=rf'$\mathbf{{{secondary_symbol}}}$',
        bin_edges=bin_edges, fit_method=fit_method, show=show,
        title=(rf"$\mathbf{{\langle {secondary_symbol} \rangle}}$ from Max. "
               rf"$\mathbf{{\sigma}}$ Scan ($\mathbf{{|{primary_symbol}| > 0}}$)"))

    a_corr2, _ = popt_corr2
    
    # Convert to rad^-2 for a larger number for supergalactic structure test statistic.
    print(f"Supergalactic Curvature Statistic "
          f"a_{secondary_name}={a_corr2*(180/np.pi)**2:.3f}")
    print(f"R²_adj: {R2_corr2:3f}")

    # Median secondary statistic in equal solid angle bins of SGB for wedges given by 
    # primary statistic abs(tau/Lambda) scan.
    popt_med_corr2, R2_med_corr2, _ = _parabola_fig(
        sgb, corr2, members, result_path+f'SuperCorr_Median{secondary_file}',
        ylim=(-1,1), stat='Median', varname=rf'$\mathbf{{{secondary_symbol}}}$',
        bin_edges=bin_edges, fit_method=fit_method, show=show,
        title=(rf"median($\mathbf{{{secondary_symbol}}}$) from Max. "
               rf"$\mathbf{{\sigma}}$ Scan ($\mathbf{{|{primary_symbol}| > 0}}$)"))

    a_med_corr2, _ = popt_med_corr2
    
    print(f"Supergalactic Curvature Statistic "
          f"a_med_{secondary_name}={a_med_corr2*(180/np.pi)**2:.3f}")
    print(f"R²_adj: {R2_med_corr2:3f}")
    
    # Mean sigma from tau/Lambda<0 maximum sigma scan. 
    # Also, a supergalactic structure indicator.
    popt_neg_sigma, R2_neg_sigma, _ = _parabola_fig(sgb, neg_sigma, members, 
        result_path+'SuperCorr_MeanNegSigma', ylim='sigma', stat='Mean', 
        varname=r'$\mathbf{\sigma}$', bin_edges=bin_edges, fit_method=fit_method,
        show=show,
        title=(r"$\mathbf{\langle \sigma \rangle}$ from Max. $\mathbf{\sigma}$ Scan "
               rf"($\mathbf{{{primary_symbol} < 0}}$)"))

    # Median sigma from tau/Lambda<0 maximum sigma scan. 
    # Also, a supergalactic structure indicator.
    popt_med_neg_sigma, R2_med_neg_sigma, _ = _parabola_fig(sgb, neg_sigma, members, 
        result_path+'SuperCorr_MedianNegSigma', ylim='sigma', stat='Median', 
        varname=r'$\mathbf{\sigma}$', bin_edges=bin_edges, fit_method=fit_method,
        show=show,
        title=(r"median($\mathbf{\sigma}$) from Max. $\mathbf{\sigma}$ Scan "
               rf"($\mathbf{{{primary_symbol} < 0}}$)"))

    # Mean Siegel slope from abs(tau/Lambda) scan. A strong correlation does not 
    # necessarily correspond to a strong magnetic field (or slope of distance vs 1/E).
    # Also, a supergalactic structure indicator.
    popt_siegel, R2_siegel, _ = _parabola_fig(sgb, SS, members, 
        result_path+'SuperCorr_MeanSiegel', stat='Mean', varname='Siegel Slope', 
        ylim='siegel', bin_edges=bin_edges, fit_method=fit_method, show=show,
        title=r"Mean Siegel Slopes: Distance vs 1/E ($\mathbf{\sigma}$ Scan "
              rf"$\mathbf{{|{primary_symbol}| > 0}}$)")
    
    # Median Siegel slope from abs(tau/Lambda) scan. A strong correlation does not 
    # necessarily correspond to a strong magnetic field (or slope of distance vs 1/E).
    # Also, a supergalactic structure indicator.
    popt_med_siegel, R2_med_siegel, _ = _parabola_fig(sgb, SS, members, 
        result_path+'SuperCorr_MedianSiegel', stat='Median', varname='Siegel Slope',
        ylim='siegel', bin_edges=bin_edges, fit_method=fit_method, show=show,
        title=r"Median Siegel Slopes: Distance vs 1/E ($\mathbf{\sigma}$ Scan "
              rf"$\mathbf{{|{primary_symbol}| > 0}}$)")  

    # Mean primary statistic in equal solid angle bins of Galactic latitude (b) for 
    # abs(tau/Lambda) scan.
    popt_galactic, R2_galactic, _ = _parabola_fig(
        sgb, corr, members, result_path+f'SuperCorr_Mean{primary_file}_Galactic',
        grid=grid, proj='Galactic', ylim=(-1,1),
        varname=rf'$\mathbf{{{primary_symbol}}}$',
        fit_method=fit_method, show=show,
        title=(rf"$\mathbf{{\langle {primary_symbol} \rangle}}$ from Max. "
               rf"$\mathbf{{\sigma}}$ Scan "
               rf"($\mathbf{{|{primary_symbol}| > 0}}$)"))

    a_galactic, _ = popt_galactic
    print(f"Galactic Curvature Statistic a_galactic={a_galactic*(180/np.pi)**2:.3f}") 
    print(f"R²_adj: {R2_galactic:3f}")

    return (popt_mean, R2_mean, popt_med, R2_median, popt_corr2, R2_corr2,
            popt_med_corr2, R2_med_corr2, popt_neg_sigma, R2_neg_sigma,
            popt_med_neg_sigma, R2_med_neg_sigma, popt_siegel, R2_siegel,
            popt_med_siegel, R2_med_siegel, popt_galactic, R2_galactic)


def _parabola_stats(grid, corr, members, corr2, neg_sigma, SS, fit_method='rotated'):
    """Compute parabola-fit statistics without creating any figures."""
    sgb = np.asarray(grid.sgb.deg)
    bin_edges = _default_lat_bin_edges(sgb, n_bins=20)

    popt_mean, R2_mean, *_ = _parabola_fit(
        sgb, corr, members, stat='Mean', bin_edges=bin_edges, method=fit_method)
    popt_med, R2_median, *_ = _parabola_fit(
        sgb, corr, members, stat='Median', bin_edges=bin_edges, method=fit_method)
    popt_corr2, R2_corr2, *_ = _parabola_fit(
        sgb, corr2, members, stat='Mean', bin_edges=bin_edges, method=fit_method)
    popt_med_corr2, R2_med_corr2, *_ = _parabola_fit(
        sgb, corr2, members, stat='Median', bin_edges=bin_edges, method=fit_method)

    popt_neg_sigma, R2_neg_sigma, *_ = _parabola_fit(
        sgb, neg_sigma, members, stat='Mean', bin_edges=bin_edges, method=fit_method)
    popt_med_neg_sigma, R2_med_neg_sigma, *_ = _parabola_fit(
        sgb, neg_sigma, members, stat='Median', bin_edges=bin_edges, method=fit_method)

    popt_siegel, R2_siegel, *_ = _parabola_fit(
        sgb, SS, members, stat='Mean', bin_edges=bin_edges, method=fit_method)
    popt_med_siegel, R2_med_siegel, *_ = _parabola_fit(
        sgb, SS, members, stat='Median', bin_edges=bin_edges, method=fit_method)

    grid_gal = grid.transform_to('galactic')
    b_gal = np.asarray(grid_gal.b.deg)
    bin_edges_gal = _default_lat_bin_edges(b_gal, n_bins=20)
    popt_galactic, R2_galactic, *_ = _parabola_fit(
        b_gal, corr, members, stat='Mean', bin_edges=bin_edges_gal,
        method=fit_method)

    return (popt_mean, R2_mean, popt_med, R2_median, popt_corr2, R2_corr2,
            popt_med_corr2, R2_med_corr2, popt_neg_sigma, R2_neg_sigma,
            popt_med_neg_sigma, R2_med_neg_sigma, popt_siegel, R2_siegel,
            popt_med_siegel, R2_med_siegel, popt_galactic, R2_galactic)


def _default_lat_bin_edges(lat_deg, n_bins=20):
    lat_deg = np.asarray(lat_deg, float)
    lat_deg = lat_deg[np.isfinite(lat_deg)]
    if lat_deg.size == 0:
        return np.linspace(-90, 90, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    return np.percentile(lat_deg, percentiles)

