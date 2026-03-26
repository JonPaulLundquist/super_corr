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

def _parabola_figures(grid, tau, members, lambdas, neg_sigma, SS, result_path):
    sgb = grid.sgb.deg
    # Figure of Mean tau in equal solid angle bins of supergalactic latitude (SGB) for 
    # abs(tau) scan. 
    # popt[0] is the parameter used to test supergalactic structure significance.
    bin_edges = _default_lat_bin_edges(sgb, n_bins=20)
    
    popt_mean, R2_mean, bin_edges = _parabola_fig(sgb, tau, members, 
        result_path+'SuperCorr_MeanTau', ylim=(-1,1), stat='Mean', bin_edges=bin_edges,
        title=(r"$\mathbf{\langle \tau \rangle}$ from Max. $\mathbf{\sigma}$ Scan "
               r"($\mathbf{|\tau| > 0}$)"))
    
    # a is close to paper's test statistic of supergalactic structure of correlations
    # Fit y = a*x^2+c instead of y = a*x^2 + b*x + c to penalize <tau> minimum off 0 SGB
    a_mean, _ = popt_mean
    # Convert to rad^-2 for a larger number for supergalactic structure test statistic.
    print(f"Supergalactic Curvature Test Statistic a={a_mean*(180/np.pi)**2:.3f}") 
    print(f"R²_adj: {R2_mean:3f}")
    
    # Figure of Median tau in equal solid angle bins of supergalactic latitude (sgb) for 
    # abs(tau) scan. This popt[0] parameter could also be used to test significance.
    popt_med, R2_median, _ = _parabola_fig(sgb, tau, members, 
        result_path+'SuperCorr_MedianTau', ylim=(-1,1), stat='Median', 
        bin_edges=bin_edges, title=(r"median($\mathbf{\tau}$) from Max. "
                                    r"$\mathbf{\sigma}$ Scan ($\mathbf{|\tau| > 0}$)"))

    a_med, _ = popt_med
    # Convert to rad^-2 for a larger number for supergalactic structure test statistic.
    print(f"Supergalactic Curvature Test Statistic "
          f"a_median={a_med*(180/np.pi)**2:.3f}") 
    print(f"R²_adj: {R2_median:3f}")

    # Mean Lambda in equal solid angle bins of SGB for wedges given by abs(tau) scan. 
    popt_lambda, R2_lambda, _ = _parabola_fig(sgb, lambdas, members, 
        result_path+'SuperCorr_MeanLambda', ylim=(-1,1), stat='Mean', 
        varname=r'$\mathbf{\Lambda}$', bin_edges=bin_edges, 
        title=(r"$\mathbf{\langle \Lambda \rangle}$ from Max. "
               r"$\mathbf{\sigma}$ Scan ($\mathbf{|\tau| > 0}$)"))

    a_lambda, _ = popt_lambda
    
    # Convert to rad^-2 for a larger number for supergalactic structure test statistic.
    print(f"Supergalactic Curvature Statistic a_lambda={a_lambda*(180/np.pi)**2:.3f}") 
    print(f"R²_adj: {R2_lambda:3f}")

    # Mean Lambda in equal solid angle bins of SGB for wedges given by abs(tau) scan. 
    popt_med_lambda, R2_med_lambda, _ = _parabola_fig(sgb, lambdas, members, 
        result_path+'SuperCorr_MedianLambda', ylim=(-1,1), stat='Median', 
        varname=r'$\mathbf{\Lambda}$', bin_edges=bin_edges, 
        title=(r"median($\mathbf{\Lambda}$) from Max. "
               r"$\mathbf{\sigma}$ Scan ($\mathbf{|\tau| > 0}$)"))

    a_med_lambda, _ = popt_med_lambda
    
    print(f"Supergalactic Curvature Test Statistic "
          f"a_med_lambda={a_med_lambda*(180/np.pi)**2:.3f}") 
    print(f"R²_adj: {R2_med_lambda:3f}")

    # Mean sigma from abs(tau) scan for wedges with tau<0.
    #_, _, _ = _parabola_fig(sgb[tau<0], sigma[tau<0], members[tau<0], 
    #    result_path+'SuperCorr_MeanSigma', stat='Mean', varname=r'$\mathbf{\sigma}$', 
    #    ylim=(3.4,4.6), bin_edges=bin_edges,
    #    title=r"$\mathbf{\langle \sigma \rangle}$ from Max. "
    #          r"$\mathbf{\sigma}$ Scan ($\mathbf{|\tau| > 0}$) for $\mathbf{\tau < 0}$")
    
    # Mean sigma from tau<0 maximum sigma scan. 
    # Also, a supergalactic structure indicator.
    popt_neg_sigma, R2_neg_sigma, _ = _parabola_fig(sgb, neg_sigma, members, 
        result_path+'SuperCorr_MeanNegSigma', ylim=(3.2,4.3), stat='Mean', 
        varname=r'$\mathbf{\sigma}$', bin_edges=bin_edges,
        title=(r"$\mathbf{\langle \sigma \rangle}$ from Max. $\mathbf{\sigma}$ Scan "
               r"($\mathbf{\tau < 0}$)"))

    # Median sigma from tau<0 maximum sigma scan. 
    # Also, a supergalactic structure indicator.
    popt_med_neg_sigma, R2_med_neg_sigma, _ = _parabola_fig(sgb, neg_sigma, members, 
        result_path+'SuperCorr_MedianNegSigma', ylim=(3.2,4.3), stat='Median', 
        varname=r'$\mathbf{\sigma}$', bin_edges=bin_edges,
        title=(r"median($\mathbf{\sigma}$) from Max. $\mathbf{\sigma}$ Scan "
               r"($\mathbf{\tau < 0}$)"))

    # Mean Siegel slope from abs(tau) scan. A strong correlation does not necessarily 
    # correspond to a strong magnetic field (or slope of distance vs 1/E).
    # Also, a supergalactic structure indicator.
    popt_siegel, R2_siegel, _ = _parabola_fig(sgb, SS, members, 
        result_path+'SuperCorr_MeanSiegel', stat='Mean', varname='Siegel Slope', 
        ylim=(-50,30), bin_edges=bin_edges,
        title=r"Mean Siegel Slopes: Distance vs 1/E ($\mathbf{\sigma}$ Scan "
              r"$\mathbf{|\tau| > 0}$)")
    
    # Median Siegel slope from abs(tau) scan. A strong correlation does not necessarily 
    # correspond to a strong magnetic field (or slope of distance vs 1/E).
    # Also, a supergalactic structure indicator.
    popt_med_siegel, R2_med_siegel, _ = _parabola_fig(sgb, SS, members, 
        result_path+'SuperCorr_MedianSiegel', stat='Median', varname='Siegel Slope', 
        ylim=(-30,30), bin_edges=bin_edges,
        title=r"Median Siegel Slopes: Distance vs 1/E ($\mathbf{\sigma}$ Scan "
              r"$\mathbf{|\tau| > 0}$)")

    # Mean tau in equal solid angle bins of Galactic latitude (b) for abs(tau) scan.
    popt_galactic, R2_galactic, _ = _parabola_fig(sgb, tau, members, 
        result_path+'SuperCorr_MeanTau_Galactic', grid=grid, proj='Galactic', 
        ylim=(-1,1),
        title=(r"$\mathbf{\langle \tau \rangle}$ from Max. $\mathbf{\sigma}$ Scan "
               r"($\mathbf{|\tau| > 0}$"))

    a_galactic, _ = popt_galactic
    print(f"Galactic Curvature Statistic a_galactic={a_galactic*(180/np.pi)**2:.3f}") 
    print(f"R²_adj: {R2_galactic:3f}")

    return (popt_mean, R2_mean, popt_med, R2_median, popt_lambda, R2_lambda,
            popt_med_lambda, R2_med_lambda, popt_neg_sigma, R2_neg_sigma,
            popt_med_neg_sigma, R2_med_neg_sigma, popt_siegel, R2_siegel,
            popt_med_siegel, R2_med_siegel, popt_galactic, R2_galactic)


def _parabola_stats(grid, tau, members, lambdas, neg_sigma, SS):
    """Compute parabola-fit statistics without creating any figures."""
    sgb = np.asarray(grid.sgb.deg)
    bin_edges = _default_lat_bin_edges(sgb, n_bins=20)

    popt_mean, R2_mean, *_ = _parabola_fit(
        sgb, tau, members, stat='Mean', bin_edges=bin_edges, method='wls')
    popt_med, R2_median, *_ = _parabola_fit(
        sgb, tau, members, stat='Median', bin_edges=bin_edges, method='wls')
    popt_lambda, R2_lambda, *_ = _parabola_fit(
        sgb, lambdas, members, stat='Mean', bin_edges=bin_edges, method='wls')
    popt_med_lambda, R2_med_lambda, *_ = _parabola_fit(
        sgb, lambdas, members, stat='Median', bin_edges=bin_edges, method='wls')

    popt_neg_sigma, R2_neg_sigma, *_ = _parabola_fit(
        sgb, neg_sigma, members, stat='Mean', bin_edges=bin_edges, method='wls')
    popt_med_neg_sigma, R2_med_neg_sigma, *_ = _parabola_fit(
        sgb, neg_sigma, members, stat='Median', bin_edges=bin_edges, method='wls')

    popt_siegel, R2_siegel, *_ = _parabola_fit(
        sgb, SS, members, stat='Mean', bin_edges=bin_edges, method='wls')
    popt_med_siegel, R2_med_siegel, *_ = _parabola_fit(
        sgb, SS, members, stat='Median', bin_edges=bin_edges, method='wls')

    grid_gal = grid.transform_to('galactic')
    b_gal = np.asarray(grid_gal.b.deg)
    bin_edges_gal = _default_lat_bin_edges(b_gal, n_bins=20)
    popt_galactic, R2_galactic, *_ = _parabola_fit(
        b_gal, tau, members, stat='Mean', bin_edges=bin_edges_gal, method='wls')

    return (popt_mean, R2_mean, popt_med, R2_median, popt_lambda, R2_lambda,
            popt_med_lambda, R2_med_lambda, popt_neg_sigma, R2_neg_sigma,
            popt_med_neg_sigma, R2_med_neg_sigma, popt_siegel, R2_siegel,
            popt_med_siegel, R2_med_siegel, popt_galactic, R2_galactic)


def _default_lat_bin_edges(lat_deg, n_bins=20):
    lat_deg = np.asarray(lat_deg, float)
    lat_deg = lat_deg[np.isfinite(lat_deg)]
    if lat_deg.size == 0:
        return np.linspace(-90, 90, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    return np.percentile(lat_deg, percentiles)

