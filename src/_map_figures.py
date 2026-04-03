#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 12:21:37 2026

@author: Jon Paul Lundquist
"""

import numpy as np

from _wedge import _wedge
from _map_fig import _map_supergalactic

# --- Hammer-projection figures ---

def _map_figures(grid, ipix, nside, corr, corr2, sigma, neg_sigma, SS, neg_SS, Dir, 
                 neg_Dir, result_path, stat='tau', show=True):
    stat = str(stat).strip().lower()
    if stat == 'tau':
        primary_name = "Kendall's"
        primary_symbol = r'\tau'
        primary_file = 'Tau'
        primary_cmap = 'tau'
        secondary_name = "Lundquist's"
        secondary_symbol = r'\Lambda'
        secondary_file = 'Lambda'
        secondary_cmap = 'lambda'
    elif stat == 'lambda':
        primary_name = "Lundquist's"
        primary_symbol = r'\Lambda'
        primary_file = 'Lambda'
        primary_cmap = 'lambda'
        secondary_name = "Kendall's"
        secondary_symbol = r'\tau'
        secondary_file = 'Tau'
        secondary_cmap = 'tau'
    else:
        raise ValueError(f"Invalid statistic: {stat}")

    sgl = grid.sgl
    sgb = grid.sgb
    # Hammer-projection figure of tau/Lambda from abs(tau/Lambda) maximum significance 
    # scan
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside, 
                       c_title=rf"{primary_name} $\mathbf{{{primary_symbol}}}$",
                       c=corr, cmap=primary_cmap,
                       file=result_path+f'SuperCorr_{primary_file}Map.png',
                       proj='supergalactic',
                       savefig=1,
                       show=show,
                       title=rf"$\mathbf{{{primary_symbol}}}$ from Max. "
                             rf"$\mathbf{{\sigma}}$ Scan "
                             rf"($\mathbf{{|{primary_symbol}| > 0}}$)")
    
    # Hammer-proj. of secondary statistic tau/Lambda in wedges from primary statistic 
    # abs(Lambda/tau) maximum sigma scan
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside,
                       c_title=rf"{secondary_name} $\mathbf{{{secondary_symbol}}}$",
                       c=corr2, cmap=secondary_cmap,
                       file=result_path+f'SuperCorr_{secondary_file}Map.png',
                       proj='supergalactic',
                       savefig=1,
                       show=show,
                       title=rf"$\mathbf{{{secondary_symbol}}}$ from Max. "
                             rf"$\mathbf{{\sigma}}$ Scan "
                             rf"($\mathbf{{|{primary_symbol}| > 0}}$)")

    # Hammer-proj. of tau/Lambda sigma significance from abs(tau/Lambda) scan
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside,
                       c_title=r'$\mathbf{\sigma}$ (Local Significance)', c=sigma, 
                       cmap='sigma', file=result_path+'SuperCorr_SigmaMap.png', 
                       proj='supergalactic', savefig=1, show=show,
                       title=r"$\mathbf{\sigma}$ from Max. $\mathbf{\sigma}$ Scan "
                             rf"($\mathbf{{|{primary_symbol}| > 0}}$)")
    
    # Hammer-proj. of tau/Lambda sigma significance from abs(tau/Lambda) scan for 
    # points with tau/Lambda<0.
    _map_supergalactic(sgl[corr<0], sgb[corr<0], ipix=ipix[corr<0], nside=nside,
                       c_title=r'$\mathbf{\sigma}$ (Local Significance)', c=sigma[corr<0], 
                       cmap='sigma',
                       file=result_path+f'SuperCorr_SigmaMap_Neg{primary_file}.png',
                       proj='supergalactic', savefig=1, show=show,
                       title=r"$\mathbf{\sigma}$ from Max. $\mathbf{\sigma}$ Scan "
                             rf"($\mathbf{{|{primary_symbol}| > 0}}$) for "
                             rf"$\mathbf{{{primary_symbol} < 0}}$")
    
    # Hammer-proj. of tau/Lambda sigma significance from tau/Lambda<0 maximum sigma scan
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside,
                       c_title=r'$\mathbf{\sigma}$ (Local Significance)', c=neg_sigma, 
                       cmap='sigma', file=result_path+'SuperCorr_NegSigmaMap.png', 
                       proj='supergalactic', savefig=1, show=show,
                       title=r"$\mathbf{\sigma}$ from Max. $\mathbf{\sigma}$ Scan "
                             rf"($\mathbf{{{primary_symbol} < 0}}$)")

    # Hammer-proj. of Siegel slopes from abs(tau/Lambda) maximum significance scan.
    # A strong correlation does not necessarily correspond to a strong magnetic field 
    # (or slope of distance vs 1/E).
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside, c_title=r'Siegel Slope', c=SS, 
                       cmap='tau', file=result_path+'SuperCorr_SiegelMap.png', 
                       proj='supergalactic', savefig=1, show=show,
                       title=r"Siegel Slopes: Distance vs 1/E ($\mathbf{\sigma}$ Scan "
                             rf"$\mathbf{{|{primary_symbol}| > 0}}$)")
    
    # Hammer-proj. of magnetic field from abs(tau/Lambda) maximum significance scan
    # Only include points with positive Siegel slopes and negative tau
    SS_ind = (SS>0) & (corr<0)
    _map_supergalactic(sgl[SS_ind], sgb[SS_ind],
                       c_title=r'B$\cdot$S$\cdot$Z  [nG$\cdot$Mpc]', c=SS[SS_ind], 
                       cmap='plasma', file=result_path+'SuperCorr_FieldMap.png', 
                       proj='supergalactic', savefig=1, arrows=True, dirs=Dir[SS_ind], 
                       arrow_len=np.exp(sigma[SS_ind]-min(sigma[SS_ind]))/1.5,
                       show=show,
                       title=r"Magnetic Field Map from Multiplets ($\mathbf{\sigma}$ "
                             rf"Scan $\mathbf{{|{primary_symbol}| > 0}}$)")
    
    # Hammer-proj. of Siegel slopes from tau/Lambda<0 maximum sigma scan
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside, c_title=r'Siegel Slope', 
                       c=neg_SS, cmap='plasma', 
                       file=result_path+'SuperCorr_NegSiegelMap.png', 
                       proj='supergalactic', savefig=1, show=show,
                       title=r"Siegel Slopes: Distance vs 1/E ($\mathbf{\sigma}$ Scan "
                             rf"$\mathbf{{{primary_symbol} < 0}}$)")
    
    # Hammer-proj. of magnetic field from tau/Lambda<0 maximum sigma scan
    # Only include points with positive Siegel slopes (possible multiplets).
    SS_ind2 = (neg_SS > 0)
    _map_supergalactic(sgl[SS_ind2], sgb[SS_ind2],
                       c_title=r'B$\cdot$S$\cdot$Z  [nG$\cdot$Mpc]', c=neg_SS[SS_ind2], 
                       cmap='plasma', file=result_path+'SuperCorr_NegFieldMap.png',
                       proj='supergalactic', savefig=1, arrows=True,
                       dirs=neg_Dir[SS_ind2],
                       arrow_len=np.exp(neg_sigma[SS_ind2]-neg_sigma[SS_ind2].min())/1.5,
                       show=show,
                       title=r"Magnetic Field Map from Multiplets ($\mathbf{\sigma}$ "
                             rf"Scan $\mathbf{{{primary_symbol} < 0}}$)")

# --- Multiplet figure ---

def _multiplet_figure(corr, sigma, grid, events, energies, Dir, Dist, W, E, SS, 
                     result_path, stat='tau', show=True):
    stat = str(stat).strip().lower()
    if stat == 'tau':
        corr_symbol = r'\tau'
    elif stat == 'lambda':
        corr_symbol = r'\Lambda'
    else:
        raise ValueError(f"Invalid statistic: {stat}")

    # Select most significant possible magnetic deflection multiplet (tau/Lambda<0) 
    # based on tau/Lambda and sigma significance.
    mask = (corr < 0) & np.isfinite(corr) & np.isfinite(sigma)
    # Indices of tau/Lambda<0, finite tau/Lambda, and finite sigma
    idx = np.where(mask)[0]
    if idx.size == 0:
      raise ValueError("No grid point with corr < 0 and finite corr, sigma; cannot "
                             "select best multiplet.")

    smax = np.max(sigma[idx])  # Maximum sigma significance
    close = idx[np.abs(sigma[idx] - smax) <= 1e-4] # Indices of sigma within 1e-4 of max

    best = close[np.argmin(corr[close])] # Index of tau/Lambda<0 with minimum tau/Lambda

    inside, _ = _wedge(events, energies, grid[best], Dir[best], W[best], E[best], 
                        Dist[best])
    
    inside_sgl = events[inside].sgl.rad
    inside_sgb = events[inside].sgb.rad
    inside_energy = energies[inside]

    # Hammer-proj. of most significant multiplet (arrow at wedge scan center grid[best])
    _map_supergalactic(inside_sgl, inside_sgb, x0=grid.sgl[best], y0=grid.sgb[best],
                       c_title=r'Energy  [EeV]', multiplet=True, dirs=Dir[best], 
                       arrow_len=3.5, marker='o', c=inside_energy, s=25, cmap='Reds_r', 
                       B=SS[best], file=result_path+'SuperCorr_HighestSigma.png', 
                       proj='supergalactic', savefig=1, show=show,
                       title=rf"Most Significant Multiplet: "
                             rf"$\mathbf{{\sigma={sigma[best]:.2f},\ "
                             rf"{corr_symbol}={corr[best]:.2f}}}$")