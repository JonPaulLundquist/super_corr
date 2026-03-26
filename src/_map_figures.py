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

def _map_figures(grid, ipix, nside, tau, Lambda, sigma, neg_sigma, SS, neg_SS, Dir, 
                neg_Dir, result_path):
    sgl = grid.sgl
    sgb = grid.sgb
    # Hammer-projection figure of tau from abs(tau) maximum significance scan
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside, 
                       c_title=r"Kendall's $\mathbf{\tau}$", c=tau, cmap='tau', 
                       file=result_path+'SuperCorr_TauMap.png', proj='supergalactic', 
                       savefig=1,
                       title=r"$\mathbf{\tau}$ from Max. $\mathbf{\sigma}$ Scan "
                             r"($\mathbf{|\tau| > 0}$)")
    
    # Hammer-proj. of Lundquist's Lambda in wedges from abs(tau) maximum sigma scan
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside,
                       c_title=r"Lundquist's $\mathbf{\Lambda}$", c=Lambda, cmap='tau', 
                       file=result_path+'SuperCorr_LambdaMap.png', proj='supergalactic', 
                       savefig=1, 
                       title=r"$\mathbf{\Lambda}$ from Max. $\mathbf{\sigma}$ Scan "
                             r"($\mathbf{|\tau| > 0}$)")

    # Hammer-proj. of tau sigma significance from abs(tau) scan
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside,
                       c_title=r'$\mathbf{\sigma}$ (Local Significance)', c=sigma, 
                       cmap='sigma', file=result_path+'SuperCorr_SigmaMap.png', 
                       proj='supergalactic', savefig=1, 
                       title=r"$\mathbf{\sigma}$ from Max. $\mathbf{\sigma}$ Scan "
                             r"($\mathbf{|\tau| > 0}$)")
    
    # Hammer-proj. of tau sigma significance from abs(tau) scan for points with tau<0.
    _map_supergalactic(sgl[tau<0], sgb[tau<0], ipix=ipix[tau<0], nside=nside,
                       c_title=r'$\mathbf{\sigma}$ (Local Significance)', c=sigma[tau<0], 
                       cmap='sigma', file=result_path+'SuperCorr_SigmaMap_NegTau.png', 
                       proj='supergalactic', savefig=1, 
                       title=r"$\mathbf{\sigma}$ from Max. $\mathbf{\sigma}$ Scan "
                             r"($\mathbf{|\tau| > 0}$) for $\mathbf{\tau < 0}$")
    
    # Hammer-proj. of tau sigma significance from tau<0 maximum sigma scan
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside,
                       c_title=r'$\mathbf{\sigma}$ (Local Significance)', c=neg_sigma, 
                       cmap='sigma', file=result_path+'SuperCorr_NegSigmaMap.png', 
                       proj='supergalactic', savefig=1, 
                       title=r"$\mathbf{\sigma}$ from Max. $\mathbf{\sigma}$ Scan "
                             r"($\mathbf{\tau < 0}$)")

    # Hammer-proj. of Siegel slopes from abs(tau) maximum significance scan.
    # A strong correlation does not necessarily correspond to a strong magnetic field 
    # (or slope of distance vs 1/E).
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside, c_title=r'Siegel Slope', c=SS, 
                       cmap='tau', file=result_path+'SuperCorr_SiegelMap.png', 
                       proj='supergalactic', savefig=1,
                       title=r"Siegel Slopes: Distance vs 1/E ($\mathbf{\sigma}$ Scan "
                             r"$\mathbf{|\tau| > 0}$)")
    
    # Hammer-proj. of magnetic field from abs(tau) maximum significance scan
    # Only include points with positive Siegel slopes and negative tau
    SS_ind = (SS>0) & (tau<0)
    _map_supergalactic(sgl[SS_ind], sgb[SS_ind],
                       c_title=r'B$\cdot$S$\cdot$Z  [nG$\cdot$Mpc]', c=SS[SS_ind], 
                       cmap='plasma', file=result_path+'SuperCorr_FieldMap.png', 
                       proj='supergalactic', savefig=1, arrows=True, dirs=Dir[SS_ind], 
                       arrow_len=np.exp(sigma[SS_ind]-min(sigma[SS_ind]))/1.5,
                       title=r"Magnetic Field Map from Multiplets ($\mathbf{\sigma}$ "
                             r"Scan $\mathbf{|\tau| > 0}$)")
    
    # Hammer-proj. of Siegel slopes from tau<0 maximum sigma scan
    _map_supergalactic(sgl, sgb, ipix=ipix, nside=nside, c_title=r'Siegel Slope', 
                       c=neg_SS, cmap='plasma', 
                       file=result_path+'SuperCorr_NegSiegelMap.png', 
                       proj='supergalactic', savefig=1,
                       title=r"Siegel Slopes: Distance vs 1/E ($\mathbf{\sigma}$ Scan "
                             r"$\mathbf{\tau < 0}$)")
    
    # Hammer-proj. of magnetic field from tau<0 maximum sigma scan
    # Only include points with positive Siegel slopes (possible multiplets).
    SS_ind2 = (neg_SS > 0)
    _map_supergalactic(sgl[SS_ind2], sgb[SS_ind2],
                       c_title=r'B$\cdot$S$\cdot$Z  [nG$\cdot$Mpc]', c=neg_SS[SS_ind2], 
                       cmap='plasma', file=result_path+'SuperCorr_NegFieldMap.png',
                       proj='supergalactic', savefig=1, arrows=True,
                       dirs=neg_Dir[SS_ind2],
                       arrow_len=np.exp(neg_sigma[SS_ind2]-neg_sigma[SS_ind2].min())/1.5,
                       title=r"Magnetic Field Map from Multiplets ($\mathbf{\sigma}$ "
                             r"Scan $\mathbf{\tau < 0}$)")

# --- Multiplet figure ---

def _multiplet_figure(tau, sigma, grid, events, energies, Dir, Dist, W, E, SS, 
                     result_path):
    # Select most significant possible magnetic deflection multiplet (tau<0) based on 
    # tau and sigma significance.
    mask = (tau < 0) & np.isfinite(tau) & np.isfinite(sigma)
    idx = np.where(mask)[0]  # Indices of tau<0, finite tau, and finite sigma
    if idx.size == 0:
      raise ValueError("No grid point with tau < 0 and finite tau, sigma; cannot "
                             "select best multiplet.")

    smax = np.max(sigma[idx])  # Maximum sigma significance
    close = idx[np.abs(sigma[idx] - smax) <= 1e-4] # Indices of sigma within 1e-4 of max

    best = close[np.argmin(tau[close])]  # Index of tau<0 with minimum tau

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
                       proj='supergalactic', savefig=1,
                       title=rf"Most Significant Multiplet: "
                             rf"$\mathbf{{\sigma={sigma[best]:.2f},\ "
                             rf"\tau={tau[best]:.2f}}}$")