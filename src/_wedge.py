#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 02:37:12 2026

@author: Jon Paul Lundquist
"""

import numpy as np
from scipy.stats import siegelslopes
from lambda_corr import lambda_corr

def _wedge(events, energies, center, Dir, W, E_cut, Dist):
    # Create a wedge of events with energy >= E_cut, angular width <= W, and angular 
    # distance <= Dist starting at location center, extending in pointing direction Dir.
    # Returns indices of events inside wedge and their angular separation from the 
    # wedge location (aka scan center).

    # Calculate separation distance in degrees of events from wedge location        
    separation = center.separation(events).deg
    # Calculate azimuthal angle of events around North from wedge location
    azimuthal = center.position_angle(events).deg
    # Calculate relative azimuthal angle between wedge pointing and event directions
    width = (azimuthal - Dir) % 360
    # Ensure smaller angle
    width = np.where(width > 180, 360 - width, width)
    # Create a mask of events that are inside the wedge
    inside = ((energies>=E_cut) & (width<=W) & (separation<=Dist))
    
    return inside, separation[inside]


def _wedge_members(events, energies, grid, Dir, W, E, Dist):
    # For each wedge get event members inside it. Used to estimate uncertainties for 
    # parabola fits. We'd rather keep track of this outside the scan for optimization
    n_wedges = len(grid)
    n_events = len(events)
    members = np.zeros((n_wedges, n_events), dtype=bool)
    for i, center in enumerate(grid):
        inside, _ = _wedge(events, energies, center, Dir[i], W[i], E[i], Dist[i])
        members[i] = inside

    return members

# --- Post-scan: wedge statistics ---

def _siegel_slopes(events, energies, grid, Dir, W, E, Dist):
    # Calculate Z*S*B magnetic field estimate using median-of-medians robust slope 
    # estimate of distance vs 1/E.
    slopes = np.zeros(len(grid))
    for i in range(len(grid)):
        inside, sep = _wedge(events, energies, grid[i], Dir[i], W[i], E[i], Dist[i])
        res = siegelslopes(sep, 1/(energies[inside]))
        slope = res.slope
        slopes[i] = slope/(0.5*10**2) # Energies are EeV. This results in Mpc*nG.

    return slopes


def _lambda_and_siegel(events, energies, grid, Dir, W, E, Dist):
    # Lambda correlation scan is slow but we can compare to tau with the wedges given.
    lambdas = np.zeros(len(grid))
    SS = np.zeros(len(grid))
    for i in range(len(grid)):
        inside, sep = _wedge(events, energies, grid[i], Dir[i], W[i], E[i], Dist[i])
        lambdas[i], *_ = lambda_corr(energies[inside], sep, pvals=False)
        
        # Calculate Z*S*B magnetic field estimate using median-of-medians robust slope 
        # estimate of distance vs 1/E.
        res = siegelslopes(sep, 1/(energies[inside]))
        slope = res.slope
        SS[i] = slope/(0.5*10**2) # Energies are EeV. This results in Mpc*nG.

    return lambdas, SS