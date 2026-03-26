#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 11:46:10 2026

@author: Jon Paul Lundquist
"""

import numpy as np
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from astropy import units as u

from _scan_kernel import _scan_center

def _prepare_scan_inputs(grid, events, energy, Ecut, max_distance):
    # Lowest energy cut
    events = events[energy >= Ecut]
    energies = energy[energy >= Ecut].astype(np.float32)
    
    # Initial sort necessary for fastest possible scan_center scan.
    ind = np.argsort(energies)
    events = events[ind]
    energies = energies[ind]
    
    # For each grid point calculate all events angular distance from it
    center_seps = grid[:, None].separation(events)
    # Mask to select events within scan maximum distance of 90 deg. from each grid point
    # This is a spherical cap (or top hat) filter. Wedges are spherical cap sections.
    event_mask = (center_seps <= max_distance*u.deg)
    # Precompute azimuth angles from North for events inside 90 degrees for grid points
    azimuthal_list = [
        center.position_angle(events[event_mask[i]]).deg.astype(np.float32, order='C')
        for i, center in enumerate(grid)
    ]
    
    # Many NumPy operations (and lots of C/Numba code) are fastest on contiguous arrays.
    energies = energies.astype(np.float32, order='C')
    separation = center_seps.deg.astype(np.float32, order='C')

    return events, energies, separation, event_mask, azimuthal_list

# --- Scan kernel orchestration ---

def _run_scan(grid_size, event_mask, energies, separation, azimuthal_list,
                    directions, Ecuts, widths, distances, minN, num_workers):
    # Runs scan_center_unpack(args) once per grid point in parallel threads, using 
    # ex.map(...). tqdm wraps iterator for progress bar while all centers are scanned.
    args = _iter_scan_args(grid_size, event_mask, energies, separation, azimuthal_list,
                            directions, Ecuts, widths, distances, minN)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        return list(tqdm(ex.map(_scan_unpack, args),
                         total=grid_size, smoothing=0.2))


def _scan_unpack(args):
    # Unpack the arguments and call the scan_center function for multiprocessing.
    return _scan_center(*args)


def _iter_scan_args(grid_size, event_mask, energies, separation, azimuthal_list,
                     directions, Ecuts, widths, distances, minN):
    # Loops over every grid point (for i in range(grid.size)) and builds argument tuple 
    # for that center
    for i in range(grid_size):
        m = event_mask[i] # boolean mask selecting which events belong to center (i)
        yield (energies[m], separation[i, m], azimuthal_list[i],
               directions, Ecuts, widths, distances, minN)


# --- Scan results unpacking into super_corr ---
def _unpack_results(results, grid_size):
    # Unpack the per-center scan results into arrays.
    # abs(tau) maximum sigma significance scan results.
    tau = np.empty(grid_size)  # Kendall tau correlation in each wedge.
    p_value = np.empty(grid_size) # Two-sided p-value for tau=0 null hypothesis.
    E = np.empty(grid_size) # Minimum energy cut for each wedge.
    Dir = np.empty(grid_size) # Direction of each wedge.
    Dist = np.empty(grid_size) # Maximum distance cut for each wedge.
    W = np.empty(grid_size) # Maximum width cut for each wedge.
    N = np.empty(grid_size) # Number of events in each wedge.

    # Negative tau only maximum sigma significance scan results.
    neg_tau = np.empty(grid_size) # Kendall tau correlation in wedge.
    neg_p = np.empty(grid_size) # Two-sided p-value for tau=0 null hypothesis.
    neg_E = np.empty(grid_size) # Minimum energy cut for wedge.
    neg_Dir = np.empty(grid_size) # Direction of wedge.
    neg_Dist = np.empty(grid_size) # Maximum distance cut for wedge.
    neg_W = np.empty(grid_size) # Maximum width cut for wedge.
    neg_N = np.empty(grid_size) # Number of events in wedge.
    
    scans = np.empty(grid_size) # Number of scans for each wedge.

    for i, res in enumerate(results):
        (tau[i], p_value[i], E[i], Dir[i], Dist[i], W[i], N[i],
         neg_tau[i], neg_p[i], neg_E[i], neg_Dir[i], neg_Dist[i], neg_W[i], neg_N[i],
         scans[i]) = res
    
    # Sigma significance of p-values
    sigma = norm.isf(p_value/2) # Inverse survival function of unit normal distribution
    neg_sigma = norm.isf(neg_p/2)

    return (tau, sigma, p_value, E, Dir, Dist, W, N, neg_tau, neg_sigma, neg_p, neg_E, 
            neg_Dir, neg_Dist, neg_W, neg_N, scans)