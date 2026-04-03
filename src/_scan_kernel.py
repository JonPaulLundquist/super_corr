#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kernel for the UHECR wedge scan using Kendall's tau correlation.

This module evaluates the most significant wedge correlation at a single sky center by 
scanning direction, energy cut, wedge width, and distance cut. It uses reusable index 
buffers and preallocated scratch arrays to minimize temporary allocations inside nested 
loops. The implementation is designed for high-throughput center-by-center scans where
memory traffic and allocation overhead can dominate runtime.

- Applies hierarchical cuts (direction -> energy -> width -> distance).
- Evaluates Kendall tau / p-value on surviving event subsets.
- Tracks best |tau|-scan significance and best tau<0 significance.
- Returns correlation values, p-values, best-fit wedge parameters, N events in best 
  wedge, and number of evaluated candidates.

Created on Thu Mar 20 11:40:51 2026

@author: Jon Paul Lundquist
"""

from hyper_corr import kendalltau_ties, kendalltau_noties
from numba import njit
import numpy as np


@njit(nogil=True, fastmath=True)
def _scan_center(energies, separation, azimuthal, directions, Ecuts,
                 widths, distances, minN):
    """
    Scan a single sky-center candidate over wedge parameters and return the best
    |tau|-scan and tau<0 scan results.

    Parameters
    ----------
    energies : 1D float array
        Event energies for this center. Expected sorted ascending.
    separation : 1D float array
        Event angular separations (deg) from the center.
    azimuthal : 1D float array
        Event position angles (deg) around the center.
    directions, Ecuts, widths, distances : 1D float arrays
        Scan grids for wedge direction, minimum energy cut, half-width, and
        maximum distance cut.
    minN : int
        Minimum number of events required to evaluate Kendall tau.

    Returns
    -------
    tuple
        (best_tau, best_p, best_E, best_dir, best_distance, best_width, best_N,
         neg_tau, neg_p, neg_E, neg_dir, neg_distance, neg_width, neg_N,
         scan_count)
    """
    # Energies and separations have been cut for maximum 90 deg distance around the 
    # scan center. Process wedge correlation center using event energies, separations, 
    # azimuthal angles, and possible wedge directions, energy cuts, widths, distances.

    # Initialize wedge parameters for the best (lowest) p-value |tau|>0
    best_tau = 0.0
    best_p = 1.0
    best_E = np.nan
    best_dir = np.nan
    best_distance = np.nan
    best_width = np.nan
    best_N = 0
    scan_count = 0

    # Best scan constrained to tau < 0
    neg_tau = 0.0
    neg_p = 1.0
    neg_E = np.nan
    neg_dir = np.nan
    neg_distance = np.nan
    neg_width = np.nan
    neg_N = 0

    # Initial event set size 90 deg distance around center
    N = energies.size
    max_width = np.max(widths)

    # Reusable scratch buffers
    width_arr = np.empty(N, dtype=np.float32)
    idx_dir = np.empty(N, dtype=np.int32)
    idx_e = np.empty(N, dtype=np.int32)
    idx_w = np.empty(N, dtype=np.int32)
    idx_d = np.empty(N, dtype=np.int32)
    prev_idx_dir = np.empty(N, dtype=np.int32)
    prev_n_dir = -1

    E_final = np.empty(N, dtype=energies.dtype)
    D_final = np.empty(N, dtype=separation.dtype)

    # Center-level check for duplicates in correlation variables before scanning.
    energy_dup0 = False
    if N > 1:
        # Energies are sorted so only one loop is needed to check for duplicates
        prev_e = energies[0]
        for i in range(1, N):
            ei = energies[i]
            if ei == prev_e:
                energy_dup0 = True
                break
            prev_e = ei

    dist_dup0 = False
    for i in range(N):
        # Separation is unsorted so two loops are needed to check for duplicates
        si = separation[i]
        for j in range(i + 1, N):
            if si == separation[j]:
                dist_dup0 = True
                break

        if dist_dup0:
            break

    # Scan wedge pointing direction
    for direction in directions:
        # Build relative widths and initial direction cut in one pass 
        # using index buffer idx_dir
        n_dir = 0
        # Ensure smaller angle between wedge pointing and event directions
        for i in range(N):
            w = azimuthal[i] - direction
            if w < 0.0:
                w += 360.0
            if w > 180.0:
                w = 360.0 - w

            width_arr[i] = w
            # Get indices of events inside maximum width around pointing direction
            if w <= max_width:
                idx_dir[n_dir] = i
                n_dir += 1

        # If we have less than minN events continue with next direction
        if n_dir < minN:
            continue

        # If direction has same set of events as previous continue to next direction.
        if n_dir == prev_n_dir and _same_events(idx_dir, prev_idx_dir, n_dir):
            continue

        # Store previous direction indices
        for i in range(n_dir):
            prev_idx_dir[i] = idx_dir[i]
        prev_n_dir = n_dir

        # Scan wedge energy cuts
        NE_prev = N + 1
        for Ecut in Ecuts:
            # Indices and number of events E>=Ecut and +/- 45 degrees around direction 
            # using index buffer idx_e
            n_e = _filter_idx_ge(energies, idx_dir, n_dir, Ecut, idx_e)
            # If we have less than minN events E>=Ecut break to next direction
            if n_e < minN:
                break
            # If no new events removed continue to larger energy cut
            if n_e == NE_prev:
                continue
            NE_prev = n_e
            
            # Scan wedge width cuts
            NW_prev = N + 1
            # Indices and number of events E>=Ecut, +/- width degrees, and 
            # distance from scan center using index buffer idx_w
            for width_cut in widths:
                n_w = _filter_idx_le(width_arr, idx_e, n_e, width_cut, idx_w)
                # If we have less than minN events E>=Ecut break to next direction
                if n_w < minN:
                    break
                # If no new events removed continue to smaller width cut
                if n_w == NW_prev:
                    continue
                NW_prev = n_w

                # Scan wedge distance cuts
                ND_prev = N + 1
                for distance_cut in distances:
                    # Indices and number of events E>=Ecut, +/- width degrees, and 
                    # distance from scan center using index buffer idx_w
                    n_d = _filter_idx_le(separation, idx_w, n_w, distance_cut, idx_d)
                    # If we have less than minN events E>=Ecut break to next direction
                    if n_d < minN:
                        break
                    if n_d == ND_prev:
                        continue
                    ND_prev = n_d

                    # Gather paired (E, D) only for surviving candidates.
                    for k in range(n_d):
                        ii = idx_d[k]
                        E_final[k] = energies[ii]
                        D_final[k] = separation[ii]

                    # Duplicate checks only when full center set had duplicates.
                    if energy_dup0:
                        energy_dup = _sorted_dups(energies, idx_d, n_d)
                        
                    else:
                        energy_dup = False

                    # Duplicate checks only when full center set had duplicates.
                    if dist_dup0:
                        dist_dup = _unsorted_dups(separation, idx_d, n_d)

                    else:
                        dist_dup = False

                    # hyper_corr Kendall tau optimized based on ties or no ties
                    # Returns tau correlation and two-sided p-value (tau=0 null)
                    if energy_dup or dist_dup:
                        t_tau, t_p = kendalltau_ties(E_final[:n_d], D_final[:n_d], n_d)

                    else:
                        t_tau, t_p = kendalltau_noties(E_final[:n_d], D_final[:n_d], 
                                                       n_d)

                    scan_count += 1.0

                    # Absolute tau scan used in supergalactic structure paper.
                    # If p-value is smaller scan is better. If p-value is the same, 
                    # greater absolute value tau is better.
                    if (t_p < best_p) or (
                        (t_p == best_p) and (abs(t_tau) >= abs(best_tau))):
                        best_p = t_p
                        best_tau = t_tau
                        best_E = Ecut
                        best_dir = direction
                        best_distance = distance_cut
                        best_width = width_cut
                        best_N = n_d # Final number of events in wedge

                    # Negative tau only scan. Largely, only sigma significance is 
                    # interesting in regards to it's supergalactic symmetry.
                    # If p-value is smaller for tau<0 scan is better. If p-value is the 
                    # same, more negative tau is better.
                    if ((t_p < neg_p) and (t_tau < 0.0)) or (
                        (t_p == neg_p) and (t_tau <= neg_tau)):
                        neg_p = t_p
                        neg_tau = t_tau
                        neg_E = Ecut
                        neg_dir = direction
                        neg_distance = distance_cut
                        neg_width = width_cut
                        neg_N = n_d # Final number of events in wedge

    return (best_tau, best_p, best_E, best_dir, best_distance, best_width, best_N,
            neg_tau, neg_p, neg_E, neg_dir, neg_distance, neg_width, neg_N, scan_count)

#---Internal helper functions for _scan_center---

@njit(nogil=True, fastmath=True, inline="always")
def _same_events(a, b, n):
    # Check if two arrays are the same up to the first `n` elements
    for i in range(n):
        if a[i] != b[i]:
            return False

    return True


@njit(nogil=True, fastmath=True, inline="always")
def _filter_idx_ge(values, idx_in, n_in, cut, idx_out):
    # Built for numba to speed up the selection by indexing of events in x >= cut, 
    # where x is size n
    n_out = 0
    for k in range(n_in):
        i = idx_in[k]
        if values[i] >= cut:
            idx_out[n_out] = i
            n_out += 1

    return n_out


@njit(nogil=True, fastmath=True, inline="always")
def _filter_idx_le(values, idx_in, n_in, cut, idx_out):
    # Built for numba to speed up the selection by indexing of values <= cut, 
    # where values is size n_in
    n_out = 0
    for k in range(n_in):
        i = idx_in[k]
        if values[i] <= cut:
            idx_out[n_out] = i
            n_out += 1

    return n_out


@njit(nogil=True, fastmath=True, inline="always")
def _sorted_dups(values, idx, n):
    # Duplicate check of globally pre-sorted `values`; selected `idx` preserves order
    # This only runs when duplicates are detected in the full center set.
    # If there are use the ties version of Kendall tau
    # njit likes loops over arrays
    
    #if n <= 1: # Not needed since n >= 1
    #    return False
    
    # Sorted so only check adjacent elements
    for k in range(1, n):
        if values[idx[k - 1]] == values[idx[k]]:
            return True

    return False


@njit(nogil=True, fastmath=True, inline="always")
def _unsorted_dups(values, idx, n):
    # Allocation-free O(n^2) duplicate check over selected unsorted distances.
    # This only runs when duplicates are detected in the full center set.
    # If there are use the ties version of Kendall tau
    # njit likes loops over arrays
    for i in range(n):
        vi = values[idx[i]]
        for j in range(i + 1, n):
            if vi == values[idx[j]]:
                return True

    return False

