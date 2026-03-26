#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project paths and data loading.
Created on Tue Mar 22 11:58:27 2026

@author: Jon Paul Lundquist
"""

import numpy as np
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy import units as u

from _iso_MC_Auger import _iso_MC_Auger

# --- Data loading ---

def _load_input_events(input_type, data_path, seed=None):
    data_Auger = _load_auger_dataset(data_path)

    # Wedge scan on Auger data
    if input_type == 'data':
        ra = data_Auger['sd_ra'].astype(np.float32) # TETE Right Ascension
        dec = data_Auger['sd_dec'].astype(np.float32) # TETE Declination
        energy = data_Auger['sd_energy'] # Energy in 10^18 eV (EeV)
        # Initialize data to True Equator, True Equinox SkyCoord object
        #(system used for Auger data). And convert to Supergalactic coordinates
        events = SkyCoord(ra=ra*u.degree, dec=dec*u.degree,
                          frame='tete').transform_to('supergalactic')

    # Wedge scan on isotropic Monte Carlo (null) data
    elif input_type == 'iso':
        # Create isotropic (null) data that keeps detector measurement correlations
        # intact (energy-zenith, zenith-azimuth) and detector configuration and on-time
        # (using data trigger times with the shower core coordinates). While scrambling
        # zenith, azimuth, and energy together in blocks to preserve observed
        # energy-time correlation.

        mc_coords, mc_energy, _ = _iso_MC_Auger(data_Auger, seed=seed)

        # Treat MC exactly the same as data.
        events = mc_coords.transform_to('supergalactic')
        energy = mc_energy.astype(np.float32)

    else:
        raise ValueError(f"Unknown input_type={input_type!r}. \
                         Expected 'data' or 'iso' (isotropic Monte Carlo).")

    return events, energy

def _load_auger_dataset(data_path):
    # Load Auger data from CSV file and remove rows with missing values
    data_Auger = np.genfromtxt(data_path, delimiter=',', names=True,
                               filling_values=np.nan)

    return data_Auger[~np.isnan(data_Auger['sd_ra']) & ~np.isnan(data_Auger['sd_dec'])
                      & ~np.isnan(data_Auger['sd_energy'])]

# --- Save and load results ---

def _save_results(result_path, **kwargs):
    # Save the results to a numpy .npz file.
    np.savez(result_path+'super_corr.npz', **kwargs)


def _load_results(filepath=None):
    """Load super_corr results from a .npz file.

    If filepath is None, uses the default results path (result_path + 'super_corr.npz').
    Returns a dict. with keys: tau, sigma, p_value, E, Dir, Dist, W, N, neg_tau;
    neg_sigma, neg_p, neg_E, neg_Dir, neg_Dist, neg_W, neg_N, scans; grid, ipix, nside;
    lambdas, SS, neg_SS; parabola fit stats (a_mean, y0_mean, etc.); events, energies.
    """
    if filepath is None:
        _, result_path = _get_project_paths()
        filepath = Path(result_path) / 'super_corr.npz'

    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    res = np.load(filepath, allow_pickle=True)
    out = dict(res)

    # Reconstruct SkyCoord for grid (stack scalar SkyCoords if needed)
    if 'grid' in out:
        g = out['grid']
        out['grid'] = g if isinstance(g, SkyCoord) else SkyCoord(g)

    if 'events' in out:
        ev = out['events']
        if isinstance(ev, SkyCoord):
            out['events'] = ev

        else:
            out['events'] = SkyCoord(
                sgl=[c.sgl.deg for c in ev] * u.deg,
                sgb=[c.sgb.deg for c in ev] * u.deg,
                frame='supergalactic')

    return out

# --- Paths ---

def _get_project_paths():
    # Data directory relative to repository path
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "dataSummarySD1500.csv"
    result_path = str(root / "results") + "/"

    return data_path, result_path