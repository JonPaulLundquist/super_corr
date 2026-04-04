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


def _normalize_stat(stat):
    stat = str(stat).strip().lower()
    if stat not in {"tau", "lambda"}:
        raise ValueError(f"Invalid statistic: {stat}")
    return stat


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


def _resolve_results_npz_path(filepath=None, stat='tau'):
    """Resolve a results input to an existing super_corr .npz path."""
    if filepath is None:
        _, result_path = _get_project_paths(stat=stat)
        filepath = Path(result_path) / 'super_corr.npz'
    else:
        filepath = Path(filepath).expanduser()

    # Accept common CLI inputs:
    # - explicit file: results/tau/super_corr.npz
    # - directory:     results/tau/MC1  -> results/tau/MC1/super_corr.npz
    # - basename:      results/tau/MC1/super_corr -> .../super_corr.npz
    repo_root = Path(__file__).resolve().parent.parent
    bases = [filepath]
    if not filepath.is_absolute():
        bases.append(repo_root / filepath)

    candidates = []
    for base in bases:
        if base.is_dir():
            candidates.append(base / "super_corr.npz")
        else:
            candidates.append(base)
            if base.suffix == "":
                candidates.append(base.with_suffix(".npz"))
                candidates.append(base / "super_corr.npz")

    resolved = next((p for p in candidates if p.is_file()), None)
    if resolved is None:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Results file not found. Tried: {tried}")
    return resolved


def _load_results(filepath=None, stat='tau'):
    """Load super_corr results from a .npz file.

    If filepath is None, uses the default stat-specific path
    results/<stat>/super_corr.npz.

    Returns a dict with common scan keys (sigma, p_value, E, Dir, Dist, W, N, scans,
    grid, ipix, nside, SS, neg_SS, events, energies, parabola fit stats) plus:
      - for stat='tau': tau, neg_tau, lambdas, a_lambda, y0_lambda, R2_lambda, ...
      - for stat='lambda': lambda, neg_lambda, taus, a_tau, y0_tau, R2_tau, ...
    """
    filepath = _resolve_results_npz_path(filepath=filepath, stat=stat)

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


def _verify_npz(filepath=None, stat='tau', verbose=True):
    """Verify key/schema and shape consistency of a super_corr results file.

    Parameters
    ----------
    filepath : path-like or None
        Path to `super_corr.npz`, or a directory containing it. If None, uses
        default `results/<stat>/super_corr.npz`.
    stat : {'tau', 'lambda'}
        Expected primary scan statistic.
    verbose : bool
        If True, print a compact pass/fail report.

    Returns
    -------
    report : dict
        {
          "ok": bool,
          "filepath": str,
          "n_grid": int or None,
          "errors": list[str],
          "warnings": list[str],
        }
    """
    stat = str(stat).strip().lower()
    if stat not in {'tau', 'lambda'}:
        raise ValueError(f"Invalid statistic: {stat}")

    p = _resolve_results_npz_path(filepath=filepath, stat=stat)
    res = _load_results(p, stat=stat)

    errors = []
    warnings = []

    def _size1d(x):
        try:
            return int(np.asarray(x).reshape(-1).size)
        except Exception:
            return None

    required_common = {
        'sigma', 'p_value', 'E', 'Dir', 'Dist', 'W', 'N',
        'neg_sigma', 'neg_p', 'neg_E', 'neg_Dir', 'neg_Dist', 'neg_W', 'neg_N',
        'scans', 'grid', 'ipix', 'nside', 'SS', 'neg_SS',
        'events', 'energies',
        'a_mean', 'y0_mean', 'a_median', 'y0_median',
        'a_neg_sigma', 'y0_neg_sigma', 'a_med_neg_sigma', 'y0_med_neg_sigma',
        'a_siegel', 'y0_siegel', 'a_med_siegel', 'y0_med_siegel',
        'a_galactic', 'y0_galactic',
        'R2_mean', 'R2_median', 'R2_neg_sigma', 'R2_med_neg_sigma',
        'R2_siegel', 'R2_med_siegel', 'R2_galactic', 'stat',
    }
    required_stat = (
        {'tau', 'neg_tau', 'lambdas', 'a_lambda', 'y0_lambda', 'a_med_lambda',
         'y0_med_lambda', 'R2_lambda', 'R2_med_lambda'}
        if stat == 'tau' else
        {'lambda', 'neg_lambda', 'taus', 'a_tau', 'y0_tau', 'a_med_tau',
         'y0_med_tau', 'R2_tau', 'R2_med_tau'}
    )
    required = required_common | required_stat

    missing = sorted(k for k in required if k not in res)
    if missing:
        errors.append("Missing keys: " + ", ".join(missing))

    file_stat = str(np.asarray(res.get('stat', '')).item()).strip().lower() \
        if 'stat' in res else ''
    if file_stat and file_stat != stat:
        errors.append(
            f"Requested stat={stat!r} but file contains stat={file_stat!r}."
        )

    n_grid = _size1d(res.get('grid')) if 'grid' in res else None
    if n_grid is None or n_grid <= 0:
        errors.append("Invalid grid: could not determine positive grid size.")

    if n_grid is not None and n_grid > 0:
        grid_like_keys = [
            'sigma', 'p_value', 'E', 'Dir', 'Dist', 'W', 'N',
            'neg_sigma', 'neg_p', 'neg_E', 'neg_Dir', 'neg_Dist', 'neg_W', 'neg_N',
            'SS', 'neg_SS', 'ipix', 'scans',
            ('tau' if stat == 'tau' else 'lambda'),
            ('lambdas' if stat == 'tau' else 'taus'),
        ]
        for k in grid_like_keys:
            if k in res:
                sz = _size1d(res[k])
                if sz is None:
                    errors.append(f"{k} has unreadable shape.")
                elif sz != n_grid:
                    errors.append(f"{k} size={sz} does not match grid size={n_grid}.")

    if 'nside' in res:
        try:
            nside_val = int(np.asarray(res['nside']).item())
            if nside_val <= 0:
                errors.append(f"nside must be > 0, got {nside_val}.")
        except Exception:
            errors.append("nside is not a valid integer scalar.")

    scalar_keys = [
        'a_mean', 'y0_mean', 'a_median', 'y0_median',
        'a_neg_sigma', 'y0_neg_sigma', 'a_med_neg_sigma', 'y0_med_neg_sigma',
        'a_siegel', 'y0_siegel', 'a_med_siegel', 'y0_med_siegel',
        'a_galactic', 'y0_galactic',
        'R2_mean', 'R2_median', 'R2_neg_sigma', 'R2_med_neg_sigma',
        'R2_siegel', 'R2_med_siegel', 'R2_galactic',
        ('a_lambda' if stat == 'tau' else 'a_tau'),
        ('y0_lambda' if stat == 'tau' else 'y0_tau'),
        ('a_med_lambda' if stat == 'tau' else 'a_med_tau'),
        ('y0_med_lambda' if stat == 'tau' else 'y0_med_tau'),
        ('R2_lambda' if stat == 'tau' else 'R2_tau'),
        ('R2_med_lambda' if stat == 'tau' else 'R2_med_tau'),
    ]
    for k in scalar_keys:
        if k not in res:
            continue
        arr = np.asarray(res[k]).reshape(-1)
        if arr.size != 1:
            errors.append(f"{k} must be scalar, got size={arr.size}.")
            continue
        if not np.isfinite(float(arr[0])):
            warnings.append(f"{k} is non-finite ({arr[0]}).")

    if 'events' in res and 'energies' in res:
        n_events = _size1d(res['events'])
        n_energies = _size1d(res['energies'])
        if n_events is not None and n_energies is not None and n_events != n_energies:
            errors.append(
                f"events size={n_events} and energies size={n_energies} do not match."
            )

    ok = len(errors) == 0
    report = {
        'ok': ok,
        'filepath': str(p),
        'n_grid': n_grid,
        'errors': errors,
        'warnings': warnings,
    }
    if verbose:
        status = "PASSED" if ok else "FAILED"
        print(f"Integrity check {status}: {p}")
        if warnings:
            print("Warnings:")
            for w in warnings:
                print(f"  - {w}")
        if errors:
            print("Errors:")
            for e in errors:
                print(f"  - {e}")
    return report

# --- Paths ---

def _get_project_paths(stat='tau'):
    # Data directory relative to repository path
    stat = str(stat).strip().lower()
    if stat not in {'tau', 'lambda'}:
        raise ValueError(f"Invalid statistic: {stat}")

    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "dataSummarySD1500.csv"
    result_dir = root / "results" / stat
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = str(result_dir) + "/"

    return data_path, result_path


def _next_mc_dir(stat='tau', seed=None):
    """Return default isotropic output directory under results/<stat>/.

    If seed is provided -> results/<stat>/MC_<seed>
    If seed is None     -> next auto-incremented results/<stat>/MC#
    """
    _, stat_result_path = _get_project_paths(stat=stat)
    base = Path(stat_result_path)

    if seed is not None:
        mc_dir = base / f"MC_{int(seed)}"
        mc_dir.mkdir(parents=True, exist_ok=True)
        return str(mc_dir)

    existing = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("MC")]
    nums = []
    for p in existing:
        suffix = p.name[2:]
        if suffix.isdigit():
            nums.append(int(suffix))

    next_idx = (max(nums) + 1) if nums else 1
    mc_dir = base / f"MC{next_idx}"
    mc_dir.mkdir(parents=True, exist_ok=True)
    return str(mc_dir)