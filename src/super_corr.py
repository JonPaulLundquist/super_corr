#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
super_corr.py

Implements the wedge-based (spherical cap section) UHECR energy-angle correlation, 
or multiplet, scan and the supergalactic curvature test statistic, used to test whether 
UHECR multiplets exhibit large-scale supergalactic structure.

The analysis improves upon the methodology from my Telescope Array paper: "Evidence 
for a Supergalactic Structure of Magnetic Deflection Multiplets of Ultra-High Energy
Cosmic Rays" and applies the analysis framework to Pierre Auger Open Data. The goal 
is to test whether the same large-scale supergalactic structure is visible in an
independent data set from the opposite hemisphere.

References:
https://arxiv.org/abs/2005.07312
https://opendata.auger.org/

Created on Tue Mar 11 17:04:57 2025

@author: Jon Paul Lundquist
"""

import numpy as np
from tqdm import tqdm
import os
import gc
from pathlib import Path

from _scan_driver import _run_scan, _unpack_results, _prepare_scan_inputs
from _scan_params import _build_scan_params, _build_scan_grid
from _super_io import _get_project_paths, _load_input_events, _save_results
from _super_io import _load_results
from _map_figures import _multiplet_figure, _map_figures
from _wedge import _wedge_members, _siegel_slopes, _lambda_and_siegel
from _parabolas import _parabola_figures, _parabola_stats

# --- Main analysis ---

def super_corr(input_type='data', make_figures=True, save_npz=True, result_path=None,
               seed=None):
    # Determine the number of workers for multithreading the wedge scan
    num_workers = int(np.round(os.cpu_count()))
    
    # Build scan parameters and grid.
    minN, min_Ecut, distances, widths, directions, Ecuts = _build_scan_params()
    grid, ipix, nside = _build_scan_grid()
    data_path, default_result_path = _get_project_paths()
    if result_path is None:
        result_path = default_result_path
    else:
        root = Path(__file__).resolve().parent.parent
        rp = Path(result_path)
        if not rp.is_absolute():
            rp = (root / result_path).resolve()
        else:
            rp = rp.resolve()
        rp.mkdir(parents=True, exist_ok=True)
        result_path = str(rp).rstrip("/") + "/"
    
    # Load input events and prepare scan inputs.
    events, energy = _load_input_events(input_type, data_path, seed=seed)
    events, energies, separations, event_mask, azimuthal_list = _prepare_scan_inputs(
        grid, events, energy, min_Ecut, np.max(distances))
    
    # Run scan for maximum sigma significance for each grid point in parallel threads.
    results = _run_scan(grid.size, event_mask, energies, separations, 
                              azimuthal_list, directions, Ecuts, widths, distances, 
                              minN, num_workers)
    
    # Unpack the scan results. neg_* variables are for the negative tau only scan.
    tau, sigma, p_value, E, Dir, Dist, W, N, neg_tau, neg_sigma, neg_p, neg_E, \
        neg_Dir, neg_Dist, neg_W, neg_N, scans = _unpack_results(results, grid.size)
    
    # Build wedge members for parabola fits uncertainty estimation.
    members = _wedge_members(events, energies, grid, Dir, W, E, Dist)
    
    lambdas, SS = _lambda_and_siegel(events, energies, grid, Dir, W, E, Dist)
    
    neg_SS = _siegel_slopes(events, energies, grid, neg_Dir, neg_W, neg_E, 
                                    neg_Dist)
    
    if make_figures:
        # Create the best multiplet figure
        _multiplet_figure(tau, sigma, grid, events, energies, Dir, Dist, W, E, SS,
                         result_path)

        # Create the projection map figures
        _map_figures(grid, ipix, nside, tau, lambdas, sigma, neg_sigma, SS, neg_SS, Dir,
                    neg_Dir, result_path)

        popt_mean, R2_mean, popt_median, R2_median, \
            popt_lambda, R2_lambda, popt_med_lambda, R2_med_lambda, \
            popt_neg_sigma, R2_neg_sigma, popt_med_neg_sigma, R2_med_neg_sigma, \
            popt_siegel, R2_siegel, popt_med_siegel, R2_med_siegel, \
            popt_galactic, R2_galactic = _parabola_figures(
                grid, tau, members, lambdas, neg_sigma, SS, result_path)

    else:
        popt_mean, R2_mean, popt_median, R2_median, \
            popt_lambda, R2_lambda, popt_med_lambda, R2_med_lambda, \
            popt_neg_sigma, R2_neg_sigma, popt_med_neg_sigma, R2_med_neg_sigma, \
            popt_siegel, R2_siegel, popt_med_siegel, R2_med_siegel, \
            popt_galactic, R2_galactic = _parabola_stats(
                grid, tau, members, lambdas, neg_sigma, SS)

    a_mean, y0_mean = popt_mean
    a_median, y0_median = popt_median
    a_lambda, y0_lambda = popt_lambda
    a_med_lambda, y0_med_lambda = popt_med_lambda
    a_neg_sigma, y0_neg_sigma = popt_neg_sigma
    a_med_neg_sigma, y0_med_neg_sigma = popt_med_neg_sigma
    a_siegel, y0_siegel = popt_siegel
    a_med_siegel, y0_med_siegel = popt_med_siegel
    a_galactic, y0_galactic = popt_galactic

    results_out = dict(tau=tau, sigma=sigma, p_value=p_value, E=E,
        Dir=Dir, Dist=Dist, W=W, N=N, scans=scans, neg_tau=neg_tau, neg_sigma=neg_sigma,
        neg_p=neg_p, neg_E=neg_E, neg_Dir=neg_Dir, neg_Dist=neg_Dist, neg_W=neg_W,
        neg_N=neg_N, grid=grid, ipix=ipix, nside=nside, events=events, 
        energies=energies, lambdas=lambdas, SS=SS, neg_SS=neg_SS, a_mean=a_mean, 
        y0_mean=y0_mean, a_median=a_median, y0_median=y0_median, a_lambda=a_lambda, 
        y0_lambda=y0_lambda, a_med_lambda=a_med_lambda, y0_med_lambda=y0_med_lambda, 
        a_neg_sigma=a_neg_sigma, y0_neg_sigma=y0_neg_sigma, 
        a_med_neg_sigma=a_med_neg_sigma, y0_med_neg_sigma=y0_med_neg_sigma,
        a_siegel=a_siegel, y0_siegel=y0_siegel, a_med_siegel=a_med_siegel, 
        y0_med_siegel=y0_med_siegel, a_galactic=a_galactic, y0_galactic=y0_galactic,
        R2_mean=R2_mean, R2_median=R2_median, R2_lambda=R2_lambda, 
        R2_med_lambda=R2_med_lambda, R2_neg_sigma=R2_neg_sigma, 
        R2_med_neg_sigma=R2_med_neg_sigma, R2_siegel=R2_siegel,
        R2_med_siegel=R2_med_siegel, R2_galactic=R2_galactic,
    )

    if save_npz:
        _save_results(result_path, **results_out)

# --- Garbage collection necessary for threads and memory management ---
    for _ in range(3):
        gc.collect()
    
    return results_out


def run_mc_trials(n_trials, out_path=None, seed=None):
    """Run isotropic MC trials and save per-trial a, y0, and R2.

    All saved arrays are float32 for compact storage.
    """
    _, result_path = _get_project_paths()
    if out_path is None:
        out_path = str(Path(result_path) / "mc_stats.npz")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_trials = int(n_trials)
    dt = np.float32
    a_mean = np.empty(n_trials, dtype=dt)
    a_median = np.empty(n_trials, dtype=dt)
    a_lambda = np.empty(n_trials, dtype=dt)
    a_med_lambda = np.empty(n_trials, dtype=dt)
    a_neg_sigma = np.empty(n_trials, dtype=dt)
    a_med_neg_sigma = np.empty(n_trials, dtype=dt)
    a_siegel = np.empty(n_trials, dtype=dt)
    a_med_siegel = np.empty(n_trials, dtype=dt)
    a_galactic = np.empty(n_trials, dtype=dt)

    y0_mean = np.empty(n_trials, dtype=dt)
    y0_median = np.empty(n_trials, dtype=dt)
    y0_lambda = np.empty(n_trials, dtype=dt)
    y0_med_lambda = np.empty(n_trials, dtype=dt)
    y0_neg_sigma = np.empty(n_trials, dtype=dt)
    y0_med_neg_sigma = np.empty(n_trials, dtype=dt)
    y0_siegel = np.empty(n_trials, dtype=dt)
    y0_med_siegel = np.empty(n_trials, dtype=dt)
    y0_galactic = np.empty(n_trials, dtype=dt)

    R2_mean = np.empty(n_trials, dtype=dt)
    R2_median = np.empty(n_trials, dtype=dt)
    R2_lambda = np.empty(n_trials, dtype=dt)
    R2_med_lambda = np.empty(n_trials, dtype=dt)
    R2_neg_sigma = np.empty(n_trials, dtype=dt)
    R2_med_neg_sigma = np.empty(n_trials, dtype=dt)
    R2_siegel = np.empty(n_trials, dtype=dt)
    R2_med_siegel = np.empty(n_trials, dtype=dt)
    R2_galactic = np.empty(n_trials, dtype=dt)

    if seed is None:
        # True random seeding (from OS entropy) for distributed/HPC workflows.
        seed = int(np.random.default_rng().integers(0, 2**32, dtype=np.uint32))
        
    else:
        seed = int(seed)

    trial_seeds = (seed + np.arange(n_trials, dtype=np.uint32)).astype(np.uint32)

    for i in tqdm(range(n_trials), total=n_trials, smoothing=0.2):
        # Headless, do-not-save scan.
        res = super_corr(input_type='iso', make_figures=False, save_npz=False,
                         seed=int(trial_seeds[i]))

        a_mean[i] = res['a_mean']
        a_median[i] = res['a_median']
        a_lambda[i] = res['a_lambda']
        a_med_lambda[i] = res['a_med_lambda']
        a_neg_sigma[i] = res['a_neg_sigma']
        a_med_neg_sigma[i] = res['a_med_neg_sigma']
        a_siegel[i] = res['a_siegel']
        a_med_siegel[i] = res['a_med_siegel']
        a_galactic[i] = res['a_galactic']

        y0_mean[i] = res['y0_mean']
        y0_median[i] = res['y0_median']
        y0_lambda[i] = res['y0_lambda']
        y0_med_lambda[i] = res['y0_med_lambda']
        y0_neg_sigma[i] = res['y0_neg_sigma']
        y0_med_neg_sigma[i] = res['y0_med_neg_sigma']
        y0_siegel[i] = res['y0_siegel']
        y0_med_siegel[i] = res['y0_med_siegel']
        y0_galactic[i] = res['y0_galactic']

        R2_mean[i] = res['R2_mean']
        R2_median[i] = res['R2_median']
        R2_lambda[i] = res['R2_lambda']
        R2_med_lambda[i] = res['R2_med_lambda']
        R2_neg_sigma[i] = res['R2_neg_sigma']
        R2_med_neg_sigma[i] = res['R2_med_neg_sigma']
        R2_siegel[i] = res['R2_siegel']
        R2_med_siegel[i] = res['R2_med_siegel']
        R2_galactic[i] = res['R2_galactic']

        # Save the total results to a .npz file after each trial in case of HPC preemption
        # or other interruption.
        out = dict(n_trials=n_trials, seed=seed, trial_seeds=trial_seeds,
            a_mean=a_mean, a_median=a_median, a_lambda=a_lambda, 
            a_med_lambda=a_med_lambda, a_neg_sigma=a_neg_sigma, 
            a_med_neg_sigma=a_med_neg_sigma, a_siegel=a_siegel, 
            a_med_siegel=a_med_siegel, a_galactic=a_galactic, y0_mean=y0_mean,
            y0_median=y0_median, y0_lambda=y0_lambda, y0_med_lambda=y0_med_lambda,
            y0_neg_sigma=y0_neg_sigma, y0_med_neg_sigma=y0_med_neg_sigma, 
            y0_siegel=y0_siegel, y0_med_siegel=y0_med_siegel, y0_galactic=y0_galactic,  
            R2_mean=R2_mean, R2_median=R2_median, R2_lambda=R2_lambda,
            R2_med_lambda=R2_med_lambda, R2_neg_sigma=R2_neg_sigma,
            R2_med_neg_sigma=R2_med_neg_sigma, R2_siegel=R2_siegel,
            R2_med_siegel=R2_med_siegel, R2_galactic=R2_galactic)

        np.savez_compressed(out_path, **out)
    return str(out_path)


# Display order for mc_pvalues() output (blank line between groups).
_MC_PVALUE_ORDER_GROUPS = (("a_mean", "a_lambda"), ("a_neg_sigma", "a_siegel"),
    ("a_median", "a_med_lambda", "a_med_neg_sigma", "a_med_siegel"), ("a_galactic",))
def mc_pvalues(data_npz=None, mc_npz=None, out_txt=None):
    """Compute one-sided MC p-values for each curvature parameter a_*.

    Denominator is every MC trial. The extreme-event count includes a trial only if
    the tail is satisfied and R^2 > 0 for the matching parabola (same suffix as a_*,
    e.g. a_mean <-> R2_mean). Trials with R^2 <= 0 still count toward the denominator
    but not toward the numerator.

    Tail directions (numerator uses tail & R2>0):
      - a_mean, a_median, a_lambda, a_med_lambda : count #(a_mc >= a_data & R2>0)
      - others (currently 5)                     : count #(a_mc <= a_data & R2>0)
    """
    _, result_path = _get_project_paths()
    if data_npz is None:
        data_npz = Path(result_path) / "super_corr.npz"

    else:
        data_npz = Path(data_npz)

    if mc_npz is None:
        mc_npz = Path(result_path) / "mc_stats.npz"

    else:
        mc_npz = Path(mc_npz)

    if out_txt is None:
        out_txt = Path(result_path) / "mc_pvalues.txt"

    else:
        out_txt = Path(out_txt)

    data = _load_results(data_npz)
    mc = dict(np.load(mc_npz, allow_pickle=True))

    # Define which tail is "more extreme" for each statistic.
    # If a key is not listed here, we will error to avoid silently using the wrong tail.
    ge_keys = {"a_mean", "a_median", "a_lambda", "a_med_lambda"}

    a_keys = sorted(k for k in mc.keys() if k.startswith("a_") and k in data)
    if not a_keys:
        raise ValueError(
            "No shared curvature keys starting with 'a_' found between "
            f"{data_npz} and {mc_npz}."
        )

    unknown = [k for k in a_keys if k not in ge_keys and k != "a_neg_sigma"
               and k != "a_med_neg_sigma" and k != "a_siegel" and k != "a_med_siegel"
               and k != "a_galactic"]

    if unknown:
        raise ValueError(
            "Unknown a_* keys for tail selection: "
            + ", ".join(unknown)
            + ". Update mc_pvalues() tail rules."
        )

    n_trials = int(mc.get("n_trials", len(mc[a_keys[0]])))

    lines = []
    lines.append(f"data_file: {data_npz}")
    lines.append(f"mc_file:   {mc_npz}")
    lines.append(f"n_trials:  {n_trials}")
    lines.append("")
    lines.append("  - Preferred test statistic: a_mean (likeliest to be the most " 
                 "informative single metric).")
    lines.append("  - In general, mean-based parameters are likelier to be useful than " 
    "median-based variants.")
    lines.append("  - Galactic curvature is reported to show that it is not " 
                 "significant.")
    lines.append("")
    lines.append("p-values (one-sided):")
    lines.append("  - p = count / n_MC. All trials count in n_MC; adj. R^2>0 trials " 
    "add to count.")
    lines.append("  - a_mean, a_median, a_lambda, a_med_lambda:  count = #(a_mc >= " 
                 "a_data & R^2>0)")
    lines.append("  - a_neg_sigma, a_med_neg_sigma, a_siegel, a_med_siegel, a_galactic:"
                 "  count = #(a_mc <= a_data & R^2>0)")
    lines.append("")

    # Order: mean/lambda, neg_sigma/siegel, median variants, galactic.
    ordered_keys = []
    for gi, group in enumerate(_MC_PVALUE_ORDER_GROUPS):
        present = [k for k in group if k in a_keys]
        if not present:
            continue
        if ordered_keys:
            ordered_keys.append(None)  # blank line between groups
        ordered_keys.extend(present)

    # Any shared a_* keys not listed above (should not happen for standard outputs).
    listed = {k for g in _MC_PVALUE_ORDER_GROUPS for k in g}
    extras = sorted(k for k in a_keys if k not in listed)
    if extras:
        if ordered_keys:
            ordered_keys.append(None)
        ordered_keys.extend(extras)

    for entry in ordered_keys:
        if entry is None:
            lines.append("")
            continue

        k = entry
        a_data = float(np.asarray(data[k]).squeeze())
        a_mc = np.asarray(mc[k], dtype=float)
        if a_mc.ndim != 1:
            a_mc = a_mc.reshape(-1)

        if a_mc.size != n_trials:
            n_trials = a_mc.size

        r2_key = "R2_" + k[2:]  # a_mean -> R2_mean, a_med_lambda -> R2_med_lambda
        if r2_key not in mc:
            raise ValueError(
                f"MC file {mc_npz} has {k} but no {r2_key}; cannot apply R2>0 cut."
            )

        r2_mc = np.asarray(mc[r2_key], dtype=float)
        if r2_mc.ndim != 1:
            r2_mc = r2_mc.reshape(-1)

        if r2_mc.size != a_mc.size:
            raise ValueError(
                f"Length mismatch: {k} has {a_mc.size} trials, {r2_key} has "
                f"{r2_mc.size}."
            )

        pass_mask = r2_mc > 0
        n_pass_r2 = int(np.sum(pass_mask))
        n_all = int(a_mc.size)

        if k in ge_keys:
            count = int(np.sum((a_mc >= a_data) & pass_mask))
            tail = ">="

        else:
            count = int(np.sum((a_mc <= a_data) & pass_mask))
            tail = "<="

        if n_all > 0:
            p = float(count / n_all)
            lines.append(
                f"{k:16s}  tail {tail}  a_data={a_data: .6e}  p={p:.6g}  "
                f"({count}/{n_all} MC trials; {n_pass_r2} with adj.R^2>0)"
            )

        else:
            lines.append(
                f"{k:16s}  tail {tail}  a_data={a_data: .6e}  p=nan  "
                f"(0 MC trials)"
            )

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    return str(out_txt)


def redo_figures(filepath=None, result_path=None):
    """Load results from a .npz file and regenerate all figures.

    Use this to redo map, multiplet, and parabola figures without re-running the scan.
    Requires the .npz to contain scan results and, for multiplet and parabola figures,
    events and energies (saved by default from super_corr()).

    Parameters
    ----------
    filepath : path-like or None
        Path to the results .npz file. If None, uses default results/super_corr.npz.
    result_path : str or None
        Directory where figure files are written (trailing slash optional).
        If None, uses the parent of filepath, or the default results path when
        filepath is None.
    """
    res = _load_results(filepath)
    if result_path is None:
        if filepath is None:
            _, result_path = _get_project_paths()

        else:
            result_path = str(Path(filepath).resolve().parent) + "/"

    result_path = result_path if result_path.endswith("/") else result_path + "/"

    grid = res['grid']
    ipix = res['ipix']
    nside = res['nside']
    tau = res['tau']
    sigma = res['sigma']
    Dir = res['Dir']
    Dist = res['Dist']
    W = res['W']
    E = res['E']
    SS = res['SS']
    neg_sigma = res['neg_sigma']
    neg_SS = res['neg_SS']
    neg_Dir = res['neg_Dir']

    lambdas = res['lambdas']

    # Map figures only need scan results
    _map_figures(grid, ipix, nside, tau, lambdas, sigma, neg_sigma, SS, neg_SS, Dir,
                neg_Dir, result_path)

    # Multiplet and parabola figures need events and energies
    if 'events' in res and 'energies' in res:
        events = res['events']
        energies = res['energies']
        _multiplet_figure(tau, sigma, grid, events, energies, Dir, Dist, W, E, SS,
                        result_path)
        members = _wedge_members(events, energies, grid, Dir, W, E, Dist)
        _parabola_figures(grid, tau, members, lambdas, neg_sigma, SS, result_path)
        
    else:
        print("Skipping multiplet and parabola figures (events/energies not in file).")

# --- Main block ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Wedge-based supergalactic UHECR correlation scan")

    parser.add_argument( "--figures", nargs="?", const=True, default=False,
        help="Recreate figures from an existing results .npz (optional path argument).")

    parser.add_argument("--mc-trials", type=int, default=0,
        help="Run N isotropic Monte Carlo trials and save per-trial a, y0, and R2 "
             "(no figures, no per-trial scan files).")

    parser.add_argument("--mc-out", type=str, default=None,
        help="Output path for MC statistics .npz (default: results/mc_stats.npz).")

    parser.add_argument("--iso", action="store_true",
        help="Run a single isotropic MC scan (with figures).")

    parser.add_argument("--seed", type=int, default=None,
        help="Seed for isotropic modes: single-run --iso or --mc-trials base seed. "
             "If omitted in --mc-trials mode, true random entropy-based seeding is used.")

    parser.add_argument("--results-dir", type=str, default=None,
        help="Directory for figures and super_corr.npz (default: results/). "
             "Relative paths are under the repository root.")

    parser.add_argument("--no-save", action="store_true",
        help="Do not write results/super_corr.npz for single-run modes.")

    parser.add_argument("--mc-pvalues", action="store_true",
        help="Compute MC p-values from results/super_corr.npz and results/mc_stats.npz "
             "(p = count(tail & R2>0) / n_MC; all trials in denominator) and write "
             "results/mc_pvalues.txt.")

    parser.add_argument("--data-npz", type=str, default=None,
        help="Path to super_corr results .npz for --mc-pvalues (default: "
             "results/super_corr.npz).")

    parser.add_argument("--mc-npz", type=str, default=None,
        help="Path to MC trials .npz for --mc-pvalues (default: results/mc_stats.npz).")

    parser.add_argument("--pvals-out", type=str, default=None,
        help="Output text path for --mc-pvalues (default: results/mc_pvalues.txt).")

    args = parser.parse_args()

    if args.figures:
        npz_path = None if args.figures is True else args.figures
        redo_figures(filepath=npz_path)

    elif args.mc_pvalues:
        mc_pvalues(data_npz=args.data_npz, mc_npz=args.mc_npz, out_txt=args.pvalues_out)

    elif args.mc_trials and args.mc_trials > 0:
        out = run_mc_trials(args.mc_trials, out_path=args.mc_out, seed=args.seed)
        print(f"Saved MC trial statistics to {out}")

    else:
        input_type = "iso" if args.iso else "data"
        super_corr(input_type=input_type, make_figures=True, save_npz=not args.no_save,
            result_path=args.results_dir, seed=args.seed)
