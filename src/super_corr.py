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
import os
import gc
from pathlib import Path

from _scan_driver import _run_scan, _unpack_results, _prepare_scan_inputs
from _scan_params import _build_scan_params, _build_scan_grid
from _super_io import (_get_project_paths, _load_input_events, _save_results,
                       _next_mc_dir, _normalize_stat)
from _super_io import _load_results
from _map_figures import _multiplet_figure, _map_figures
from _wedge import _wedge_members, _siegel_slopes, _lambda_and_siegel, _tau_and_siegel
from _parabolas import _parabola_figures, _parabola_stats
from mc_trials import (merge_mc_trial_shards, mc_pvalues, run_mc_trials,
                       run_mc_trials_shard) 

# --- Main analysis ---

def super_corr(input_type='data', make_figures=True, save_npz=True, result_path=None,
               seed=None, stat='tau', fit_method='rotated', show_figures=True):
    # Determine the number of workers for multithreading the wedge scan
    num_workers = int(np.round(os.cpu_count()))
    
    # Build scan parameters and grid.
    minN, min_Ecut, distances, widths, directions, Ecuts = _build_scan_params(stat=stat)
    grid, ipix, nside = _build_scan_grid()
    data_path, default_result_path = _get_project_paths(stat=stat)
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
    results = _run_scan(grid.size, event_mask, energies, separations, azimuthal_list, 
                        directions, Ecuts, widths, distances, minN, num_workers, 
                        stat=stat)
    
    # Unpack the scan results. neg_* variables are for the negative tau/Lambda only scan.
    corr, sigma, p_value, E, Dir, Dist, W, N, neg_corr, neg_sigma, neg_p, neg_E, \
        neg_Dir, neg_Dist, neg_W, neg_N, scans = _unpack_results(results, grid.size)
    
    # Build wedge members for parabola fits uncertainty estimation.
    members = _wedge_members(events, energies, grid, Dir, W, E, Dist)
    
    if stat == 'tau':
        corr2, SS = _lambda_and_siegel(events, energies, grid, Dir, W, E, Dist)

    elif stat == 'lambda':
        corr2, SS = _tau_and_siegel(events, energies, grid, Dir, W, E, Dist)

    else:
        raise ValueError(f"Invalid statistic: {stat}")
    
    # Calculate the siegel slopes for the negative tau/Lambda only scan.
    neg_SS = _siegel_slopes(events, energies, grid, neg_Dir, neg_W, neg_E, neg_Dist)
    
    if make_figures:
        # Create the best multiplet figure
        _multiplet_figure(corr, sigma, grid, events, energies, Dir, Dist, W, E, SS,
                          result_path, stat=stat, show=show_figures)

        # Create the projection map figures
        _map_figures(grid, ipix, nside, corr, corr2, sigma, neg_sigma, SS, neg_SS, Dir,
                     neg_Dir, result_path, stat=stat, show=show_figures)

        (popt_mean, R2_mean, popt_median, R2_median, popt_corr2, R2_corr2,
            popt_med_corr2, R2_med_corr2, popt_neg_sigma, R2_neg_sigma,
            popt_med_neg_sigma, R2_med_neg_sigma, popt_siegel, R2_siegel,
            popt_med_siegel, R2_med_siegel, popt_galactic, R2_galactic,
        ) = _parabola_figures(grid, corr, members, corr2, neg_sigma, SS, result_path, 
                              stat=stat, fit_method=fit_method,
                              show=show_figures)

    else:
        (popt_mean, R2_mean, popt_median, R2_median, popt_corr2, R2_corr2,
            popt_med_corr2, R2_med_corr2, popt_neg_sigma, R2_neg_sigma,
            popt_med_neg_sigma, R2_med_neg_sigma, popt_siegel, R2_siegel,
            popt_med_siegel, R2_med_siegel, popt_galactic, R2_galactic,
        ) = _parabola_stats(grid, corr, members, corr2, neg_sigma, SS,
                            fit_method=fit_method)

    a_mean, y0_mean = popt_mean
    a_median, y0_median = popt_median
    a_corr2, y0_corr2 = popt_corr2
    a_med_corr2, y0_med_corr2 = popt_med_corr2
    a_neg_sigma, y0_neg_sigma = popt_neg_sigma
    a_med_neg_sigma, y0_med_neg_sigma = popt_med_neg_sigma
    a_siegel, y0_siegel = popt_siegel
    a_med_siegel, y0_med_siegel = popt_med_siegel
    a_galactic, y0_galactic = popt_galactic

    if stat == 'tau':
        results_out = dict(tau=corr, sigma=sigma, p_value=p_value, E=E, Dir=Dir,
            Dist=Dist, W=W, N=N, scans=scans, neg_tau=neg_corr, neg_sigma=neg_sigma,
            neg_p=neg_p, neg_E=neg_E, neg_Dir=neg_Dir, neg_Dist=neg_Dist, neg_W=neg_W,
            neg_N=neg_N, grid=grid, ipix=ipix, nside=nside, events=events,
            energies=energies, lambdas=corr2, SS=SS, neg_SS=neg_SS, a_mean=a_mean,
            y0_mean=y0_mean, a_median=a_median, y0_median=y0_median, a_lambda=a_corr2,
            y0_lambda=y0_corr2, a_med_lambda=a_med_corr2, y0_med_lambda=y0_med_corr2,
            a_neg_sigma=a_neg_sigma, y0_neg_sigma=y0_neg_sigma,
            a_med_neg_sigma=a_med_neg_sigma, y0_med_neg_sigma=y0_med_neg_sigma,
            a_siegel=a_siegel, y0_siegel=y0_siegel, a_med_siegel=a_med_siegel,
            y0_med_siegel=y0_med_siegel, a_galactic=a_galactic, y0_galactic=y0_galactic,
            R2_mean=R2_mean, R2_median=R2_median, R2_lambda=R2_corr2,
            R2_med_lambda=R2_med_corr2, R2_neg_sigma=R2_neg_sigma,
            R2_med_neg_sigma=R2_med_neg_sigma, R2_siegel=R2_siegel,
            R2_med_siegel=R2_med_siegel, R2_galactic=R2_galactic, stat=stat)

    elif stat == 'lambda':
        results_out = dict(**{"lambda": corr}, sigma=sigma, p_value=p_value, E=E,
            Dir=Dir, Dist=Dist, W=W, N=N, scans=scans, neg_lambda=neg_corr,
            neg_sigma=neg_sigma, neg_p=neg_p, neg_E=neg_E, neg_Dir=neg_Dir,
            neg_Dist=neg_Dist, neg_W=neg_W, neg_N=neg_N, grid=grid, ipix=ipix,
            nside=nside, events=events, energies=energies, taus=corr2, SS=SS,
            neg_SS=neg_SS, a_mean=a_mean, y0_mean=y0_mean, a_median=a_median,
            y0_median=y0_median, a_tau=a_corr2, y0_tau=y0_corr2, a_med_tau=a_med_corr2,
            y0_med_tau=y0_med_corr2, a_neg_sigma=a_neg_sigma, y0_neg_sigma=y0_neg_sigma,
            a_med_neg_sigma=a_med_neg_sigma, y0_med_neg_sigma=y0_med_neg_sigma,
            a_siegel=a_siegel, y0_siegel=y0_siegel, a_med_siegel=a_med_siegel,
            y0_med_siegel=y0_med_siegel, a_galactic=a_galactic, y0_galactic=y0_galactic,
            R2_mean=R2_mean, R2_median=R2_median, R2_tau=R2_corr2,
            R2_med_tau=R2_med_corr2, R2_neg_sigma=R2_neg_sigma,
            R2_med_neg_sigma=R2_med_neg_sigma, R2_siegel=R2_siegel,
            R2_med_siegel=R2_med_siegel, R2_galactic=R2_galactic, stat=stat)

    else:
        raise ValueError(f"Invalid statistic: {stat}")

    if save_npz:
        _save_results(result_path, **results_out)

    # --- Garbage collection necessary for threads and memory management ---
    for _ in range(3):
        gc.collect()

    return results_out


def redo_figures(filepath=None, result_path=None, stat='tau', fit_method='rotated',
                 figure_mode='all', update_npz=True, show_figures=True):
    """Load results from a .npz file and regenerate selected figures.

    Use this to redo map, multiplet, and parabola figures without re-running the scan.
    Requires the .npz to contain scan results and, for multiplet and parabola figures,
    events and energies (saved by default from super_corr()).

    Parameters
    ----------
    filepath : path-like or None
        Path to the results .npz file. If None, uses default
        results/<stat>/super_corr.npz.
    stat : {'tau', 'lambda'}
        Statistic used only when filepath is None to pick the default results folder.
        If filepath is provided, the saved file's own `stat` value is used for labels
        and key selection.
    result_path : str or None
        Directory where figure files are written (trailing slash optional).
        If None, uses the parent of filepath, or the default results path when
        filepath is None.
    figure_mode : {'all', 'maps', 'parabolas'}
        Figure subset to regenerate.
        - all: maps + multiplet + parabola figures
        - maps: map figures only
        - parabolas: multiplet + parabola figures only
    update_npz : bool
        If True, writes refreshed parabola a_*, y0_*, and R2_* values back to the
        source super_corr.npz (when parabola figures are generated).
    """
    figure_mode = str(figure_mode).strip().lower()
    if figure_mode not in {'all', 'maps', 'parabolas'}:
        raise ValueError(
            f"Invalid figure_mode={figure_mode!r}. Expected 'all', 'maps', or "
            "'parabolas'."
        )
    # Resolve the exact .npz path so refreshed parabola metrics can be written back.
    if filepath is None:
        _, default_result_path = _get_project_paths(stat=stat)
        npz_path = Path(default_result_path) / "super_corr.npz"
    else:
        path_in = Path(filepath)
        candidates = []
        if path_in.is_dir():
            candidates.append(path_in / "super_corr.npz")
        else:
            candidates.append(path_in)
            if path_in.suffix == "":
                candidates.append(path_in.with_suffix(".npz"))
                candidates.append(path_in / "super_corr.npz")
        npz_path = next((p for p in candidates if p.is_file()), None)
        if npz_path is None:
            tried = ", ".join(str(p) for p in candidates)
            raise FileNotFoundError(f"Results file not found. Tried: {tried}")

    res = _load_results(npz_path, stat=stat)
    if result_path is None:
        result_path = str(npz_path.resolve().parent) + "/"

    result_path = result_path if result_path.endswith("/") else result_path + "/"

    stat = _normalize_stat(np.asarray(res['stat']).item())
    grid = res['grid']
    ipix = res['ipix']
    nside = res['nside']
    sigma = res['sigma']
    Dir = res['Dir']
    Dist = res['Dist']
    W = res['W']
    E = res['E']
    SS = res['SS']
    neg_sigma = res['neg_sigma']
    neg_SS = res['neg_SS']
    neg_Dir = res['neg_Dir']

    if stat == 'tau':
        corr = res['tau']
        corr2 = res['lambdas']

    elif stat == 'lambda':
        corr = res['lambda']
        corr2 = res['taus']

    else:
        raise ValueError(f"Invalid statistic: {stat}    ")

    # Map figures only need scan results
    if figure_mode in {'all', 'maps'}:
        _map_figures(grid, ipix, nside, corr, corr2, sigma, neg_sigma, SS, neg_SS, Dir,
                     neg_Dir, result_path, stat=stat, show=show_figures)

    # Multiplet and parabola figures need events and energies
    if figure_mode in {'all', 'parabolas'}:
        events = res['events']
        energies = res['energies']
        _multiplet_figure(corr, sigma, grid, events, energies, Dir, Dist, W, E, SS,
                          result_path, stat=stat, show=show_figures)
        members = _wedge_members(events, energies, grid, Dir, W, E, Dist)
        (
            popt_mean, R2_mean, popt_median, R2_median, popt_corr2, R2_corr2,
            popt_med_corr2, R2_med_corr2, popt_neg_sigma, R2_neg_sigma,
            popt_med_neg_sigma, R2_med_neg_sigma, popt_siegel, R2_siegel,
            popt_med_siegel, R2_med_siegel, popt_galactic, R2_galactic,
        ) = _parabola_figures(
            grid, corr, members, corr2, neg_sigma, SS, result_path,
            stat=stat, fit_method=fit_method, show=show_figures
        )

        a_mean, y0_mean = popt_mean
        a_median, y0_median = popt_median
        a_corr2, y0_corr2 = popt_corr2
        a_med_corr2, y0_med_corr2 = popt_med_corr2
        a_neg_sigma, y0_neg_sigma = popt_neg_sigma
        a_med_neg_sigma, y0_med_neg_sigma = popt_med_neg_sigma
        a_siegel, y0_siegel = popt_siegel
        a_med_siegel, y0_med_siegel = popt_med_siegel
        a_galactic, y0_galactic = popt_galactic

        # Refresh curvature and adjusted R^2 values in-place in super_corr.npz.
        res.update(
            a_mean=a_mean, y0_mean=y0_mean,
            a_median=a_median, y0_median=y0_median,
            a_neg_sigma=a_neg_sigma, y0_neg_sigma=y0_neg_sigma,
            a_med_neg_sigma=a_med_neg_sigma, y0_med_neg_sigma=y0_med_neg_sigma,
            a_siegel=a_siegel, y0_siegel=y0_siegel,
            a_med_siegel=a_med_siegel, y0_med_siegel=y0_med_siegel,
            a_galactic=a_galactic, y0_galactic=y0_galactic,
            R2_mean=R2_mean, R2_median=R2_median,
            R2_neg_sigma=R2_neg_sigma, R2_med_neg_sigma=R2_med_neg_sigma,
            R2_siegel=R2_siegel, R2_med_siegel=R2_med_siegel,
            R2_galactic=R2_galactic,
        )
        if stat == 'tau':
            res.update(
                a_lambda=a_corr2, y0_lambda=y0_corr2,
                a_med_lambda=a_med_corr2, y0_med_lambda=y0_med_corr2,
                R2_lambda=R2_corr2, R2_med_lambda=R2_med_corr2,
            )
        elif stat == 'lambda':
            res.update(
                a_tau=a_corr2, y0_tau=y0_corr2,
                a_med_tau=a_med_corr2, y0_med_tau=y0_med_corr2,
                R2_tau=R2_corr2, R2_med_tau=R2_med_corr2,
            )

        # Keep a normalized stat tag and persist all keys back to the same file.
        if update_npz:
            res['stat'] = stat
            np.savez(npz_path, **res)
            print(f"Updated parabola a_* and R2_* values in {npz_path}")
        else:
            print("Skipped updating super_corr.npz (--no-update).")

# --- Main block ---
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Wedge-based supergalactic UHECR correlation scan")

    parser.add_argument("--figures", nargs="*", default=None,
        help="Recreate figures from an existing results .npz. "
             "Optional values: mode ({all,maps,parabolas}) and/or NPZ_PATH "
             "(file or directory). Examples: --figures, --figures maps, "
             "--figures parabolas results/tau/MC1")

    parser.add_argument("--no-update", action="store_true",
        help="With --figures all/parabolas, do not update parabola a_*/R2_* values "
             "inside super_corr.npz.")

    parser.add_argument("--mc-trials", type=int, default=0,
        help="Run N isotropic Monte Carlo trials and save per-trial a, y0, and R2 "
             "(no figures, no per-trial scan files).")

    parser.add_argument("--mc-out", type=str, default=None,
        help="Output path for MC statistics .npz "
             "(default: results/<stat>/mc_trials.npz).")

    parser.add_argument("--mc-shard", action="store_true",
        help="MPI distributed MC (requires mpi4py): launch with mpirun/srun. Uses "
             "MPI.COMM_WORLD rank/size for trial striding; each rank runs super_corr "
             "in a loop with per-trial shard checkpoints. Use the same --seed on all "
             "ranks. Merge with --mc-merge-shards. Not with --mc-trials.")

    parser.add_argument("--mc-merge-shards", nargs="+", default=None, metavar="NPZ",
        help="Merge shard .npz files into one mc_trials archive; requires --mc-out.")

    parser.add_argument("--iso", nargs="?", const=1, default=0, type=int,
        help="Run isotropic MC scan(s) with figures. Optional value sets number "
             "of runs (e.g., --iso 5).")

    parser.add_argument("--seed", type=int, default=None,
        help="Seed for isotropic modes: --iso, --mc-trials base seed, or --mc-shard "
             "(required when MPI size > 1). If omitted in --mc-trials, OS entropy is used.")

    parser.add_argument("--stat", type=str, default="tau", choices=("tau", "lambda"),
        help="Primary scan statistic to use. "
             "Kendall's tau (default) or Lundquist's Lambda correlation coefficient.")

    parser.add_argument("--results-dir", type=str, default=None,
        help="Directory for figures and super_corr.npz (default: results/<stat>/). "
             "Relative paths are under the repository root.")

    parser.add_argument("--fit-method", type=str, default="rotated",
        choices=("wls", "lad", "bisquare", "rotated"),
        help="Parabola fit model used for curvature metrics and figures.")

    parser.add_argument("--no-save", action="store_true",
        help="Do not write results/super_corr.npz for single-run modes.")
    parser.add_argument("--no-show", action="store_true",
        help="Save figures without displaying interactive windows.")

    parser.add_argument("--pvals", dest="run_pvals", action="store_true",
        help="Compute MC p-values from results/<stat>/super_corr.npz and "
             "results/<stat>/mc_trials.npz "
             "(p = count(tail & R2>0) / n_MC; all trials in denominator) and write "
             "results/<stat>/mc_pvalues.txt.")

    parser.add_argument("--data-npz", type=str, default=None,
        help="Path to super_corr results .npz for --pvals (default: "
             "results/<stat>/super_corr.npz).")

    parser.add_argument("--mc-npz", type=str, default=None,
        help="Path to MC trials .npz for --pvals "
             "(default: results/<stat>/mc_trials.npz).")

    parser.add_argument("--pvals-out", dest="pvals_out", type=str, default=None,
        help="Output text path for --pvals "
             "(default: results/<stat>/mc_pvalues.txt).")

    args = parser.parse_args()

    if args.figures is not None:
        figure_mode = "all"
        npz_path = None
        mode_tokens = {"all", "maps", "parabolas"}
        for token in args.figures:
            token_s = str(token).strip()
            token_l = token_s.lower()
            if token_l in mode_tokens:
                figure_mode = token_l
            elif npz_path is None:
                npz_path = token_s
            else:
                raise ValueError(
                    "Too many values for --figures. Provide at most one mode "
                    "({all,maps,parabolas}) and one NPZ_PATH."
                )

        redo_figures(
            filepath=npz_path,
            stat=args.stat,
            fit_method=args.fit_method,
            figure_mode=figure_mode,
            update_npz=not args.no_update,
            show_figures=not args.no_show,
        )
        if args.no_show:
            plt.close("all")

    elif args.run_pvals:
        mc_pvalues(data_npz=args.data_npz, mc_npz=args.mc_npz,
                   out_txt=args.pvals_out, stat=args.stat)

    elif args.mc_merge_shards is not None:
        if args.mc_out is None:
            raise ValueError("--mc-merge-shards requires --mc-out (merged .npz path).")
        out = merge_mc_trial_shards(
            args.mc_merge_shards, args.mc_out, stat=args.stat)
        print(f"Merged MC trial shards to {out}")

    elif args.mc_shard:
        if args.mc_trials and args.mc_trials > 0:
            raise ValueError("Use either --mc-trials or --mc-shard, not both.")
        try:
            from mpi4py import MPI
        except ImportError as e:
            raise ImportError(
                "Distributed MC (--mc-shard) requires mpi4py. "
                "Install with: pip install mpi4py"
            ) from e
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if rank == 0:
            print(f"super_corr MPI MC: COMM_WORLD size = {size}", flush=True)
        comm.Barrier()
        run_mc_trials_shard(
            out_path=args.mc_out,
            seed=args.seed,
            stat=args.stat,
            fit_method=args.fit_method,
            node_rank=rank,
            num_nodes=size,
        )

    elif args.mc_trials and args.mc_trials > 0:
        out = run_mc_trials(args.mc_trials, out_path=args.mc_out, seed=args.seed,
                            stat=args.stat, fit_method=args.fit_method)
        print(f"Saved MC trial statistics to {out}")

    else:
        if args.iso:
            import matplotlib.pyplot as plt

            n_iso = int(args.iso)
            if n_iso <= 0:
                raise ValueError("--iso count must be >= 1.")

            for i in range(n_iso):
                run_seed = None if args.seed is None else int(args.seed) + i
                if args.results_dir is None:
                    run_results_dir = _next_mc_dir(args.stat, seed=run_seed)
                elif n_iso == 1:
                    run_results_dir = args.results_dir
                else:
                    run_results_dir = (
                        f"{str(args.results_dir).rstrip('/')}/run_{i+1:03d}"
                    )

                print(
                    f"Running isotropic scan {i+1}/{n_iso}: "
                    f"seed={run_seed}, out={run_results_dir}"
                )
                super_corr(
                    input_type="iso",
                    make_figures=True,
                    save_npz=not args.no_save,
                    result_path=run_results_dir,
                    seed=run_seed,
                    stat=args.stat,
                    fit_method=args.fit_method,
                    show_figures=not args.no_show,
                )
                if n_iso > 1:
                    plt.close("all")
        else:
            super_corr(input_type="data", make_figures=True,
                       save_npz=not args.no_save, result_path=args.results_dir,
                       seed=args.seed, stat=args.stat,
                       fit_method=args.fit_method,
                       show_figures=not args.no_show)
