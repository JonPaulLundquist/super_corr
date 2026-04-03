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
from _super_io import (_get_project_paths, _load_input_events, _save_results,
                       _next_mc_dir)
from _super_io import _load_results
from _map_figures import _multiplet_figure, _map_figures
from _wedge import _wedge_members, _siegel_slopes, _lambda_and_siegel, _tau_and_siegel
from _parabolas import _parabola_figures, _parabola_stats

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


def run_mc_trials(n_trials, out_path=None, seed=None, stat='tau',
                  fit_method='rotated'):
    """Run isotropic MC trials and save per-trial a, y0, and R2.

    All saved arrays are float32 for compact storage.
    """
    stat = _normalize_stat(stat)
    _, result_path = _get_project_paths(stat=stat)
    if out_path is None:
        out_path = str(Path(result_path) / "mc_trials.npz")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_add = int(n_trials)
    if n_add < 0:
        raise ValueError("n_trials must be >= 0.")

    dt = np.float32

    common_keys = [
        "a_mean", "a_median", "a_neg_sigma", "a_med_neg_sigma", "a_siegel",
        "a_med_siegel", "a_galactic", "y0_mean", "y0_median", "y0_neg_sigma",
        "y0_med_neg_sigma", "y0_siegel", "y0_med_siegel", "y0_galactic",
        "R2_mean", "R2_median", "R2_neg_sigma", "R2_med_neg_sigma", "R2_siegel",
        "R2_med_siegel", "R2_galactic",
    ]
    if stat == "tau":
        stat_keys = ["a_lambda", "a_med_lambda", "y0_lambda", "y0_med_lambda",
                     "R2_lambda", "R2_med_lambda"]
        wrong_stat_key = "a_tau"
    else:
        stat_keys = ["a_tau", "a_med_tau", "y0_tau", "y0_med_tau",
                     "R2_tau", "R2_med_tau"]
        wrong_stat_key = "a_lambda"
    all_keys = common_keys + stat_keys

    old_n = 0
    old = None
    if out_path.is_file():
        old = dict(np.load(out_path, allow_pickle=True))
        if "seed" not in old:
            raise ValueError(f"Existing MC file missing seed: {out_path}")
        if wrong_stat_key in old:
            raise ValueError(
                f"Existing MC file {out_path} appears to be for a different stat."
            )
        base_seed = int(np.asarray(old["seed"]).item())
        if seed is not None and int(seed) != base_seed:
            print(
                f"Note: ignoring provided seed={int(seed)}; continuing with "
                f"seed={base_seed} from existing file."
            )
        if "trial_seeds" not in old:
            raise ValueError(f"Existing MC file missing trial_seeds: {out_path}")
        old_n = int(np.asarray(old.get("n_trials", len(old["trial_seeds"]))).item())
    else:
        if seed is None:
            # True random seeding (from OS entropy) for distributed/HPC workflows.
            base_seed = int(np.random.default_rng().integers(0, 2**32, dtype=np.uint32))
        else:
            base_seed = int(seed)

    n_total = old_n + n_add
    trial_seeds = (base_seed + np.arange(n_total, dtype=np.uint32)).astype(np.uint32)

    arr = {}
    for k in all_keys:
        a = np.empty(n_total, dtype=dt)
        if old is not None:
            if k not in old:
                raise ValueError(f"Existing MC file missing required key: {k}")
            old_arr = np.asarray(old[k], dtype=dt).reshape(-1)
            if old_arr.size < old_n:
                raise ValueError(
                    f"Existing MC file key {k} shorter than n_trials ({old_n})."
                )
            a[:old_n] = old_arr[:old_n]
        arr[k] = a

    def _build_out(n_done):
        out = dict(n_trials=n_done, seed=base_seed, trial_seeds=trial_seeds[:n_done])
        out.update({k: arr[k][:n_done] for k in common_keys})
        out.update({k: arr[k][:n_done] for k in stat_keys})
        return out

    mode = "append" if old_n > 0 else "new"
    print(
        f"MC trials ({stat}) {mode}: old_n={old_n}, add_n={n_add}, "
        f"new_total={n_total}, seed={base_seed}, out={out_path}"
    )

    for i in tqdm(range(old_n, n_total), total=n_add, smoothing=0.2):
        # Headless, do-not-save scan.
        res = super_corr(input_type='iso', make_figures=False, save_npz=False,
                         seed=int(trial_seeds[i]), stat=stat,
                         fit_method=fit_method)

        for k in common_keys:
            arr[k][i] = res[k]

        # Save the total results to a .npz file after each trial in case of HPC preemption
        # or other interruption.
        for k in stat_keys:
            arr[k][i] = res[k]

        n_done = i + 1
        np.savez_compressed(out_path, **_build_out(n_done))

    # If no new trials were requested, keep/initialize a consistent output file.
    if n_add == 0:
        np.savez_compressed(out_path, **_build_out(old_n))

    return str(out_path)


# Display order for mc_pvalues() output (blank line between groups).
def _normalize_stat(stat):
    stat = str(stat).strip().lower()
    if stat not in {"tau", "lambda"}:
        raise ValueError(f"Invalid statistic: {stat}")
    return stat


def _mc_pvalue_config(stat):
    stat = _normalize_stat(stat)
    if stat == "tau":
        ge_keys = {"a_mean", "a_median", "a_lambda", "a_med_lambda"}
        order_groups = (
            ("a_mean", "a_lambda"),
            ("a_neg_sigma", "a_siegel"),
            ("a_median", "a_med_lambda", "a_med_neg_sigma", "a_med_siegel"),
            ("a_galactic",),
        )
    else:
        ge_keys = {"a_mean", "a_median", "a_tau", "a_med_tau"}
        order_groups = (
            ("a_mean", "a_tau"),
            ("a_neg_sigma", "a_siegel"),
            ("a_median", "a_med_tau", "a_med_neg_sigma", "a_med_siegel"),
            ("a_galactic",),
        )
    return ge_keys, order_groups


def mc_pvalues(data_npz=None, mc_npz=None, out_txt=None, stat='tau'):
    """Compute one-sided MC p-values for each curvature parameter a_*.

    Denominator is every MC trial. The extreme-event count includes a trial only if
    the tail is satisfied and R^2 > 0 for the matching parabola (same suffix as a_*,
    e.g. a_mean <-> R2_mean). Trials with R^2 <= 0 still count toward the denominator
    but not toward the numerator.

    Tail directions (numerator uses tail & R2>0):
      - a_mean, a_median, a_lambda/tau, a_med_lambda/tau: count #(a_mc >= a_data & R2>0)
      - others (currently 5)                     : count #(a_mc <= a_data & R2>0)
    """
    stat = _normalize_stat(stat)
    _, result_path = _get_project_paths(stat=stat)
    if data_npz is None:
        data_npz = Path(result_path) / "super_corr.npz"

    else:
        data_npz = Path(data_npz)

    if mc_npz is None:
        mc_npz = Path(result_path) / "mc_trials.npz"

    else:
        mc_npz = Path(mc_npz)

    if out_txt is None:
        out_txt = Path(result_path) / "mc_pvalues.txt"

    else:
        out_txt = Path(out_txt)

    data = _load_results(data_npz)
    mc = dict(np.load(mc_npz, allow_pickle=True))
    trial_seeds_mc = np.asarray(mc.get("trial_seeds", []), dtype=np.uint32).reshape(-1)

    ge_keys, order_groups = _mc_pvalue_config(stat)

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

    n_trials_raw = int(mc.get("n_trials", len(mc[a_keys[0]])))

    # Build a per-trial signature from all a_* and R2_* arrays and keep only
    # unique samples. This guards against duplicated trials when combining runs
    # done in different places/times (with or without seeds).
    sig_keys = sorted(
        k for k in mc.keys()
        if (k.startswith("a_") or k.startswith("R2_"))
        and np.asarray(mc[k]).reshape(-1).size >= n_trials_raw
    )
    unique_mask = np.ones(n_trials_raw, dtype=bool)
    duplicate_summary = None
    if sig_keys and n_trials_raw > 0:
        sig_mat = np.empty((n_trials_raw, len(sig_keys)), dtype=np.float64)
        for j, k in enumerate(sig_keys):
            sig_mat[:, j] = np.asarray(mc[k], dtype=np.float64).reshape(-1)[:n_trials_raw]
        row_bytes = np.ascontiguousarray(sig_mat).view(
            np.dtype((np.void, sig_mat.dtype.itemsize * sig_mat.shape[1]))
        ).reshape(-1)
        _, unique_idx = np.unique(row_bytes, return_index=True)
        unique_idx = np.sort(unique_idx)
        if unique_idx.size < n_trials_raw:
            unique_mask[:] = False
            unique_mask[unique_idx] = True
            duplicate_summary = (
                f"Warning: detected {n_trials_raw - unique_idx.size} duplicated MC "
                f"samples ({n_trials_raw} total, {unique_idx.size} unique). "
                "Using unique samples only for p-value calculations."
            )
    n_trials = int(np.sum(unique_mask))

    lines = []
    lines.append(f"data_file: {data_npz}")
    lines.append(f"mc_file:   {mc_npz}")
    lines.append(f"n_trials:  {n_trials}")
    lines.append(f"stat:      {stat}")
    if duplicate_summary is not None:
        lines.append(duplicate_summary)
        print(duplicate_summary)
    lines.append("")
    lines.append("  - Preferred test statistic: a_mean (likeliest to be the most " 
                 "informative single metric).")
    lines.append("  - In general, mean-based parameters are likelier to be useful than " 
                 "median-based variants.")
    lines.append("    The median has large error bars that are difficult to "
                 "calculate accurately due to the oversampling nature of the analysis.")
    lines.append("  - Galactic curvature is reported to show that it is not " 
                 "significant.")
    lines.append("")
    deg2_factor = (180.0 / np.pi) ** 2

    lines.append("p-values (one-sided):")
    lines.append("  - p = count / n_MC. All trials count in n_MC; adj. R^2>0 trials " 
    "add to count.")
    lines.append("  - Reported a_data and a_mc values are in rad^-2.")
    if stat == "tau":
        lines.append("  - a_mean, a_median, a_lambda, a_med_lambda:  count = "
                     "#(a_mc >= a_data & R^2>0)")
    else:
        lines.append("  - a_mean, a_median, a_tau, a_med_tau:  count = "
                     "#(a_mc >= a_data & R^2>0)")
    lines.append("  - a_neg_sigma, a_med_neg_sigma, a_siegel, a_med_siegel, a_galactic:"
                 "  count = #(a_mc <= a_data & R^2>0)")
    lines.append("  - All passing MC trials are listed as (a_mc, seed).")
    lines.append("")   
    lines.append("Full run with: python src/super_corr.py --iso --seed <seed> for "
                 "all variables and figures.")
    lines.append("")

    # Order: mean/lambda, neg_sigma/siegel, median variants, galactic.
    ordered_keys = []
    for gi, group in enumerate(order_groups):
        present = [k for k in group if k in a_keys]
        if not present:
            continue
        if ordered_keys:
            ordered_keys.append(None)  # blank line between groups
        ordered_keys.extend(present)

    # Any shared a_* keys not listed above (should not happen for standard outputs).
    listed = {k for g in order_groups for k in g}
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
        if a_mc.size < n_trials_raw:
            raise ValueError(
                f"MC file {mc_npz} has {a_mc.size} entries for {k}, expected at least "
                f"{n_trials_raw}."
            )
        a_mc = a_mc[:n_trials_raw][unique_mask]

        r2_key = "R2_" + k[2:]  # a_mean -> R2_mean, a_med_lambda -> R2_med_lambda

        r2_mc = np.asarray(mc[r2_key], dtype=float)
        if r2_mc.ndim != 1:
            r2_mc = r2_mc.reshape(-1)
        if r2_mc.size < n_trials_raw:
            raise ValueError(
                f"MC file {mc_npz} has {r2_mc.size} entries for {r2_key}, expected at "
                f"least {n_trials_raw}."
            )
        r2_mc = r2_mc[:n_trials_raw][unique_mask]
        if r2_mc.size != a_mc.size:
            raise ValueError(
                f"Length mismatch: {k} has {a_mc.size} trials, {r2_key} has "
                f"{r2_mc.size}."
            )

        pass_mask = r2_mc > 0
        n_pass_r2 = int(np.sum(pass_mask))
        n_all = int(a_mc.size)

        if k in ge_keys:
            tail_mask = (a_mc >= a_data) & pass_mask
            count = int(np.sum(tail_mask))
            tail = ">="

        else:
            tail_mask = (a_mc <= a_data) & pass_mask
            count = int(np.sum(tail_mask))
            tail = "<="

        idx_tail = np.where(tail_mask)[0]

        if n_all > 0:
            p = float(count / n_all)
            a_data_rad2 = a_data * deg2_factor
            lines.append(
                f"{k:16s}  tail {tail}  a_data={a_data_rad2: .6e}  p={p:.6g}  "
                f"({count}/{n_all} MC trials; {n_pass_r2} with adj.R^2>0)"
            )
            if count > 0 and trial_seeds_mc.size >= n_trials_raw:
                trial_seeds_used = trial_seeds_mc[:n_trials_raw][unique_mask]
                if k in ge_keys:
                    order = np.argsort(-a_mc[idx_tail])
                else:
                    order = np.argsort(a_mc[idx_tail])
                sorted_idx = idx_tail[order]
                lines.append("  tail_trials:")
                for j in sorted_idx:
                    lines.append(
                        f"    (a_mc = {a_mc[j] * deg2_factor: .6e}, seed = "
                        f"{int(trial_seeds_used[j])})"
                    )

        else:
            a_data_rad2 = a_data * deg2_factor
            lines.append(
                f"{k:16s}  tail {tail}  a_data={a_data_rad2: .6e}  p=nan  "
                f"(0 MC trials)"
            )

    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # Histogram of MC a_mean in rad^-2 with data value marker.
    import matplotlib.pyplot as plt

    a_data_mean_rad2 = data["a_mean"] * deg2_factor
    a_mc_mean_rad2 = (
        np.asarray(mc["a_mean"], dtype=float).reshape(-1)[:n_trials_raw][unique_mask]
        * deg2_factor
    )
    a_mc_mean_rad2 = a_mc_mean_rad2[np.isfinite(a_mc_mean_rad2)]

    if a_mc_mean_rad2.size > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        counts, _, _ = ax.hist(
            a_mc_mean_rad2, bins=20, color="blue", alpha=0.8,
            edgecolor="black", linewidth=0.7,
            label=r"MC $\mathbf{a}$  [$\mathbf{rad}^{-2}$]"
        )
        
        all_a_rad2 = np.concatenate((a_mc_mean_rad2, np.atleast_1d(a_data_mean_rad2)))
        lo = np.floor(np.min(all_a_rad2) / 0.1) * 0.1
        hi = np.ceil(np.max(all_a_rad2) / 0.1) * 0.1
        lim = max(abs(lo), abs(hi))
        xlim = (-lim,lim)
        ax.set_xlim(xlim)
        y_max = int(np.max(counts))
        y_top = int(np.ceil(max(y_max, 1) / 10.0) * 10)
        ax.set_ylim(0, y_top)
        a_data_label = f"{a_data_mean_rad2:.3f}"
        ax.axvline(a_data_mean_rad2, color="red", linewidth=2.0,
                   label=rf"Data $\mathbf{{a}}=\mathbf{{{a_data_label}}}\ "
                         rf"\mathbf{{rad}}^{{-2}}$")

        ax.set_xlabel(r"$\mathbf{a}$  [$\mathbf{rad}^{-2}$]",
                      fontweight="semibold", size=18)
        ax.set_ylabel("Count", fontweight="semibold", size=18)
        ax.set_title(r"MC Trials Test Statistic '$\mathbf{a}$'",
                     y=1.04, fontweight="semibold", size=18)

        for tlabel in ax.get_xticklabels() + ax.get_yticklabels():
            tlabel.set_fontweight("semibold")
            tlabel.set_fontsize(14)

        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

        ax.legend(prop={"weight": "semibold", "size": 15})

        hist_file = out_txt.with_name("a_mean_hist.png")
        fig.savefig(hist_file, dpi=600)
        plt.close(fig)
        lines.append("")
        lines.append(f"Saved histogram: {hist_file}")

    out_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return str(out_txt)


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

    parser.add_argument("--iso", nargs="?", const=1, default=0, type=int,
        help="Run isotropic MC scan(s) with figures. Optional value sets number "
             "of runs (e.g., --iso 5).")

    parser.add_argument("--seed", type=int, default=None,
        help="Seed for isotropic modes: single-run --iso or --mc-trials base seed. "
             "If omitted in --mc-trials mode, true random entropy-based seeding is used.")

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
