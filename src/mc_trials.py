#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monte Carlo trial runs, MPI shard merge, and MC p-values for super_corr."""

import numpy as np
from tqdm import tqdm
from pathlib import Path

from _super_io import _get_project_paths, _load_results, _normalize_stat


def _mc_trial_key_groups(stat):
    stat = _normalize_stat(stat)
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
    return common_keys, stat_keys, wrong_stat_key


def _resolve_mc_shard_out_path(out_path, node_rank, num_nodes):
    """If several nodes share one basename, write distinct files (mc_trials.nodeR.npz)."""
    out_path = Path(out_path)
    if num_nodes <= 1:
        return out_path
    stem, suf = out_path.stem, out_path.suffix
    if stem.endswith(f".node{node_rank}"):
        return out_path
    return out_path.with_name(f"{stem}.node{node_rank}{suf}")


def run_mc_trials(n_trials, out_path=None, seed=None, stat='tau', fit_method='rotated'):
    """Run isotropic MC trials and save per-trial a, y0, and R2.

    All saved arrays are float32 for compact storage.
    """
    from super_corr import super_corr

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

    common_keys, stat_keys, wrong_stat_key = _mc_trial_key_groups(stat)
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
        res = super_corr(input_type='iso', make_figures=False, save_npz=False,
                         seed=int(trial_seeds[i]), stat=stat,
                         fit_method=fit_method)

        for k in common_keys:
            arr[k][i] = res[k]

        for k in stat_keys:
            arr[k][i] = res[k]

        n_done = i + 1
        np.savez_compressed(out_path, **_build_out(n_done))

    if n_add == 0:
        np.savez_compressed(out_path, **_build_out(old_n))

    return str(out_path)


def run_mc_trials_shard(
    out_path=None,
    seed=None,
    stat="tau",
    fit_method="rotated",
    node_rank=0,
    num_nodes=1,
):
    """Distributed MC: one OS process owns strided global trial indices; checkpoints each trial.

    Each trial calls ``super_corr(iso)`` (wedge scan uses threads on that process).

    Global trial index ``i`` uses RNG seed ``base_seed + i`` (uint32 wrap). Rank ``r``
    of ``N`` runs ``i = r, r+N, r+2N, ...``. With ``num_nodes > 1``, use the same
    ``seed`` everywhere. For MPI under ``mpirun``/``srun``, set ``r`` and ``N`` from
    ``MPI.COMM_WORLD.Get_rank()`` and ``Get_size()`` (``super_corr.py --mc-shard`` or
    ``HPC/super_corr_mc_mpi.py``).

    Writes a shard ``.npz`` (``trial_global_indices``). Merge with
    ``merge_mc_trial_shards`` before ``mc_pvalues``. Runs until the process exits;
    does not return in normal use.
    """
    from super_corr import super_corr

    stat = _normalize_stat(stat)
    num_nodes = int(num_nodes)
    node_rank = int(node_rank)
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1.")
    if not (0 <= node_rank < num_nodes):
        raise ValueError("node_rank must satisfy 0 <= node_rank < num_nodes.")

    _, result_path = _get_project_paths(stat=stat)
    if out_path is None:
        out_path = str(Path(result_path) / "mc_trials.npz")
    out_path = _resolve_mc_shard_out_path(Path(out_path), node_rank, num_nodes)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    common_keys, stat_keys, wrong_stat_key = _mc_trial_key_groups(stat)
    all_keys = common_keys + stat_keys
    dt = np.float32

    trial_indices_list = []
    row_vals = {k: [] for k in all_keys}
    base_seed = None

    if out_path.is_file():
        old = dict(np.load(out_path, allow_pickle=True))
        if "seed" not in old:
            raise ValueError(f"Existing shard missing seed: {out_path}")
        if wrong_stat_key in old:
            raise ValueError(
                f"Existing shard {out_path} appears to be for a different stat."
            )
        base_seed = int(np.asarray(old["seed"]).item())
        if seed is not None and int(seed) != base_seed:
            print(
                f"Note: ignoring provided seed={int(seed)}; continuing shard "
                f"seed={base_seed}."
            )
        if "trial_global_indices" not in old:
            raise ValueError(f"Existing shard missing trial_global_indices: {out_path}")
        tgi = np.asarray(old["trial_global_indices"], dtype=np.uint32).reshape(-1)
        for k in all_keys:
            if k not in old:
                raise ValueError(f"Existing shard missing key {k}: {out_path}")
            a = np.asarray(old[k], dtype=dt).reshape(-1)
            if a.size != tgi.size:
                raise ValueError(
                    f"Shard key {k} length {a.size} != trial_global_indices {tgi.size}."
                )
            row_vals[k] = list(a)
        trial_indices_list = list(tgi)
        if "mc_node_rank" in old:
            if int(np.asarray(old["mc_node_rank"]).item()) != node_rank:
                raise ValueError(
                    f"Shard {out_path} mc_node_rank does not match node_rank={node_rank}."
                )
        if "mc_num_nodes" in old:
            if int(np.asarray(old["mc_num_nodes"]).item()) != num_nodes:
                raise ValueError(
                    f"Shard {out_path} mc_num_nodes does not match num_nodes={num_nodes}."
                )
    else:
        if num_nodes > 1 and seed is None:
            raise ValueError(
                "With num_nodes > 1, pass an explicit seed so every node agrees."
            )
        if seed is None:
            base_seed = int(np.random.default_rng().integers(0, 2**32, dtype=np.uint32))
        else:
            base_seed = int(seed)

    def _next_global_index():
        if not trial_indices_list:
            return node_rank
        return int(trial_indices_list[-1]) + num_nodes

    def _save_shard():
        tgi = np.array(trial_indices_list, dtype=np.uint32)
        ts = (np.int64(base_seed) + tgi.astype(np.int64)).astype(np.uint32)
        out = dict(
            n_trials=int(tgi.size),
            seed=base_seed,
            trial_global_indices=tgi,
            trial_seeds=ts,
            mc_node_rank=np.int32(node_rank),
            mc_num_nodes=np.int32(num_nodes),
            stat=np.array(stat),
            fit_method=np.array(fit_method),
        )
        for k in all_keys:
            out[k] = np.array(row_vals[k], dtype=dt)
        np.savez_compressed(out_path, **out)

    print(
        f"MC shard ({stat}): node_rank={node_rank}/{num_nodes}, "
        f"seed={base_seed}, out={out_path}"
    )

    pbar = tqdm(smoothing=0.2, unit="trial")
    try:
        while True:
            i = _next_global_index()
            ts_i = int((np.int64(base_seed) + np.int64(i)) % (2**32))
            res = super_corr(
                input_type="iso",
                make_figures=False,
                save_npz=False,
                seed=ts_i,
                stat=stat,
                fit_method=fit_method,
            )
            trial_indices_list.append(i)
            for k in all_keys:
                row_vals[k].append(np.float32(res[k]))
            _save_shard()
            pbar.update(1)
    finally:
        pbar.close()


def merge_mc_trial_shards(in_paths, out_path, stat=None):
    """Merge per-node shard ``.npz`` files into one ``mc_trials.npz`` for ``mc_pvalues``."""
    paths = [Path(p) for p in in_paths]
    if not paths:
        raise ValueError("No shard paths given.")
    stat = _normalize_stat(stat) if stat is not None else None
    common_keys, stat_keys, wrong_stat_key = None, None, None
    chunks = []

    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
        d = dict(np.load(p, allow_pickle=True))
        if "trial_global_indices" not in d:
            raise ValueError(f"Not an MC shard (no trial_global_indices): {p}")
        if wrong_stat_key is None:
            st = str(np.asarray(d.get("stat", "")).item())
            if st not in ("tau", "lambda"):
                raise ValueError(f"Shard {p} missing or invalid stat metadata.")
            if stat is not None and st != stat:
                raise ValueError(f"Shard stat={st} does not match expected {stat}.")
            common_keys, stat_keys, wrong_stat_key = _mc_trial_key_groups(st)
            stat = st
        st = str(np.asarray(d.get("stat", "")).item())
        if st != stat:
            raise ValueError(f"Mixed stat in shards: {p} has {st}, expected {stat}.")
        if wrong_stat_key in d:
            raise ValueError(f"Shard {p} looks like wrong stat key set.")
        seed = int(np.asarray(d["seed"]).item())
        tgi = np.asarray(d["trial_global_indices"], dtype=np.uint32).reshape(-1)
        chunks.append((seed, tgi, d))

    seeds = {c[0] for c in chunks}
    if len(seeds) != 1:
        raise ValueError(f"Shards must share one base seed; got seeds={seeds}")
    base_seed = seeds.pop()

    all_idx = []
    for seed, tgi, d in chunks:
        all_idx.append(tgi)
    cat_idx = np.concatenate(all_idx, axis=0)
    if cat_idx.size != np.unique(cat_idx).size:
        raise ValueError("Duplicate global trial indices across shards.")
    order = np.argsort(cat_idx, kind="mergesort")
    cat_idx = cat_idx[order]
    n_total = int(cat_idx.size)
    trial_seeds = (np.int64(base_seed) + cat_idx.astype(np.int64)).astype(np.uint32)

    all_keys = common_keys + stat_keys
    out_arrays = {}
    for k in all_keys:
        parts = []
        for seed, tgi, d in chunks:
            parts.append(np.asarray(d[k], dtype=np.float32).reshape(-1))
        cat = np.concatenate(parts, axis=0)
        if cat.size != cat_idx.size:
            raise ValueError(f"Key {k}: length mismatch vs merged indices.")
        out_arrays[k] = cat[order]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(
        n_trials=n_total,
        seed=base_seed,
        trial_seeds=trial_seeds,
    )
    out.update(out_arrays)
    np.savez_compressed(out_path, **out)
    return str(out_path)


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

    ordered_keys = []
    for gi, group in enumerate(order_groups):
        present = [k for k in group if k in a_keys]
        if not present:
            continue
        if ordered_keys:
            ordered_keys.append(None)
        ordered_keys.extend(present)

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

        r2_key = "R2_" + k[2:]

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
        xlim = (-lim, lim)
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
