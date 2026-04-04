#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI driver for distributed isotropic MC (Monte Carlo trials) with super_corr.

Same idea as a standalone MPI+HPC script: mpi4py provides the process grid; each rank
runs its own strided sequence of super_corr(iso) trials and writes a shard .npz.

Example (from repository root, after pip install mpi4py)::

    mpirun -np 32 python HPC/super_corr_mc_mpi.py --seed 21645 --stat tau

Or from ``src/``::

    cd src && mpirun -np 32 python ../HPC/super_corr_mc_mpi.py --seed 21645

Equivalent: ``cd src && mpirun -np 32 python super_corr.py --mc-shard --seed 21645``.

Merge shard files with ``super_corr.py --mc-merge-shards ...`` then ``--pvals``.
"""

from mpi4py import MPI
import argparse
import socket
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mc_trials import run_mc_trials_shard  # noqa: E402


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"rank {rank}/{size} on {socket.gethostname()}", flush=True)

    p = argparse.ArgumentParser(
        description="MPI distributed MC shards for super_corr (mpirun -np N ...).")
    p.add_argument("--mc-out", type=str, default=None,
                   help="Base path for shard .npz (per-rank suffix added if size>1).")
    p.add_argument("--seed", type=int, default=None,
                   help="Base seed; required when size > 1 so all ranks agree.")
    p.add_argument("--stat", type=str, default="tau", choices=("tau", "lambda"))
    p.add_argument("--fit-method", type=str, default="rotated",
                   choices=("wls", "lad", "bisquare", "rotated"))
    args = p.parse_args()

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


if __name__ == "__main__":
    main()
