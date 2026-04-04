#!/bin/bash
# Slurm + OpenMPI launcher for distributed isotropic MC (super_corr --mc-shard).

#SBATCH --account=owner-guest
#SBATCH --partition=kingspeak-guest
#SBATCH --job-name=super_corr_mc
#SBATCH --time=10:00:00
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=32G
#SBATCH --output=super_corr_mc-%j.out
#SBATCH --error=super_corr_mc-%j.err
##SBATCH --mail-type=FAIL,END
##SBATCH --mail-user=jplundquist@utah.edu
#SBATCH --exclude=kp292

set -euo pipefail

# --- Repository on the cluster path ---
SUPER_CORR_ROOT=/uufs/chpc.utah.edu/common/home/u0446071/super_corr

# Required when SLURM_NTASKS > 1 so all MPI ranks share one MC ensemble
#SEED="${SEED:?Set SEED for reproducible distributed MC}"
SEED=21645

STAT="${STAT:-tau}"
FIT_METHOD="${FIT_METHOD:-rotated}"
MC_OUT="${MC_OUT:-}"

module purge
module load deeplearning/2025.4
module load openmpi

m=$((SLURM_CPUS_PER_TASK * 2))
export NUMBA_NUM_THREADS="$m"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export OMP_PROC_BIND=false
export OMP_PLACES=threads

export APPTAINER_CACHEDIR=/scratch/general/vast/$USER/apptainer_cache
mkdir -p "$APPTAINER_CACHEDIR"

unset TMPDIR
export PRTE_MCA_prte_tmpdir_base=/tmp
export PRTE_MCA_prte_silence_shared_fs=1

SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-/uufs/chpc.utah.edu/sys/installdir/r8/deeplearning/2025.4/deeplearning_2025.4.sif}"

echo "JOBID=$SLURM_JOB_ID"
echo "NodeList=$SLURM_JOB_NODELIST"
echo "SUPER_CORR_ROOT=$SUPER_CORR_ROOT SEED=$SEED STAT=$STAT"
srun -n "$SLURM_NTASKS" hostname

cd "$SUPER_CORR_ROOT/src"

MC_ARGS=(--mc-shard --seed "$SEED" --stat "$STAT" --fit-method "$FIT_METHOD")
if [[ -n "$MC_OUT" ]]; then
  MC_ARGS+=(--mc-out "$MC_OUT")
fi

# One MPI rank per node; each rank runs super_corr with threaded wedge scan on that node.
mpirun -np "$SLURM_NTASKS" --bind-to none \
  singularity exec "$SINGULARITY_IMAGE" \
  python -u super_corr.py "${MC_ARGS[@]}"
