# super_corr -- Supergalactic Structure of Magnetic Deflection Multiplets

Implements the wedge-based (spherical cap section) UHECR energy-angle correlation, 
or multiplet, scan and the supergalactic curvature test statistic, used to test whether 
UHECR multiplets exhibit large-scale supergalactic structure.

The analysis improves upon the methodology from my Telescope Array paper: "Evidence 
for a Supergalactic Structure of Magnetic Deflection Multiplets of Ultra-High Energy
Cosmic Rays" and applies the analysis framework to Pierre Auger Open Data. The goal 
is to test whether the same large-scale supergalactic structure is visible in an
independent data set from the opposite hemisphere.

Further information about the experiment, analysis, and results are shown in the 
included Super_Corr_Auger_Public.pdf and Super_Corr_Auger_Public_Lambda.pdf.

Full Analysis Chain Example (Kendall's tau):
```bash
python src/super_corr.py                  # Run analysis on included data file
python src/super_corr.py --mc-trials 100  # Analyze N isotropic MC based on data
python src/super_corr.py --pvals          # Calculate significance of data result
```

Full Analysis Chain Example (Lundquist's Lambda):
```bash
python src/super_corr.py --stat 'lambda'
python src/super_corr.py --mc-trials 100 --stat 'lambda'
python src/super_corr.py --pvals --stat 'lambda'
```

References:
https://arxiv.org/abs/2005.07312
https://opendata.auger.org/


## Changes from Paper
- Completely new and much faster code.
- Test statistic parabola fit is a rotatable y = a*x^2+c instead of y = a*x^2 + b*x + c.
  This should hopefully further penalize results that do not have minimums near SGB=0.
- Parabola fits use 20 bins instead of 17.
- Parabola bin uncertainties are estimated by sampling event disjoint wedges.
  This takes into account the oversampling nature of the analysis.
- Because we have more proper error bars a 1/sigma^2 weighted least squares parabola fit 
  is done instead of an iteratively reweighted bisquare fit without using error bars.
- The sky scan grid is made from the much more well known Healpix method. This also has
  slightly less variation in the distance between grid points.
- The scan grid is much finer with 1 deg mean separation.
- The wedge shape and energy cut scans are much finer.
- Test statistic is reported in rad^-2 instead of deg^-2 to reduce decimal points.
- Lowest energy cut has been optimized for the Auger Open Data.
- Two correlation statistics are compared: Kendall's tau and my Lambda repeated-average
  rank correlation.

Note: The effect of fitting a rotated y = a*x^2+c instead of y = a*x^2 + b*x + c 
(rotated or not) on the variance of the isotropic MC test statistic should be tested. 
This may not be the best choice. A smaller variance of MC trials is preferred. 

A simple y = a*x^2+c resulted in some absurd results so another degree of freedom is 
necessary.
See: results/lambda/MC4/SuperCorr_MeanTau_NoRotation.png

## Pre-requisite Installations
The library targets Python 3.8+.

```bash
# Commonly included with Anaconda distributions
pip install numba numpy scipy matplotlib astropy tqdm

# Needed for scan grid
pip install healpy

# Necessary Numba compatible correlation libraries created by the author
pip install hyper-corr lambda-corr

# Necessary for MC Trials _iso_MC_Auger.py
pip install utm

# Optional for Numba fast math optimizations on Intel CPUs
pip install icc_rt

# If you want to install super_corr do this in its directory
pip install -e .

```

## Example Runs
```bash
# 1) Default real-data scan
# Uses --stat tau by default and writes to results/tau/
python src/super_corr.py

# Variants:
# Run with Lambda as the primary scan statistic (writes to results/lambda/)
python src/super_corr.py --stat lambda
# Use a iteratively re-weighted bisquare parabola fit a*x^2+c = 0 instead (or LAR or WLS)
python src/super_corr.py --fit-method bisquare

# Custom results directory
python src/super_corr.py --results-dir results/custom_run

# Do not write super_corr.npz (figures are still generated in single-run modes)
python src/super_corr.py --no-save

# Custom results directory with no .npz save
python src/super_corr.py --results-dir results/custom_run --no-save


# 2) Single isotropic Monte-Carlo scan
# Entropy-random isotropic draw (with figures and super_corr.npz)
# Default output directory is auto-incremented: results/tau/MC1, MC2, ...
python src/super_corr.py --iso
# Run N isotropic scans (with figures)
python src/super_corr.py --iso 5
# Lambda-primary isotropic run defaults to results/lambda/MC#
python src/super_corr.py --iso --stat lambda

# Reproduce a specific isotropic draw
# Default output directory for seeded iso run: results/<stat>/MC_<seed>/
python src/super_corr.py --iso --seed 20260325
# Seeded sequence for multiple runs uses seed+i per run
python src/super_corr.py --iso 3 --seed 20260325

# Variants:
# Custom results directory
python src/super_corr.py --iso --results-dir results/iso_run

# Keep figures only (no super_corr.npz)
python src/super_corr.py --iso --no-save

# Custom results directory + no .npz save
python src/super_corr.py --iso --results-dir results/custom_run --no-save


# 3) Regenerate figures from existing .npz
# Uses default results/tau/super_corr.npz, mode=all
python src/super_corr.py --figures
# Use Lambda defaults (results/lambda/super_corr.npz)
python src/super_corr.py --figures --stat lambda

# Mode selection
python src/super_corr.py --figures maps
python src/super_corr.py --figures parabolas

# Use a specific .npz path (output figures go to that file's parent directory)
python src/super_corr.py --figures results/custom_run/super_corr.npz
# You can also pass a results directory (loads <dir>/super_corr.npz)
python src/super_corr.py --figures results/tau/MC1

# Mode + path together
python src/super_corr.py --figures parabolas results/tau/MC1

# Do not update a_*/R2_* values inside super_corr.npz when redrawing parabolas
python src/super_corr.py --figures parabolas results/tau/MC1 --no-update


# 4) Run isotropic MC trials and save trial stats
# No --seed: entropy-random base seed each run
python src/super_corr.py --mc-trials 1000
# Lambda-primary MC trials (writes to results/lambda/mc_trials.npz by default)
python src/super_corr.py --mc-trials 1000 --stat lambda

# Explicit --seed: deterministic/reproducible trials
python src/super_corr.py --mc-trials 1000 --seed 20260325

# Custom MC output file
python src/super_corr.py --mc-trials 1000 --mc-out results/mc_trials_custom.npz

# Reproducible trials + custom MC output file
python src/super_corr.py --mc-trials 1000 --seed 20260325 --mc-out results/mc_trials_custom.npz

# Resume/append behavior:
# If the target MC file already exists (default or --mc-out), new trials are appended.
# The file's stored seed is reused, and trial_seeds continue from the existing trial count.
# A newly supplied --seed is ignored when appending to an existing MC file.


# 5) Replay one significant MC trial with full figures
# Example: extract trial_seeds[i] from mc_trials.npz, then rerun:
python -c "import numpy as np; i=17; d=np.load('results/tau/mc_trials.npz'); print(int(d['trial_seeds'][i]))"
python src/super_corr.py --iso --seed <trial_seed_from_previous_command> --results-dir results/replay_trial_17


# 6) Compute MC p-values from data+MC files
# Default directories (tau)
python src/super_corr.py --pvals
# Lambda defaults
python src/super_corr.py --pvals --stat lambda
# Custom directories for data and MC.
python src/super_corr.py --pvals --data-npz results/tau/super_corr.npz --mc-npz results/tau/mc_trials.npz
# Custom p-value output
python src/super_corr.py --pvals --pvals-out results/pvals_custom.txt
# All custom directories
python src/super_corr.py --pvals --data-npz results/custom_data.npz --mc-npz results/custom_mc.npz --pvals-out results/custom_pvals.txt
```

## Command-Line Flags
Use `python src/super_corr.py --help` for the current CLI help output.

| Flag | Type / Default | What it does |
|---|---|---|
| `--figures [MODE] [NPZ_PATH]` | Optional values / `None` | Recreate figures from existing results. `MODE` is one of `all`, `maps`, `parabolas` (default `all`). `NPZ_PATH` may be a `.npz` file or results directory containing `super_corr.npz`. |
| `--no-update` | flag / `False` | With `--figures all/parabolas`, skip writing refreshed parabola `a_*`, `y0_*`, and `R2_*` values back to `super_corr.npz`. |
| `--mc-trials N` | `int` / `0` | Run `N` isotropic MC trials and save per-trial statistics (`a`, `y0`, `R2`). If output file exists, appends `N` trials to it using the file seed and continuing trial index. |
| `--mc-out PATH` | `str` / `results/<stat>/mc_trials.npz` | Output path for MC trial statistics `.npz` when using `--mc-trials`. |
| `--iso [N]` | optional `int` / `0` | Run isotropic MC scan(s) with figures. `--iso` means 1 run; `--iso N` means N runs. |
| `--seed SEED` | `int` / `None` | Seed for isotropic modes (`--iso`) or base seed for `--mc-trials`. If omitted for MC trials, entropy-based random seeding is used. |
| `--stat {tau,lambda}` | `str` / `tau` | Primary scan statistic. Also selects default result directory `results/<stat>/...` unless an explicit path override is supplied. |
| `--results-dir DIR` | `str` / `results/<stat>/` | Directory for figures and `super_corr.npz` in single-run modes (`data` or `--iso`). |
| `--no-save` | flag / `False` | Do not write `super_corr.npz` in single-run modes (figures can still be generated). |
| `--pvals` | flag / `False` | Compute MC p-values from data and MC `.npz` files; writes `results/<stat>/mc_pvalues.txt` by default. |
| `--data-npz PATH` | `str` / `results/<stat>/super_corr.npz` | Input data `.npz` path for `--pvals`. |
| `--mc-npz PATH` | `str` / `results/<stat>/mc_trials.npz` | Input MC `.npz` path for `--pvals`. |
| `--pvals-out PATH` | `str` / `results/<stat>/mc_pvalues.txt` | Output text path for `--pvals`. |

Notes:
- In `--iso` mode with no `--results-dir`:
  - if `--seed` is omitted, output is written to `results/<stat>/MC#` where `#` auto-increments.
  - if `--seed` is provided, output is written to `results/<stat>/MC_<seed>`.

## Licensing and Data Provenance
The code in this repository is released under the MIT License, while the included Pierre Auger Observatory Open Data is distributed under CC BY-SA 4.0. If you use or redistribute this project, you should preserve the MIT notice for the software and the attribution/share-alike terms that apply to the dataset. Third-party attribution details are summarized in [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md), and the bundled Pierre Auger data license text is provided in [`licenses/PAO_LICENSE.txt`](./licenses/PAO_LICENSE.txt).

## License
Released under the MIT License for software code, alongside third-party data terms noted above. See [LICENSE](./LICENSE), [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md), and [`licenses/PAO_LICENSE.txt`](./licenses/PAO_LICENSE.txt) for details.
