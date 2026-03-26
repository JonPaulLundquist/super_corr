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

References:
https://arxiv.org/abs/2005.07312
https://opendata.auger.org/


## Changes from Paper
- Completely new and much faster code.
- The test statistic parabola fit is y = a*x^2+c instead of y = a*x^2 + b*x + c.
  This further penalizes results that do not have minimums near SGB=0.
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
python src/super_corr.py

# Variants:
# Custom results directory
python src/super_corr.py --results-dir results/custom_run

# Do not write super_corr.npz (figures are still generated in single-run modes)
python src/super_corr.py --no-save

# Custom results directory with no .npz save
python src/super_corr.py --results-dir results/custom_run --no-save


# 2) Single isotropic Monte-Carlo scan
# Entropy-random isotropic draw (with figures and super_corr.npz)
python src/super_corr.py --iso

# Reproduce a specific isotropic draw
python src/super_corr.py --iso --seed 20260325

# Variants:
# Custom results directory
python src/super_corr.py --iso --results-dir results/iso_run

# Keep figures only (no super_corr.npz)
python src/super_corr.py --iso --no-save

# Custom results directory + no .npz save
python src/super_corr.py --iso --results-dir results/custom_run --no-save


# 3) Regenerate figures from existing .npz
# Uses default results/super_corr.npz
python src/super_corr.py --figures

# Use a specific .npz path (output figures go to that file's parent directory)
python src/super_corr.py --figures results/custom_run/super_corr.npz


# 4) Run isotropic MC trials and save trial stats
# No --seed: entropy-random base seed each run
python src/super_corr.py --mc-trials 1000

# Explicit --seed: deterministic/reproducible trials
python src/super_corr.py --mc-trials 1000 --seed 20260325

# Custom MC output file
python src/super_corr.py --mc-trials 1000 --mc-out results/mc_stats_custom.npz

# Reproducible trials + custom MC output file
python src/super_corr.py --mc-trials 1000 --seed 20260325 --mc-out results/mc_stats_custom.npz


# 5) Replay one significant MC trial with full figures
# Example: extract trial_seeds[i] from mc_stats.npz, then rerun:
python -c "import numpy as np; i=17; d=np.load('results/mc_stats.npz'); print(int(d['trial_seeds'][i]))"
python src/super_corr.py --iso --seed <trial_seed_from_previous_command> --results-dir results/replay_trial_17


# 6) Compute MC p-values from data+MC files
# Default directories
python src/super_corr.py --mc-pvalues
# Custom directories for data and MC.
python src/super_corr.py --mc-pvalues --data-npz results/super_corr.npz --mc-npz results/mc_stats.npz
# Custom p-value output
python src/super_corr.py --mc-pvalues --pvals-out results/pvals_custom.txt
# All custom directories
python src/super_corr.py --mc-pvalues --data-npz results/custom_data.npz --mc-npz results/custom_mc.npz --pvals-out results/custom_pvals.txt
```

## Licensing and Data Provenance
The code in this repository is released under the MIT License, while the included Pierre Auger Observatory Open Data is distributed under CC BY-SA 4.0. If you use or redistribute this project, you should preserve the MIT notice for the software and the attribution/share-alike terms that apply to the dataset. Third-party attribution details are summarized in [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md), and the bundled Pierre Auger data license text is provided in [`licenses/PAO_LICENSE.txt`](./licenses/PAO_LICENSE.txt).

## License
Released under the MIT License for software code, alongside third-party data terms noted above. See [LICENSE](./LICENSE), [THIRD_PARTY_LICENSES.md](./THIRD_PARTY_LICENSES.md), and [`licenses/PAO_LICENSE.txt`](./licenses/PAO_LICENSE.txt) for details.
