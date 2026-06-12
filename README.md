# PDE with Random Coefficient - FEM solver and MLMC estimator

Repository for experimenting with 1D finite element discretizations of PDEs with random coefficients and Monte Carlo / MLMC estimators. 
The code is minimal, exploratory, and intended for research/teaching.

## Overview

This project demonstrates generating Gaussian random fields with a Matern-like covariance via an
FEM discretization of the SPDE $(Id - k^{-2} \Delta ) u = W'$, using those fields as coefficients in
a second-order elliptic PDE, solving the PDE with a weighted stiffness matrix, and estimating
statistics (e.g. expected L^2 norm of the PDE solution) using Monte Carlo. The code is 1D and
designed for experiments and clarity rather than production performance.

## Repository structure

- [pde_random_coefficient.py](pde_random_coefficient.py) — driver for Monte Carlo experiments; builds samples,
  runs the FEM solve, computes norms, and saves histograms / convergence plots in `results/`.
- [auxiliary_fem.py](auxiliary_fem.py) — minimal 1D FEM utilities: basis functions, local/global mass and stiffness
  matrices, weighted stiffness assembly, and numeric RHS assembly (`rhs`).
- [random_field_generator.py](random_field_generator.py) — constructs Gaussian random field samples solving
  (M + k^{-2} K) u = b where `b` is assembled from element-wise Gaussian samples with covariance `M_e`.
- [test_0.py](test_0.py), [test_1.py](test_1.py), [test_2.py](test_2.py) — example scripts and sanity checks
  that exercise the FEM routines and deterministic/weighted PDE solver.
- [requirements.txt](requirements.txt) — Python packages used by the project.
- [results/](results/) — output directory where plots and text results (e.g. monte_carlo_estimation_results.txt)
  are written.

## Installation

1. Create and activate a Python environment. Example using the included `venv1` or a new `venv`:

```bash
# using existing venv1 on Windows (PowerShell)
. venv1/Scripts/Activate.ps1

# or create a fresh virtualenv
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # macOS / Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Notes: the repository was developed with NumPy, Matplotlib and (optionally) SymPy for tests.

## Usage

- Run the Monte Carlo driver (long):

```bash
python pde_random_coefficient.py
```

This script sets a seed, constructs a domain mesh, runs `monte_carlo_estimation` with `num_samples`,
and writes results and plots to `results/`.

- Run individual tests / examples:

```bash
python test_0.py   # basic rhs + mass matrix sanity check
python test_1.py   # solves -u'' + u = f with known exact solution; convergence plot
python test_2.py   # deterministic weighted-PDE test and comparison with exact q(x)
python random_field_generator.py  # quick visualization of random-field samples
```

## Key functions and expectations

- `auxiliary_fem.py`:
  - `phi_i`, `phi_i_prime`, `phi_i_0`, `phi_i_N`, etc. — basis functions and derivatives for 1D linear elements.
  - `local_mass_matrix(h)`, `local_stiffness_matrix(h)` — element matrices.
  - `global_mass_matrix(xl)`, `global_stiffness_matrix(xl)` — assembled global matrices for node vector `xl`.
  - `weighted_stiffness_matrix(xl, w)` — assembles stiffness with per-element weight `w` (length `len(xl)-1`).
  - `rhs(f, xx, n_nodes=100)` — numeric quadrature to assemble load vector for function `f` on nodes `xx`.

- `random_field_generator.py`:
  - `generate_random_field(xl, k, g0, gN)` — samples a Gaussian vector `b` with element covariances `M_e`,
    solves `(M + k^{-2} K) u = b` for interior nodes, and returns `u` with Dirichlet BCs `g0,gN` applied.
    Parameters:
    - `xl`: node coordinates (1D array)
    - `k`: correlation length parameter (affects operator M + k^{-2} K)
    - `g0`, `gN`: Dirichlet boundary values (defaults used in scripts are 0)

- `pde_random_coefficient.py`:
  - `solve_pde_problem(x_G, x_D, f, g0, gN, k)` — workflow:
    1. call `generate_random_field(x_D, k, g0, gN)` to get `u(x)` on `x_D`;
    2. truncate/interpolate `u` to `x_G` domain region [-0.5,0.5] used for PDE coefficient;
    3. form weights `exp(u)` and assemble weighted stiffness `A` on `x_G`;
    4. solve for `q` and compute L^2 norm `sqrt(q^T M q)` using `global_mass_matrix(x_G)`.
    Inputs:
    - `x_G`: node coordinates for the PDE solve (domain of interest)
    - `x_D`: node coordinates passed to the random-field generator (usually a larger domain)
    - `f`: right-hand side callable `f(x)`
    - `g0`, `gN`: Dirichlet BCs values
    - `k`: parameter forwarded to the random field generator

  - `monte_carlo_estimation(num_samples, x_G, X_D, f, g0, gN, k)` — runs `solve_pde_problem` repeatedly and
    returns the array of L^2 norms. `pde_random_coefficient.py` then plots a histogram and the running mean.

## Outputs

- `results/monte_carlo_estimation_results.txt` — appended summary lines from Monte Carlo runs.
- PNG files in `results/` created by `pde_random_coefficient.py`:
  - histogram of estimates;
  - convergence plot of running mean;
  - single-sample PDE solution plot.

## Reproducibility & configuration

- `pde_random_coefficient.py` sets `seed = 2026` and calls `np.random.seed(seed)` for reproducible runs.
- Key parameters to change in the driver:
  - `num_samples` (Monte Carlo sample count)
  - `x_D`, `x_G` node vectors
  - `k` (random-field correlation parameter)

## TODO and open questions (from code comments)

- Refactor and clarify function input parameters and mesh conventions.
- Implement a multilevel Monte Carlo (MLMC) estimator for variance reduction.
- Add unit tests for small matrix assembly routines.
- Validate choices for `k`, covariance scaling, and boundary treatment with domain experts.
- Assess performance: compare different sampling strategies for `b ~ N(0, M)`.

## Development notes

- The code is intentionally explicit and uses simple NumPy linear solvers (`np.linalg.solve`). For larger
  experiments consider switching to sparse matrices and solvers (SciPy `sparse` + `spsolve`) and vectorized
  sampling techniques.
- For production/large-scale Monte Carlo, profile the random-field generator, consider reuse of factorizations,
  and implement multi-threading / job distribution.

## Running CI / tests

There is no CI configured. To run the example scripts manually use the `python` commands listed above.

## Contact

Author: repository code in this workspace (research/teaching code). For questions open an issue or contact the
owner of the repository.
