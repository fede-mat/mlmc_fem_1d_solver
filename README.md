# PDE with Random Coefficient — 1D FEM + Monte Carlo / MLMC experiments

This repository provides minimal, experimental code for 1D finite-element discretizations of PDEs
with random coefficients and example Monte Carlo (MC) and Multilevel Monte Carlo (MLMC) estimators.
It is intended for research and teaching: clarity and explicitness are prioritised over performance.

## Summary

- Builds Gaussian random fields with a Matern-like covariance by solving the FEM discretization of
  (I - k^{-2} Δ) u = W (white noise forcing).
- Uses generated fields as log-coefficients for a weighted elliptic PDE
  - (exp(u(x,ω)) q'(x,ω))' = f(x) on the subdomain x ∈ (-0.5, 0.5) with Dirichlet BCs.
- Provides example drivers for Monte Carlo (`pde_random_coefficient_mc.py`) and
  Multilevel Monte Carlo (`pde_random_coefficient_mlmc.py`) estimators that compute statistics
  (e.g. expected L^2 norm of q) and save plots / summaries in `results/`.

## Files of interest

- `auxiliary_fem.py` — 1D FEM utilities (basis functions, local/global mass & stiffness assembly,
  `rhs` quadrature, `weighted_stiffness_matrix`, etc.).
- `random_field_generator.py` — sample Gaussian random fields via solving `(M + k^{-2}K) u = b` with
  element-wise sampling of `b ~ N(0, M_e)`.
- `pde_random_coefficient_mc.py` — Monte Carlo driver:
  - `function_f(x)` returns the RHS (default ones);
  - `solve_pde_problem(x_G, x_D, f, g0, gN, k)` generates a random field on `x_D`, truncates to
    [-0.5,0.5], assembles weighted stiffness on `x_G`, solves for `q`, and returns `q` and its L^2 norm;
  - `monte_carlo_estimation(num_samples, x_G, X_D, f, g0, gN, k)` runs `solve_pde_problem` repeatedly
    and returns the norms array; the script saves histogram and convergence plots in `results/`.
- `pde_random_coefficient_mlmc.py` — MLMC driver implementing a simple telescoping estimator using
  coupled random-field samples from `random_field_generator.generate_random_field_mlmc`.
- `test/` — small example scripts (e.g. `test_0.py`) that exercise FEM routines and plotting.
- `requirements.txt` — minimal package list used during development.
- `results/` — output directory with generated PNGs and text summaries.

## Requirements

Install dependencies (recommended inside a virtualenv):

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Required packages (from `requirements.txt`): `numpy`, `scipy`, `matplotlib`, `sympy` (optional for some tests).

## Quick usage

Run the Monte Carlo driver (example):

```bash
python pde_random_coefficient_mc.py
```

Run the MLMC driver (example):

```bash
python pde_random_coefficient_mlmc.py
```

Notes:
- Both drivers set `seed = 2026` and call `np.random.seed(seed)` for reproducible runs in the current code.
- Default parameters (mesh and sample counts) are defined inside the scripts; edit `num_samples`, `x_D`, `x_G`,
  and `k` in the drivers for different experiments.

## Output and plots

- Text results are appended to `results/monte_carlo_estimation_results.txt` and
  `results/multilevel_monte_carlo_estimation_results.txt` by the respective drivers.
- Example PNG outputs saved to `results/` include:
  - `histogram_mc_estimates_seed_<seed>_samples_<n>.png`
  - `convergence_mc_seed_<seed>_samples_<n>.png`
  - `convergence_mlmc_seed_<seed>_samples_<n>.png`
  - `solution_pde_sample.png`

## Tests and examples

Run the simple examples in `test/` to visualise basis functions, mass matrix solves, and other
sanity checks. Example:

```bash
python test/test_0.py
```

There is no CI configured; tests are informal plotting / sanity scripts.

## Development notes & suggestions

- Code style: explicit NumPy arrays and `np.linalg.solve` are used for clarity. For larger-scale
  experiments consider converting assembly to sparse matrices (`scipy.sparse`) and using `spsolve`.
- Performance: the random-field generator samples element-wise normals and assembles `b` each run.
  Reusing factorizations or caching operators can speed repeated solves.
- Numerical details: the scripts use simple quadrature (`rhs`) and linear Lagrange elements; verify
  mesh and quadrature choices for accuracy when changing the target PDE or domain.

## Reproducibility

- The drivers set a fixed seed (`seed = 2026`) for reproducible demonstrations. Change or expose the seed
  via a command-line argument if you need different runs.

## Contact / Attribution

This is research/teaching code in the local workspace. For questions, open an issue in the associated
repository or contact the code owner.

---
If you'd like, I can also:
- add a short usage CLI (argparse) to the MC/MLMC drivers,
- run the example scripts here and capture runtime errors (if any), or
- add a small CONTRIBUTING or DEVELOPMENT section describing how to extend the code.
