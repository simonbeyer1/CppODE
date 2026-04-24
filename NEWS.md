# CppODE 1.0.0

First tagged release collecting the `devel`-branch work on top of the
initial `master` baseline. The headline additions are a full CVODE(S)
backend with feature parity to the native CppODE integrator, two new
solvers (Tsit5 and MSODA), unbundled SuiteSparse/KLU, and sensitivity
propagation across events via the analytical saltation matrix.

## New features

- **CVODE(S) backend.** A new `CVODE()` front-end that shares the
  `solveODE()` API with `CppODE()`. Supports BDF multistep, dense and
  KLU-sparse Jacobians, time- and root-triggered events with saltation
  corrections, and the full reparametrisation chain rule. Links against
  a system-installed SUNDIALS (`libsundials-dev` and friends) detected
  by `./configure`.
- **Tsit5 solver.** Tsitouras 5(4) explicit Runge–Kutta with embedded
  error estimate and dense output — the fast path for non-stiff
  problems.
- **MSODA solver.** Automatic stiff/non-stiff switching between the
  BDF/NDF multistepper and an Adams/Tsit5 explicit path, modelled on
  LSODA.
- **NDF by default.** The multistep BDF backend now runs Klopfenstein's
  NDF formulas by default; classical BDF remains available via
  `useNDF = FALSE`.
- **Unbundled SuiteSparse/KLU.** Both backends now link against a
  system-installed SuiteSparse (`libsuitesparse-dev` / `suitesparse-devel`
  / `suite-sparse`); `./configure` detects it and falls back gracefully
  if it is missing.
- **Forward-mode AD chain-rule entry** in `funCpp` — new `derivMode`
  flag and `jac_chain` support for reparametrisations.

## Improvements

- Unified initial-step estimator across all solvers, with an explicit
  `HMIN` guard.
- Overhauled multistep step-size controller: QNDF-style safety with the
  CVODE `BIAS` constant, `qwait` gating, and pre-prediction `ewt`.
- Sensitivity-augmented WRMS norm switched to a max-over-vectors
  definition — cleaner convergence behaviour at tight tolerances.
- R sources split into `R/CppODE.R`, `R/CVODE.R`, `R/solveODE.R`,
  `R/funCpp.R`, `R/derivSymb.R`.
- `solveODE()` adopts the CVODE `return_code` scheme, giving both
  backends a single shared flag vocabulary.
- `./configure` no longer rewrites `R/CVODE.R`; the detected SUNDIALS
  configuration is instead written to `inst/cvodeConfig.dcf`.
- Rosenbrock-4 restored to the original 6-stage formulation and error
  estimator.
- Step-trace output returned as an R `data.frame` with named columns
  instead of a flat matrix.
- `funCpp` gained `attach.input` handling in `jac_chain` and
  per-function `srcfile` attributes.
- Reworked equilibrate API: direct threshold check instead of implicit
  root-finding, with a rewritten test suite.

## Bug fixes

- `derivSymb`: name ordering for the derived symbols is now correct.
- CVODE events: `maxroot` handling, `NA` parsing for `time`/`root`
  columns, and batched saltation across multiple simultaneous events.
- Windows installation: broken `configure.win` config file repaired.
- BDF/NDF: Newton-tolerance drift and the MSODA switching flip-flop
  fixed; `qwait` bookkeeping corrected.
- Jacobian refresh path restored; sparse-Jacobian constants gate
  re-enabled after refactor.
- `deriv = FALSE` integrator divergence caused by stale LU data in
  `m_W_temp` — fixed.
- Double-rescale bug in MSODA method switching — fixed.
- Initial step estimator: removed a spurious hub-cap in
  `estimate_initial_dt` that held early steps too small at tight
  tolerances.

## Documentation

- New vignette **`CppODE_CascadeSignaling`** — a 15-state MAPK cascade
  with event-driven ligand dosing, parameter sensitivities across the
  event jump, and a CVODE cross-check.
- Lotka–Volterra vignette: intro tightened and now cross-links to the
  cascade vignette.
- `README.Rmd` gains a dedicated **Vignettes** section listing both
  worked examples.

## Examples and benchmarks

- **`inst/examples/`** consolidated into a focused set: `example_ODE`,
  `example_fun`, `example_events` (time + root events),
  `example_funCpp` (observables and parameter transforms),
  `example_rootfunc` (atmospheric chemistry), `example_equilibrate`
  (steady-state detection), `example_forcings` (driven oscillator),
  `example_bamodel` (bile-acid transport), `example_brusselator_2d`
  (reaction-diffusion PDE via MOL), `example_coupled_vanderpol`.
- **`benchmarks/`** rounded out with `bench_dense`, `bench_fhn`,
  `bench_sparse_threshold`, and `bench_robertson` (moved from the
  former `WorkingExamples/` drop folder). Developer-only probe scripts
  that required custom build flags were dropped.
- All examples and benchmarks now write compiled artefacts to
  `tempdir()` instead of a source-tree `wd/` folder — safe under
  `R CMD check` and non-RStudio sessions.
- `WorkingExamples/` directory removed.
