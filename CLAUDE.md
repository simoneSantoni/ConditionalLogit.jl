# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

ConditionalLogit.jl fits conditional logistic regression by **exact** conditional maximum likelihood. Its design goal is to reproduce SAS `PROC LOGISTIC` with a `STRATA` statement number for number: the same starting values (β = 0), the Newton–Raphson stopping rule (GCONV = 1e-8, reporting the iterate that met the criterion without taking another step), fit statistics, global tests, odds-ratio tables and SAS-style printed output. It is also validated against R `survival::clogit(method = "exact")` and Stata `clogit`. Behaviour that seems odd (such as the default `gconv`, the `singular` tolerance, or `sc_n`) usually follows a SAS default, and changing it breaks the golden tests.

## Commands

```bash
julia --project -e 'using Pkg; Pkg.test()'                      # full suite (no R needed)
julia --project -e 'using Pkg; Pkg.test(julia_args=`-t 4`)'     # multithreaded; thread-determinism tests need >1 thread
julia --project=docs -e 'using Pkg; Pkg.instantiate()'          # once
julia --project=docs docs/make.jl                               # docs → docs/build/
Rscript test/reference/generate_reference.R                     # regenerate R fixtures (needs survival, jsonlite)
julia --project=benchmark -t 8 benchmark/bench.jl [choice mixed unequal bigstrata offset stringids]
julia --project=benchmark benchmark/run_threads.jl mixed        # thread sweep + bitwise-identity check
```

The test-only dependencies (CSV, DataFrames, JSON, ForwardDiff, ...) are in `[extras]`/`[targets]` of `Project.toml`, not in a `test/Project.toml`. To run one test file, use TestEnv.jl (not installed by default) so that those dependencies resolve, then include `test/helpers.jl` before the file:

```julia
using TestEnv; TestEnv.activate()
include("test/helpers.jl"); include("test/test_sas.jl")
```

The benchmark env uses `[sources]` (Julia ≥ 1.11). On 1.10, run `julia --project=benchmark -e 'using Pkg; Pkg.develop(path = ".")'` once first.

### Cross-software comparison (`benchmark/compare/`)

`scenarios.jl` writes the synthetic data to `benchmark/compare/data/`. That folder is git-ignored (about 580 MB), so run `scenarios.jl` first. Then `julia_bench.jl` (once with `-t 1` and once with `-t 8`), `r_bench.R`, `py_bench.py` (uses `benchmark/compare/.venv`, see its header) and `stata_bench.do` each write `results/*.csv`. `report.jl` merges these files into the tables in `docs/src/performance.md`. Each script's header gives its exact invocation.

### Stata manual validation (`validation/`)

`stata_manual.R` downloads the data. Stata running `stata_manual.do` writes the full-precision `data/stata_full.csv`. Then `julia --project=docs validation/stata_manual.jl` checks the results against the Stata [R] clogit manual examples.

## Architecture

The pipeline runs in include order (`src/ConditionalLogit.jl`):

1. **`frontend.jl`**: the formula/Tables and matrix entry points (`clogit`, `fit(ConditionalLogitModel, ...)`). It does listwise deletion, event-level coding (the `event` keyword), FREQ/offset handling, multi-column strata keys and option validation. It converts everything to dense stratum `codes` and calls `fit_core`.
2. **`fit.jl`**: `fit_core` is the type-stable core shared by both front ends. It builds the design, runs the optimizer and assembles a `ConditionalLogitModel` (`types.jl`). `fitted` is computed lazily (`compute_fitted`).
3. **`design.jl`**: `StratifiedDesign` stores **only informative strata** (0 < m_k < n_k), each as a contiguous column range of the transposed `Xt` (p × N). It **centres covariates within stratum** (FREQ-weighted). It also runs the in-order collinearity sweep that sets the `aliased` coefficients.
4. **`likelihood.jl`**: the per-stratum kernels. A stratum with m = 1 uses a two-pass centred softmax. Otherwise the kernel is the Gail–Lubin–Rubinstein recursion in normalised log-space "mixture" form (carrying log B, E[x_S], Var[x_S]). It runs on the complement when m > n − m. A single huge stratum takes `_glr_split!`, which splits the Hessian by column blocks over tasks. `Evaluator`/`evaluate!` farms strata out to tasks in data-determined chunks (most expensive first) and reduces the partial sums in a fixed order. `loglik_generic`/`logB_generic` are generic-`Real` reference versions, used with ForwardDiff/BigFloat in the tests.
5. **`optimize.jl`**: Newton–Raphson with the SAS criteria (GCONV/FCONV/ABSFCONV/XCONV), step-halving and separation diagnostics.
6. **`inference.jl`** (global tests, fit statistics, odds ratios), **`statsapi.jl`** (StatsAPI/StatsModels methods, `predict`) and **`show.jl`** (the SAS-layout text report). `ResultTable` subtypes are Tables.jl tables.

Invariants to preserve when editing kernels or threading:
- **Bitwise-identical results for any `ntasks`/thread count.** Chunking must depend only on the data, and reductions must run in a fixed order. `test_threads.jl` and `run_threads.jl` check this.
- The information matrix must be PSD by construction: updates are convex combinations plus rank-one PSD terms. Don't introduce subtractive formulas.
- The kernels in `Workspace` are allocation-free. Fits must stay safe to run concurrently (no global mutable state).

## Tests and fixtures

- `test_sas.jl` holds golden numbers from the SAS/STAT LOGIEX11 documentation example.
- `test_reference.jl` compares against the JSON fixtures in `test/reference/`. Those fixtures come only from `generate_reference.R`, so don't hand-edit them.
- `test/helpers.jl` provides `load_fixture` and `brute`, a BigFloat brute-force subset enumerator used as the oracle in the kernel tests.
- Aqua checks run in `test_aqua.jl`.
