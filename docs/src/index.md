# ConditionalLogit.jl

Conditional logit models in Julia, estimated by exact conditional maximum likelihood.

## Overview

The conditional logit model (conditional logistic regression) is the standard model for
data grouped into *strata*:

- **Matched case–control studies**: each matched set is a stratum, and the cases are
  its events.
- **Fixed-effects (Chamberlain) logit**: each panel unit is a stratum, and the unit
  intercepts are conditioned out of the likelihood.
- **McFadden's choice model**: each choice set is a stratum with exactly one chosen
  alternative.

ConditionalLogit.jl is written so that published results can be reproduced, whichever
program produced them. It computes the same exact conditional likelihood as SAS
`PROC LOGISTIC` with a `STRATA` statement, Stata `clogit` (and `xtlogit, fe`) and R
`survival::clogit`, and documents how to map each program's syntax, defaults and output
onto it. A sentence such as

> We estimated the conditional logit model by maximum likelihood using PROC LOGISTIC in
> SAS (release 9.3).

needs no options at all: the defaults follow SAS (starting values, Newton–Raphson
convergence rule, fit statistics, global tests, odds-ratio tables), and results print in
a SAS-like layout. For Stata, R and Python statsmodels, the [comparison pages](@ref "Comparison with SAS, Stata, R and Python") list the few differences in definitions (BIC, weights, uninformative
strata) and how to reproduce them.

## Features

- **Exact conditional likelihood** for any number of events per stratum (no Breslow or
  Efron approximation), computed with an overflow-free log-space form of the
  Gail–Lubin–Rubinstein recursion
- **Reproduces SAS, Stata, R and statsmodels**: SAS `PROC LOGISTIC` defaults (start at β = 0,
  Newton–Raphson with `GCONV=1e-8`, `MAXITER=25`) and tables; every printed number of
  the SAS and Stata manual examples; R `survival::clogit(method = "exact")` to 1e-9 or
  better
- **StatsModels formulas and a matrix interface**, any Tables.jl source, categorical
  covariates, FREQ weights, offsets, and multi-column strata keys
- **StatsAPI `RegressionModel`**: `coef`, `stderror`, `vcov`, `confint`, `coeftable`,
  `aic`, `bic`, `lrtest`, `predict`, …
- **Diagnostics**: collinearity within strata (aliased coefficients), complete and
  quasi-complete separation, non-convergence
- **Multithreaded** over strata, with results that are bitwise identical for any number
  of threads; fits can run concurrently inside `Threads.@threads`
- **Fast**: on one thread, 7–22× faster than R `survival::clogit`, 13–41× faster than
  Stata `clogit` and 144–324× faster than Python `statsmodels`; 18–53× faster than
  Stata/MP on 16 cores
  (see [Performance](@ref))

## Installation

ConditionalLogit.jl is not yet in the General registry. Install it from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/simoneSantoni/ConditionalLogit.jl")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/ConditionalLogit.jl")
```

Julia 1.10 or later is required.

## Quick Start

The SAS/STAT documentation example LOGIEX11 has 63 matched pairs from the Los Angeles
endometrial cancer study. The data ship with the package's tests:

```@example quickstart
using ConditionalLogit, CSV, DataFrames

la = CSV.read(pkgdir(ConditionalLogit, "test", "reference", "la.csv"), DataFrame)
first(la, 4)
```

```@example quickstart
m = clogit(@formula(y ~ gall), la; strata = :s)
```

Every number in the Model Fit Statistics, Global Null Hypothesis, estimates and odds
ratio tables is identical to the SAS output printed in the SAS/STAT documentation for
this example. The same model in Stata (`clogit y gall, group(s)`) and R
(`clogit(y ~ gall + strata(s))`) gives the same estimate, standard error and
likelihood-ratio test.

```@example quickstart
coeftable(m)
```

## Choosing an Entry Point

| Use Case | Call |
|----------|------|
| Formula and a table | [`clogit`](@ref)`(@formula(y ~ x1 + x2), df; strata = :id)` |
| StatsAPI style | `fit(ConditionalLogitModel, @formula(y ~ x), df; strata = :id)` |
| Design matrix | `clogit(X, y, strata)` (no intercept column) |
| Several columns form the stratum (SAS `strata a b;`) | `clogit(f, df; strata = [:hospital, :set])` |
| SAS `FREQ` | `clogit(f, df; strata = :id, freq = :n)` |
| SAS `OFFSET=` | `clogit(f, df; strata = :id, offset = :o)` |
| SAS default response ordering (models `Y = 0`) | `clogit(f, df; strata = :id, event = 0)` |
| Stata `clogit y x, group(id)` / `xtlogit y x, fe` | `clogit(@formula(y ~ x), df; strata = :id)` |
| R `clogit(y ~ x + strata(id))` | `clogit(@formula(y ~ x), df; strata = :id)` |
| statsmodels `ConditionalLogit.from_formula("y ~ 0 + x", groups="id", data=df)` | `clogit(@formula(y ~ x), df; strata = :id)` |
| Match Stata or R to every printed digit | `clogit(f, df; strata = :id, gconv = 1e-16)` |

See the [Tutorial](@ref) for worked examples, and
[Comparison with SAS, Stata, R and Python](@ref) for the details that matter when you compare
with published output from those programs.

## Performance snapshot

ConditionalLogit.jl compared with R `survival::clogit(method = "exact")`, Stata `clogit`
and Python `statsmodels` `ConditionalLogit` on seven synthetic data sets from 40,000 to
1,000,000 rows. All four programs give the same estimates up to their convergence
tolerances.

Each cell is the other program's time divided by ConditionalLogit.jl's time, so a value
above 1× means ConditionalLogit.jl finished first:

- **"vs R", "vs Stata, 1 core", "vs Stata/MP, 16 cores", "vs statsmodels":**
  ConditionalLogit.jl on one thread. R and statsmodels are single-threaded.
- **"Julia best" columns:** ConditionalLogit.jl at its fastest of 1, 8 or 24 threads.
- **Settings:** every program used its default settings on the same data.

| Scenario | Rows | vs R | vs Stata, 1 core | vs Stata/MP, 16 cores | vs statsmodels | vs R (Julia best) | vs Stata/MP (Julia best) | vs statsmodels (Julia best) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 1:1 matched pairs (`pairs`) | 40,000 | 12.2× | 41.4× | 44.6× | 322× | 14.5× | 53.0× | 383× |
| 1:4 matching (`match1to4`) | 100,000 | 22.5× | 38.9× | 33.0× | 324× | 28.5× | 41.8× | 411× |
| choice sets of 10 (`choice`) | 500,000 | 15.6× | 23.0× | 17.1× | 164× | 22.7× | 24.8× | 237× |
| 1–9 events per stratum (`multi`) | 200,000 | 14.8× | 39.4× | 16.5× | 198× | 38.1× | 42.5× | 511× |
| fixed-effects panel (`panelfe`) | 200,000 | 11.0× | 31.2× | 16.9× | 290× | 27.3× | 41.8× | 719× |
| 100-row strata, 50 events (`bigstrata`) | 50,000 | 7.3× | 13.3× | 4.0× | 285× | 31.5× | 17.6× | 1237× |
| choice sets of 10, 20 covariates (`choicelarge`) | 1,000,000 | 15.6× | 25.1× | 15.2× | 144× | 23.6× | 23.0× | 218× |

Absolute timings depend on the machine (here an Intel Core i9-14900HX, Julia 1.13, R 4.6.1
with `survival` 3.8-6, Stata/MP 15, Python 3.13 with `statsmodels` 0.15). See [Performance](@ref) for the timings, scenarios,
method and a run to full convergence.

## Documentation

```@contents
Pages = ["tutorial.md", "comparison.md", "sas.md", "stata.md", "r.md", "statsmodels.md", "algorithm.md", "validation.md", "performance.md", "development.md", "api.md"]
Depth = 2
```

## References

1. Gail, M.H., Lubin, J.H., Rubinstein, L.V. (1981). Likelihood calculations for matched case-control studies and survival studies with tied death times. *Biometrika*, 68(3), 703-707. [doi:10.1093/biomet/68.3.703](https://doi.org/10.1093/biomet/68.3.703)
2. Breslow, N.E., Day, N.E. (1980). *Statistical Methods in Cancer Research, Volume 1: The Analysis of Case-Control Studies*. IARC Scientific Publications No. 32. Lyon: International Agency for Research on Cancer.
3. McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105-142). New York: Academic Press.
4. Chamberlain, G. (1980). Analysis of covariance with qualitative data. *Review of Economic Studies*, 47(1), 225-238. [doi:10.2307/2297110](https://doi.org/10.2307/2297110)
5. SAS Institute Inc. *SAS/STAT User's Guide*, The LOGISTIC Procedure: Conditional Logistic Regression; Example "Conditional Logistic Regression for Matched Pairs Data".
6. StataCorp. *Stata Base Reference Manual*, `clogit` — Conditional (fixed-effects) logistic regression.
7. Therneau, T.M. *A Package for Survival Analysis in R* (`survival`), function `clogit`. [CRAN](https://CRAN.R-project.org/package=survival)

## Citation

```biblatex
@software{SantoniConditionalLogitJL,
  author = {Santoni, Simone},
  title = {ConditionalLogit.jl: Exact Conditional Logit Models in Julia},
  version = {0.1.0},
  year = {2026},
  url = {https://github.com/simoneSantoni/ConditionalLogit.jl},
  note = {Julia package, version 0.1.0. Source code and documentation at the URL above.}
}
```

The same entry is in `CITATION.bib`, and a machine-readable `CITATION.cff` is included
in the repository.
