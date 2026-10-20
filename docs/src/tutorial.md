# Tutorial

This tutorial fits conditional logit models to two classic matched case–control data
sets, shows how to read the results, and covers the options that matter when you
reproduce published SAS output.

```@example tut
using ConditionalLogit, CSV, DataFrames, StatsModels
```

## Matched pairs: LOGIEX11

The SAS/STAT example LOGIEX11 has 63 matched pairs. `s` is the pair, `y` is 1 for the
case, and `gall` and `hyper` are indicators for gall-bladder disease and hypertension.

```@example tut
la = CSV.read(pkgdir(ConditionalLogit, "test", "reference", "la.csv"), DataFrame)
m1 = clogit(@formula(y ~ gall + hyper), la; strata = :s)
```

There is no intercept: it is conditioned out together with the strata. `y ~ x` and
`y ~ 0 + x` are the same model, and `y ~ 1 + x` is an error.

## Reading the results

Each SAS table has an accessor. The tables are Tables.jl tables, so they convert to a
`DataFrame`:

```@example tut
DataFrame(fitstatistics(m1))
```

```@example tut
DataFrame(globaltests(m1))
```

```@example tut
DataFrame(oddsratios(m1))
```

```@example tut
DataFrame(stratasummary(m1))
```

The StatsAPI accessors work as usual:

```@example tut
coef(m1), stderror(m1), loglikelihood(m1), aic(m1), bic(m1)
```

| SAS output | Julia |
|---|---|
| Model Information / Number of Observations Read, Used | `show(m)`, `m.nobs_read`, `m.nobs_used`, `m.freq_used` |
| Strata Summary | [`stratasummary`](@ref) |
| Convergence Status | [`converged`](@ref), [`convergencestatus`](@ref), [`iterations`](@ref), [`iterationhistory`](@ref) |
| -2 Log L (Without / With Covariates) | `nulldeviance(m)` / `deviance(m)` |
| AIC, SC | `aic(m)`, `bic(m)`; both columns in [`fitstatistics`](@ref) |
| Testing Global Null Hypothesis: BETA=0 | [`globaltests`](@ref) |
| Analysis of Maximum Likelihood Estimates | `coef(m)`, `stderror(m)`, `coeftable(m)` |
| Odds Ratio Estimates (Wald CL) | [`oddsratios`](@ref); general contrasts with [`oddsratio`](@ref) |
| COVB / Wald CL for parameters | `vcov(m)`, `confint(m)` |

## Nested models

The likelihood-ratio test of nested models is the difference in −2 Log L:

```@example tut
m0 = clogit(@formula(y ~ gall), la; strata = :s)
lrtest(m0, m1)
```

## Several cases per set and categorical covariates: infert

R's `infert` data set has 83 matched sets with one or two cases and up to four controls.
`education` is a string column, so it gets full-rank (k − 1) contrasts:

```@example tut
infert = CSV.read(pkgdir(ConditionalLogit, "test", "reference", "infert.csv"), DataFrame)
m2 = clogit(@formula(case ~ spontaneous + induced), infert; strata = :stratum)
```

`education` was used for matching, so it is constant within every stratum and is
collinear with the conditioned-out stratum intercepts. The package detects this, sets
its coefficients to 0 with DF 0, and warns:

```@example tut
m3 = clogit(@formula(case ~ spontaneous + induced + education), infert; strata = :stratum)
aliased(m3)
```

## Matching SAS class coding

SAS `PROC LOGISTIC` codes `CLASS` variables with `PARAM=EFFECT` and the *last* level as
reference, while StatsModels uses dummy coding with the *first* level as base. Pass
`contrasts` to reproduce the SAS parameter table:

```julia
clogit(@formula(y ~ g), df; strata = :id,
       contrasts = Dict(:g => EffectsCoding(base = "c")))   # SAS PARAM=EFFECT, last level "c"
clogit(@formula(y ~ g), df; strata = :id,
       contrasts = Dict(:g => DummyCoding(base = "c")))     # SAS PARAM=REF
```

[`oddsratios`](@ref) reports level-versus-reference odds ratios computed from the
contrast matrix, so they match SAS under either coding.

## Response ordering

By default SAS models the probability of the *lower* ordered response value, which is
`Y = 0` for a 0/1 outcome, unless the code says `event='1'` or `descending`. Use
`event = 0` to reproduce such output. Every coefficient changes sign, and the fit
statistics and tests are unchanged:

```@example tut
m4 = clogit(@formula(y ~ gall + hyper), la; strata = :s, event = 0)
coef(m4) ≈ -coef(m1), deviance(m4) ≈ deviance(m1)
```

## Matrix interface

```@example tut
X = Matrix(la[:, [:gall, :hyper]])
m5 = clogit(X, la.y, la.s)
coef(m5) ≈ coef(m1)
```

## Convergence and accuracy

As in SAS, the fit stops when the relative gradient criterion falls below `gconv = 1e-8`
and reports that iterate without taking one more step. The estimates therefore agree
with the exact optimum to about six significant digits, which is all SAS prints. For
full accuracy, tighten the criterion:

```@example tut
m6 = clogit(@formula(y ~ gall + hyper), la; strata = :s, gconv = 1e-16)
coef(m6) .- coef(m1)
```

Non-convergence and separation give a warning and `converged(m) == false`, as SAS does.
Pass `strict = true` to throw a [`ConvergenceError`](@ref) instead.

## Predictions

`fitted(m)` gives the within-stratum probability that each observation is an event.
`predict(m, newdata; strata)` gives choice probabilities for new choice sets (one event
per stratum), and `type = :linpred` gives x'β̂:

```@example tut
predict(m1, first(la, 4); strata = first(la.s, 4))
```

## Threads

Strata are processed by several tasks (`ntasks`, default `Threads.nthreads()`), and the
results are bitwise identical for any number of threads. Start Julia with
`julia -t auto` to use them. Fits can also run concurrently, for example a bootstrap
inside `Threads.@threads`.
