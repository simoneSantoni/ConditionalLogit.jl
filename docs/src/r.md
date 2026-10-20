# R survival::clogit

This page shows how an R `survival::clogit` analysis maps onto ConditionalLogit.jl. R's
`clogit` defaults to `method = "exact"`, the exact conditional likelihood. Fully
converged, it gives the same estimates, standard errors and log-likelihood as
ConditionalLogit.jl. The R reference fixtures in the test suite and the
[Performance](@ref) benchmarks agree to about 1e-9 or better (see [Validation](@ref)).

## Syntax

| R | ConditionalLogit.jl |
|:--|:--|
| `clogit(case ~ x1 + x2 + strata(id), data = df)` | `clogit(@formula(case ~ x1 + x2), df; strata = :id)` |
| `strata(a, b)` | `strata = [:a, :b]` |
| `offset(o)` in the formula | `offset = :o` |
| `coxph(Surv(rep(1, n), case) ~ x + strata(id), ties = "exact")` | the same model |
| `method = "approximate"`, `"breslow"`, `"efron"` | not supported (identical to exact when every stratum has one event) |
| `weights =` | ignored by R's exact method; `freq =` here follows SAS `FREQ` |
| `robust = TRUE`, `cluster(id)` | not available for R's exact method; not supported here |
| `control = coxph.control(iter.max = 50)` | `maxiter = 50` |

The response is a 0/1 or logical case indicator in both. R also accepts 1/2 status
coding, where 2 is the event; ConditionalLogit.jl needs `event = 2` for that.

## Output

| R output | ConditionalLogit.jl |
|:--|:--|
| `coef`, `se(coef)`, `z`, `p` | `coeftable(m)` |
| `exp(coef)`, `lower .95`, `upper .95` (`summary(fit)$conf.int`) | `oddsratios(m)` |
| `Likelihood ratio test`, `Wald test`, `Score (logrank) test` | `globaltests(m)` (`.lr`, `.wald`, `.score`) |
| `fit$loglik` (null, fitted) | `(nullloglikelihood(m), loglikelihood(m))` |
| `vcov(fit)` | `vcov(m)` |
| `AIC(fit)` | `aic(m)` |
| `BIC(fit)` | `deviance(m) + dof(m) * log(nevents)` (see below) |
| `n`, `number of events` | `m.nobs_used`, number of events in the used rows |
| `NA` coefficient (collinear or constant within strata) | coefficient 0, `aliased(m)` |

**BIC uses a different `n`.** For `coxph` fits, including `clogit`, R's `nobs()` is the
number of events, so `BIC(fit)` is `−2ℓ + p·log(number of events)`. SAS, Stata and
ConditionalLogit.jl use the number of observations. R also counts events in
uninformative strata (all cases), which SAS and Stata drop. To get R's value:

```julia
nevents = count(==(1), df.case)            # events in all strata, after removing missing rows
deviance(m) + dof(m) * log(nevents)
```

**Uninformative strata.** R keeps strata in which every row is a case or none is. They
contribute nothing to the likelihood, but R counts them in `n` and in the number of
events. ConditionalLogit.jl drops them and lists them in `stratasummary(m)`, as SAS and
Stata do.

## Convergence

R's `coxph.control()` defaults stop when the relative change in the log-likelihood is
below `eps = 1e-9`, after at most 20 iterations. That usually takes one more Newton step
than the SAS `GCONV = 1e-8` iterate that ConditionalLogit.jl reports by default. For an
exact match, fit with `gconv = 1e-16`. In the [Performance](@ref) benchmarks, R and the
tight fits agree to 1e-14 in the coefficients when every stratum has a fixed number of
events, and to 7e-9 on the fixed-effects panel, where R stops a little earlier.

## Numerical range

R's exact method works with exp(x'β) directly. Covariates with large values within
strata (stratum-level means in the hundreds) can make it overflow, and R then returns
`NA` coefficients without an error. ConditionalLogit.jl centres covariates within
strata and computes the likelihood in log space, so it does not overflow. When you
compare the two on such data, centre the covariates within strata before fitting in R.
The likelihood does not change.
