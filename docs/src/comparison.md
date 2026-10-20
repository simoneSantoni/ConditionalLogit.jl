# Comparison with SAS, Stata, R and Python

ConditionalLogit.jl, SAS `PROC LOGISTIC` with a `STRATA` statement, Stata `clogit`, R
`survival::clogit` and Python `statsmodels` `ConditionalLogit` all maximise the same
**exact conditional likelihood**

```math
L(\beta) = \prod_k \frac{\exp\left(\sum_{i \in \text{events}_k} x_i'\beta\right)}{\sum_{S \subset \text{stratum } k,\ |S| = m_k} \exp\left(\sum_{j \in S} x_j'\beta\right)},
```

so, fully converged, they give the same estimates, standard errors and log-likelihood.
They differ in syntax, defaults, which statistics they print, and how they treat
weights, uninformative strata and collinear covariates. This page summarises the
differences. The pages for [SAS](@ref "SAS PROC LOGISTIC"), [Stata](@ref "Stata clogit"),
[R](@ref "R survival::clogit") and [Python](@ref "Python statsmodels") show how to
reproduce each program's output, and
[Performance](@ref) compares speed.

## At a glance

| | ConditionalLogit.jl | SAS `PROC LOGISTIC` + `STRATA` | Stata `clogit` | R `survival::clogit` | Python `statsmodels` `ConditionalLogit` |
|:--|:--|:--|:--|:--|:--|
| Call | `clogit(@formula(y ~ x), df; strata = :id)` | `model y(event='1') = x; strata id;` | `clogit y x, group(id)` | `clogit(y ~ x + strata(id), data = df)` | `ConditionalLogit.from_formula("y ~ 0 + x", groups="id", data=df).fit()` |
| Likelihood | exact | exact | exact ("always uses the true conditional likelihood") | exact by default; `method = "approximate"`/`"breslow"`/`"efron"` available | exact |
| Standard errors | inverse of the analytic information matrix | inverse information matrix | `vce(oim)` (default) | inverse information matrix | inverse of a numerical Hessian (finite differences of the score) |
| Event | `y == 1` / `true`, or `event =` | **lower ordered value (`Y = 0`)** unless `EVENT=` or `DESCENDING` | any nonzero value | `TRUE` / 1 (status coding) | must be coded 0/1 |
| Starting values | β = 0 | β = 0 | Stata's own | β = 0 | β = 0 |
| Convergence | SAS `GCONV = 1e-8` (reports that iterate); `gconv` to tighten | `GCONV = 1e-8` | scaled gradient `gH⁻¹g' < 1e-5` and (relative change in β ≤ 1e-6 or in ℓ ≤ 1e-7) | relative change in ℓ < 1e-9, at most 20 iterations | BFGS, gradient norm of −ℓ/n < 1e-5, at most 100 iterations; `method="newton"` available |
| Strata with all events or none | dropped; listed in the Strata Summary | dropped; listed in the Strata Summary | dropped with a note; excluded from `Number of obs` | kept; contribute nothing but count in `n` and in the number of events | dropped with a warning; excluded from `nobs` |
| Covariate constant within strata / collinear | coefficient 0, DF 0, warning; `aliased(m)` | parameter set to 0 | "omitted because of no within-group variance" | `NA` | not checked: arbitrary estimates with huge SEs, or no SEs |
| Global tests | LR, score, Wald | LR, score, Wald | LR | LR, Wald, score (logrank) | none |
| Null log-likelihood | `nullloglikelihood(m)` = ℓ(0) | −2 Log L without covariates | `e(ll_0)` | `fit$loglik[1]` | not reported (`model.loglike(zeros)`) |
| BIC / SC `n` | observations in informative strata (`sc_n = :all` for all) | observations | `e(N)`: observations in informative strata | **number of events** | not reported |
| Pseudo R² | `1 - loglikelihood(m) / nullloglikelihood(m)` | not printed | McFadden pseudo R² | not printed | not reported |
| Odds ratios | `oddsratios(m)` (Wald CI) | Odds Ratio Estimates (Wald CI) | `, or` | `summary(fit)$conf.int` | `np.exp(r.conf_int())` |
| Frequency weights | `freq =`: each **row** counts `freq` times (SAS `FREQ`) | `FREQ`: per row | `fweight`: each **group** counts `w` times; must be constant within group | ignored by the exact method (warning) | not supported |
| Other weights | rejected | `WEIGHT` not allowed with `STRATA` | `iweight`, `pweight` | ignored by the exact method | not supported |
| Robust / cluster SE | no | no | `vce(robust)`, `vce(cluster id)` | not available for the exact method | no |
| Offset | `offset =` | `OFFSET=` | `offset()` | `offset()` in the formula | `offset=` |
| Threads | yes, bitwise-identical results for any number of threads | — | Stata/MP | no | no |

## Which numbers agree

When all five programs are fully converged on the same data (statsmodels with
`method="newton"`):

- **Identical:** coefficients, standard errors, z (or Wald χ² = z²), p-values, Wald
  confidence intervals, odds ratios, log-likelihood, ℓ(0), the likelihood-ratio
  statistic, AIC.
- **Identical where printed:** score and Wald global tests (SAS, R and ConditionalLogit.jl).
- **Close:** statsmodels standard errors, which come from a numerical Hessian (about 1e-8
  relative).
- **Different by definition:**
  - BIC: R uses the number of events as `n`, SAS and Stata the number of observations.
    Stata excludes observations in uninformative strata; for SAS this is unverified (see
    [SAS PROC LOGISTIC](@ref)).
  - The number of observations reported: R counts uninformative strata; Stata and
    statsmodels do not.
  - Signs: SAS without `EVENT='1'` or `DESCENDING` models `Y = 0`, which negates every
    coefficient.
- **Different by stopping rule:** each program stops at a different point, so the last
  printed digit can differ. ConditionalLogit.jl's default reproduces SAS. With
  `gconv = 1e-16` it reproduces every digit that Stata and R print (see
  [Validation](@ref)). statsmodels' default BFGS stops much earlier (about the 5th
  significant digit), so use `method="newton"` there.

## Weights are not interchangeable

The programs mean different things by frequency weights:

- **SAS `FREQ` and `freq =` here:** a row with weight 3 is three identical observations
  *in the same stratum*. The stratum grows, and so does its number of events.
- **Stata `fweight`:** a group with weight 3 is three identical *groups*. Weights must be
  constant within each group.
- **R `clogit(method = "exact")`:** weights are ignored, with a warning.
- **statsmodels:** no weights.

To reproduce a Stata `fweight` analysis, replicate the groups (see
[Frequency weights](@ref stata-fweights)). To reproduce SAS `FREQ` in R, expand the rows.
