# Stata clogit

This page shows how a Stata `clogit` analysis maps onto ConditionalLogit.jl. Stata's
`clogit` uses the exact conditional likelihood for any number of positive outcomes per
group, so a fully converged fit gives the same numbers. The examples in the Stata
`[R] clogit` manual entry are reproduced to every printed digit (see
[Validation](@ref)).

```@example stata
using ConditionalLogit, DataFrames
```

## Syntax

| Stata | ConditionalLogit.jl |
|:--|:--|
| `clogit y x1 x2, group(id)` (or `strata(id)`) | `clogit(@formula(y ~ x1 + x2), df; strata = :id)` |
| `clogit y x1 i.race, group(id)` | `clogit(@formula(y ~ x1 + race), df; strata = :id, contrasts = Dict(:race => DummyCoding()))` |
| `xtlogit y x, fe` (panel variable `id`) | `clogit(@formula(y ~ x), df; strata = :id)` |
| `clogit y x, group(id) offset(o)` | `clogit(@formula(y ~ x), df; strata = :id, offset = :o)` |
| `clogit y x, group(id) or` | `oddsratios(m)` |
| `clogit y x, group(id) level(90)` | `coeftable(m; level = 0.90)`, `oddsratios(m; level = 0.90)` |
| `clogit y x, group(id) iterate(50)` | `clogit(...; maxiter = 50)` |
| `clogit y x [fw = w], group(id)` | replicate the groups (see [below](@ref stata-fweights)) |
| `vce(robust)`, `vce(cluster c)`, `[pw = ]`, `[iw = ]` | not supported |

**Event coding.** Stata treats any nonzero value of the dependent variable as a positive
outcome. ConditionalLogit.jl models `y == 1` (or `true`) for 0/1 and Bool responses, and
asks for `event =` otherwise. For a variable coded, say, 0/2, create the indicator
first: `df.pos = df.y .!= 0`.

**Factor variables.** Stata's `i.race` uses the lowest level as the base. StatsModels'
`DummyCoding()` does the same, so the parameter table matches. A numeric column is
treated as continuous unless you give it a contrast (or convert it to a categorical
array), as `i.` does in Stata.

## Output

| Stata output | ConditionalLogit.jl |
|:--|:--|
| `Number of obs` (`e(N)`) | `nobs(m)` (observations in informative strata) |
| note: N groups (M obs) omitted because of all positive or all negative outcomes | `stratasummary(m)`; `m.nobs_used - nobs(m)` observations |
| note: x omitted because of no within-group variance | `aliased(m)`; the coefficient is 0 |
| `LR chi2(k)`, `Prob > chi2` (`e(chi2)`) | `globaltests(m).lr` |
| `Log likelihood` (`e(ll)`) | `loglikelihood(m)` |
| `e(ll_0)` | `nullloglikelihood(m)` |
| `Pseudo R2` (`e(r2_p)`) | `1 - loglikelihood(m) / nullloglikelihood(m)` |
| `Coefficient`, `Std. err.`, `z`, `P>\|z\|`, `[95% conf. interval]` | `coeftable(m)` |
| `Odds ratio`, `[95% conf. interval]` (`, or`) | `oddsratios(m)` |
| `Std. err.` of an odds ratio (`, or`) | `exp.(coef(m)) .* stderror(m)` (delta method) |
| `e(b)`, `e(V)` | `coef(m)`, `vcov(m)` |
| `estat ic`: AIC, BIC | `aic(m)`, `bic(m)` |

`estat ic` uses `N = e(N)`, the number of observations in the groups that were kept.
That is ConditionalLogit.jl's default `n` for BIC, so `bic(m)` equals Stata's BIC.

ConditionalLogit.jl also reports what Stata does not print by default: the score and
Wald global tests (`globaltests(m)`) and the Strata Summary.

## Convergence

Stata stops when the scaled gradient `gH⁻¹g'` is below `1e-5` and either the relative
change in the coefficients is at most `1e-6` or the relative change in the
log-likelihood is at most `1e-7` (the `[R] Maximize` defaults). That is usually one
Newton step beyond the SAS `GCONV = 1e-8` iterate that ConditionalLogit.jl reports by
default. Stata prints coefficients to seven significant digits, so the last digit can
differ. Use `gconv = 1e-16` to reproduce Stata to every printed digit:

```julia
m = clogit(@formula(low ~ lwt + smoke + ptd + ht + ui + race), lowbirth2;
           strata = :pairid, contrasts = Dict(:race => DummyCoding()), gconv = 1e-16)
```

## [Frequency weights](@id stata-fweights)

In Stata, `fweight`s apply to **groups**: a group with weight 3 stands for three
identical groups, and the weight must be the same for every observation in the group.
ConditionalLogit.jl's `freq =` follows SAS `FREQ` instead, where each **row** stands for
`freq` identical observations in the same stratum. These are different models.

To reproduce a Stata `fweight` analysis, replicate each group `w` times with a new group
id. The example below is the weighted McNemar example from the Stata manual: 56 matched
pairs entered as four weighted groups.

```@example stata
pairs = DataFrame(id      = [1, 1, 2, 2, 3, 3, 4, 4],
                  case    = [1, 0, 1, 0, 1, 0, 1, 0],
                  exposed = [1, 1, 1, 0, 0, 1, 0, 0],
                  weight  = [8, 8, 22, 22, 8, 8, 18, 18])

# Stata: clogit case exposed [fw = weight], group(id) or
expanded = reduce(vcat, [transform(g, :id => (id -> fill(10_000 * id[1] + r, length(id))) => :id)
                         for g in groupby(pairs, :id) for r in 1:g.weight[1]])
m = clogit(@formula(case ~ exposed), expanded; strata = :id)
oddsratios(m)
```

Stata prints `Odds ratio = 2.75`, `[95% conf. interval] = 1.224347 6.176763`,
`Number of obs = 112`, `LR chi2(1) = 6.79`, `Log likelihood = -35.419282` and
`Pseudo R2 = 0.0875`:

```@example stata
nobs(m), round(globaltests(m).lr.chisq; digits = 2), loglikelihood(m),
round(1 - loglikelihood(m) / nullloglikelihood(m); digits = 4)
```

## Choice models (`cmclogit`)

Stata's `cmclogit` (formerly `asclogit`) fits McFadden's choice model, with one chosen
alternative per case. Its likelihood is the conditional logit likelihood with the case
as the stratum. By default `cmclogit` adds alternative-specific constants, and
case-specific variables enter through interactions with the alternatives. To reproduce
it, put those alternative dummies and interactions in the formula explicitly and use
the case id as `strata`.
