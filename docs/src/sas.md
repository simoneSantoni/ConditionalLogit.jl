# SAS PROC LOGISTIC

This page shows how a SAS `PROC LOGISTIC` analysis with a `STRATA` statement (the
conditional logistic regression in, for example, SAS 9.3) maps onto ConditionalLogit.jl,
and where the two can differ. ConditionalLogit.jl's defaults follow SAS, so this is the
comparison that needs the fewest options. See [Comparison with SAS, Stata, R and Python](@ref)
for the other programs.

## Which model

`PROC LOGISTIC` with a `STRATA` statement maximises the exact
conditional likelihood

```math
L_k(\beta) = \frac{\exp\left(\sum_{i \in \text{events}_k} x_i'\beta\right)}{\sum_{S \subset \text{stratum } k,\ |S| = m_k} \exp\left(\sum_{j \in S} x_j'\beta\right)}
```

for any number of events m_k per stratum. It does not use the Breslow or Efron
approximation. This is also what R `survival::clogit(method = "exact")` and Stata
`clogit` compute. This package computes the same likelihood with no approximation.
(R's `clogit` *default* is `method = "exact"`, but `coxph`-based code that uses
Breslow or Efron ties gives different numbers when m_k > 1.)

## Response ordering: check the signs

By default SAS models the probability of the
*lower* ordered response value. For a 0/1 outcome that is `Y = 0`, unless the code says
`model y(event='1') = ...` or `model y(descending) = ...`.

| SAS code | Julia |
|---|---|
| `model y(event='1') = x;` or `model y(descending) = x;` | `clogit(@formula(y ~ x), df; strata = :id)` (default) |
| `model y = x;` with y coded 0/1 | `clogit(@formula(y ~ x), df; strata = :id, event = 0)` |
| `model status(event='case') = x;` | `clogit(@formula(status ~ x), df; strata = :id, event = "case")` |

Changing the event level negates every coefficient (β → −β) and inverts every odds
ratio. Standard errors, Wald χ², p-values, −2 Log L, AIC, SC and all three global
tests are unchanged. If a paper's code shows no `EVENT='1'` or `DESCENDING`, negate
the published estimates (or invert its odds ratios) before comparing, or fit with
`event = 0`. With an offset, a flipped model is not a pure sign flip. For character
responses SAS orders the levels alphabetically (formatted values).

## CLASS variables

SAS `PROC LOGISTIC` defaults to `PARAM=EFFECT` (deviation coding)
with the *last* level as reference. StatsModels defaults to dummy coding with the
*first* level as base. To reproduce the SAS parameter table, pass
`contrasts = Dict(:g => EffectsCoding(base = "<last level>"))`. For `PARAM=REF`, pass
`DummyCoding(base = "<last level>")`. You will need `using StatsModels`. `oddsratios(m)`
reports level-versus-reference odds ratios computed from the contrast matrix, so they
match SAS under either coding. Under effect coding, `exp(β)` is *not* an odds ratio.

## Convergence

As in SAS, the fit:
- starts at β = 0;
- uses Newton–Raphson;
- stops when the relative gradient criterion g'I⁻¹g / (|ℓ| + 1e-6) < `gconv` = 1e-8;
- *reports that iterate*, without taking one more step.

The reported estimates can therefore differ from the exact optimum in about the 6th
significant digit. SAS prints 4 decimals, so this does not show. For full accuracy,
pass `gconv = 1e-16`. Non-convergence after `maxiter = 25` iterations, and complete or
quasi-complete separation, give a warning and `converged(m) == false`, as SAS gives a
warning and still prints the tables. Separation is detected with direction
certificates (events at least as high as non-events in every stratum along a
direction). A stratum counts as a tie only within rounding error of its own values,
and every certificate is confirmed by evaluating ℓ along it, so an outlying covariate
value does not raise a false alarm.

### Step control differs from SAS

SAS labels the conditional fit "Newton-Raphson ridge"
(NRRIDG), which handles a step that lowers ℓ by increasing a ridge on the Hessian. This
package halves the step instead, and ridges only when the information matrix is not
positive definite. When the full Newton step is accepted, which is the usual case from
β = 0, the iterates are the same. On hard problems (near-separation, poor start values)
the iterates, and hence the last digits of the reported GCONV iterate, can differ from
SAS. The report therefore says "Newton-Raphson (step-halving; ridge if not PD)".

## Fit statistics

- −2 Log L "Without Covariates" is −2ℓ(0). It equals 2Σ_k log C(n_k, m_k) when there is
  no offset.
- AIC = −2 Log L + 2p.
- SC = −2 Log L + p·log(n). Here n is the number of observations (sum of FREQ), **not**
  the number of strata. The LOGIEX11 output confirms this: 83.654 + log(126) = 88.490.

One detail is **unverified against SAS**: whether SAS counts observations in
*uninformative* strata (all events or no events) in this n. LOGIEX11 has no
uninformative strata, so it cannot settle the question. The default
`sc_n = :informative` excludes them, and `sc_n = :all` includes them. The two agree
whenever every stratum is informative. When they differ, the report prints a note
giving the n used in SC, because the "Number of Observations Used" lines include
uninformative strata. If you have SAS access, one run of a model with an all-event
stratum would settle this.

## Strata

Strata with only events or only non-events are uninformative. They are
reported in the Strata Summary and not used, as in SAS. Rows with a missing value in
the response, a covariate, the strata, FREQ or the offset are dropped. A missing stratum
id is kept as its own stratum only with `strata_missing = true`.

## FREQ and WEIGHT

- FREQ: each row counts as `freq` identical observations. Non-integer values are
  truncated, and rows with `freq < 1` are dropped, as in SAS.
- WEIGHT: SAS does not allow it with STRATA. The package therefore rejects `pweights` and
  `aweights`.

## Collinearity

A covariate that is a linear combination of earlier covariates
*within strata* is set to 0 with DF 0, with a warning. This includes a covariate that is
constant within every stratum, which is collinear with the conditioned-out stratum
intercepts. The later column is the one dropped, as in SAS and R. SAS's default
`CHECKDEPENDENCY=COVARIATES` does not detect the stratum-constant case, so our behaviour
corresponds to `CHECKDEPENDENCY=ALL`.

The default tolerance `singular = 1e7·eps ≈ 2.2e-9` is the SAS `SINGULAR=` default. A
column is dropped when 1 − R² on the earlier columns, computed from the within-stratum
centred Gram matrix, is below it. SAS instead compares a sweep pivot of the Hessian
with `SINGULAR` × a norm of the matrix. The null space is the same, but nearly
collinear covariates close to the threshold can be decided differently.

## Not supported

- The SAS `EXACT` statement (exact conditional inference).
- Profile-likelihood intervals (`CLODDS=PL`, `CLPARM=PL`).
- Firth's penalised likelihood.
- `predict(m, newdata; type = :prob)` for strata with more than one event (it gives
  one-event choice probabilities). `residuals(m)` is not defined.
- Breslow and Efron approximations.

## Cost with large FREQ counts

The exact likelihood costs O(n_k · min(m_k, n_k − m_k))
per stratum, where n_k counts FREQ copies. Aggregated tables with very large cell
counts therefore take time proportional to the counts, not to the number of rows. Memory
stays O(rows · m_k).

