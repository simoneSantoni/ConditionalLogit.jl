# Python statsmodels

This page shows how a Python `statsmodels` analysis with
`statsmodels.discrete.conditional_models.ConditionalLogit` maps onto
ConditionalLogit.jl. statsmodels computes the exact conditional likelihood for any number
of events per group, but by default it maximises it with BFGS and takes standard errors
from a numerically differentiated Hessian. Its default estimates therefore agree with
the other programs only to about 4–5 significant digits. With `method = "newton"` they
agree to the digits that matter.

## Syntax

| statsmodels | ConditionalLogit.jl |
|:--|:--|
| `ConditionalLogit.from_formula("y ~ 0 + x1 + x2", groups="id", data=df).fit()` | `clogit(@formula(y ~ x1 + x2), df; strata = :id)` |
| `ConditionalLogit(endog, exog, groups=g).fit()` | `clogit(X, y, g)` |
| `ConditionalLogit(..., offset=o)` | `offset = :o` |
| `.fit(method="newton")` | the default (Newton–Raphson) |
| `.fit(maxiter=200)` | `maxiter = 200` |
| `.fit_regularized(...)` (elastic net) | not supported |

The formula needs `0 +`: statsmodels warns if it is missing, and it rejects a design
matrix with an intercept column. ConditionalLogit.jl drops the intercept automatically.
`endog` must be coded 0/1. statsmodels also offers `ConditionalMNLogit` (multinomial)
and `ConditionalPoisson`, which have no counterpart here.

## Output

| statsmodels output | ConditionalLogit.jl |
|:--|:--|
| `coef`, `std err`, `z`, `P>\|z\|`, `[0.025 0.975]` (`r.summary()`) | `coeftable(m)` |
| `r.params`, `r.bse`, `r.cov_params()` | `coef(m)`, `stderror(m)`, `vcov(m)` |
| `Log-Likelihood` (`r.llf`) | `loglikelihood(m)` |
| `No. Observations` (`r.nobs`), `No. groups` | `nobs(m)`, `stratasummary(m)` |
| `np.exp(r.params)`, `np.exp(r.conf_int())` | `oddsratios(m)` |
| not reported: ℓ(0), LR / score / Wald global tests, AIC, BIC | `nullloglikelihood(m)`, `globaltests(m)`, `aic(m)`, `bic(m)` |

To get ℓ(0) in statsmodels, use `model.loglike(np.zeros(p))`. `No. Observations` excludes
groups with no variation in the outcome, which statsmodels drops with a warning, as Stata
does. So `r.nobs` equals `nobs(m)`.

## Differences that change results

- **Stopping rule.** The default `method = "bfgs"` stops when SciPy's gradient tolerance
  (`gtol = 1e-5`, on −ℓ/n) is met. On the SAS matched-pairs example it returns 0.955478
  where the exact maximum is 0.955511, and standard error 0.526231 where the exact
  value is 0.526235. The difference is about 6e-5 standard errors, so it is invisible in
  a 3-decimal summary but not in a replication check. Use
  `.fit(method="newton")`, which converges to the same value as SAS, Stata, R and
  ConditionalLogit.jl.
- **Standard errors.** statsmodels differentiates the analytic score numerically to get
  the Hessian, so standard errors carry a small finite-difference error. With
  `method = "newton"`, the same example gives 0.52623482 against the exact 0.52623481.
  ConditionalLogit.jl uses the analytic information matrix.
- **Collinearity.** statsmodels does not check for redundant covariates:
  - With collinear covariates (for example `z = 2 * gall`), it returns an arbitrary
    split of the effect with standard errors in the thousands, and no warning.
  - With a covariate that is constant within every group (for example `age` in the
    matched `infert` data), inverting the Hessian fails and asking for the standard
    errors raises an error.

  ConditionalLogit.jl sets the redundant coefficient to 0, reports it with `aliased(m)`,
  and warns.
- **Numerical range.** statsmodels computes exp(x'β) directly. With covariates whose
  stratum means are in the hundreds, the recursion overflows: statsmodels warns that it
  did not converge and returns a wrong estimate. Centre the covariates within groups
  before fitting there. ConditionalLogit.jl does this internally and works in log space.
- **Speed.** The recursion runs in pure Python, one group at a time, so large data sets
  take minutes (see [Performance](@ref)).
