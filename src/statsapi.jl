# StatsAPI / StatsModels interface.

coef(m::ConditionalLogitModel) = m.coef
coefnames(m::ConditionalLogitModel) = m.coefnames
responsename(m::ConditionalLogitModel) = m.responsename
StatsModels.formula(m::ConditionalLogitModel) = m.formula

"""
    vcov(m)

Inverse information matrix at β̂, the SAS covariance of the estimates (`COVB`). Rows
and columns of aliased coefficients are `NaN`.
"""
vcov(m::ConditionalLogitModel) = m.vcov
stderror(m::ConditionalLogitModel) = sqrt.(diag(m.vcov))

"""
    informationmatrix(m; expected = true)

Information matrix at β̂: minus the Hessian of the conditional log-likelihood, which is
Var[x_S] summed over strata. It does not depend on the response, so the observed and
expected information are the same. Aliased rows and columns are 0.
"""
informationmatrix(m::ConditionalLogitModel; expected::Bool = true) = m.information

"""
    score(m)

Gradient of the conditional log-likelihood at β̂ (≈ 0 at convergence). The score *test*
of β = 0 is in [`globaltests`](@ref).
"""
score(m::ConditionalLogitModel) = m.gradient

"""
    loglikelihood(m)

Exact conditional log-likelihood at β̂. Summed over informative strata, with FREQ
weighting. `-2loglikelihood(m)` is SAS "-2 Log L, With Covariates".
"""
loglikelihood(m::ConditionalLogitModel) = m.loglik

"""
    nullloglikelihood(m)

Conditional log-likelihood at β = 0 (with the offset, if any). Without an offset it is
`-Σ_k log C(n_k, m_k)`. `-2nullloglikelihood(m)` is SAS "-2 Log L, Without Covariates".
"""
nullloglikelihood(m::ConditionalLogitModel) = m.loglik0
deviance(m::ConditionalLogitModel) = -2 * m.loglik
nulldeviance(m::ConditionalLogitModel) = -2 * m.loglik0

"""
    nobs(m)

The number of observations used, which is the `n` in SC/BIC. It is the sum of FREQ over
the observations in informative strata (`sc_n = :informative`, the default), or over all
observations used (`sc_n = :all`). The two differ only if there are uninformative strata.
"""
nobs(m::ConditionalLogitModel) =
    m.options.sc_n === :all ? m.freq_used : m.summary.freq_informative

"""
    dof(m)

Number of estimated (non-aliased) parameters. There is no intercept and no dispersion
parameter.
"""
dof(m::ConditionalLogitModel) = count(!, m.aliased)

"""
    dof_residual(m)

`nobs(m) - dof(m)`. Defined so that `StatsModels.lrtest` works; the likelihood ratio
test itself uses only the deviances and `dof`.
"""
dof_residual(m::ConditionalLogitModel) = nobs(m) - dof(m)

"""
    StatsModels.isnested(m1, m2; atol = 0)

`true` if `m1` is nested in `m2`: both were fitted to the same observations (response,
strata, FREQ and offset), and every non-aliased covariate of `m1` lies, within strata, in
the span of the covariates of `m2`. Used by `StatsModels.lrtest`.
"""
function StatsModels.isnested(m1::ConditionalLogitModel, m2::ConditionalLogitModel;
                              atol::Real = 0.0)
    (m1.y == m2.y && m1.strata == m2.strata && m1.freq == m2.freq &&
     m1.offset == m2.offset && m1.options.sc_n == m2.options.sc_n) || return false
    dof(m1) <= dof(m2) || return false
    X1 = _within_centred(m1, findall(!, m1.aliased))
    X2 = _within_centred(m2, findall(!, m2.aliased))
    isempty(X1) && return true
    isempty(X2) && return false
    R = X1 - X2 * (X2 \ X1)
    for j in axes(X1, 2)
        norm(view(R, :, j)) <= sqrt(eps()) * max(norm(view(X1, :, j)), 1) + atol || return false
    end
    return true
end

# FREQ-weighted, within-stratum centred covariates (rows scaled by √f).
function _within_centred(m::ConditionalLogitModel, cols::Vector{Int})
    X = Float64.(m.X[:, cols])
    K = isempty(m.strata) ? 0 : maximum(m.strata)
    sw = zeros(K); sx = zeros(K, length(cols))
    for i in axes(X, 1)
        k = m.strata[i]; sw[k] += m.freq[i]
        for j in axes(X, 2)
            sx[k, j] += m.freq[i] * X[i, j]
        end
    end
    for i in axes(X, 1), j in axes(X, 2)
        k = m.strata[i]
        X[i, j] = sqrt(m.freq[i]) * (X[i, j] - sx[k, j] / sw[k])
    end
    return X
end
islinear(::ConditionalLogitModel) = false

"""
    converged(m)

`true` if a convergence criterion was met and no separation was detected.
"""
converged(m::ConditionalLogitModel) = m.converged

"""
    convergencestatus(m) -> Symbol

The status of the fit: `:converged`, `:maxiter`, `:stalled`, `:complete_separation` or
`:quasi_separation`.
"""
convergencestatus(m::ConditionalLogitModel) = m.status

"""
    aliased(m) -> BitVector

Coefficients set to 0 because their covariate is a linear combination of earlier
covariates within strata, or is constant within every stratum.
"""
aliased(m::ConditionalLogitModel) = m.aliased

"""
    iterations(m)

Number of Newton–Raphson iterations taken.
"""
iterations(m::ConditionalLogitModel) = m.iterations

"""
    iterationhistory(m) -> Vector{IterationRecord}

The SAS-style iteration history: iteration, log-likelihood, relative gradient (GCONV)
criterion, step-halvings, ridging and parameter values (full length, 0 for aliased
coefficients), for each iterate.
"""
iterationhistory(m::ConditionalLogitModel) = m.history

"""
    stratasummary(m) -> StrataSummary

The SAS "Strata Summary": strata grouped by response pattern (events, non-events),
uninformative patterns included. It is a Tables.jl table (`DataFrame(stratasummary(m))`)
with columns `pattern`, `events`, `nonevents`, `nstrata`, `frequency`, `informative`.
"""
stratasummary(m::ConditionalLogitModel) = m.summary

"""
    modelmatrix(m)

The covariate matrix of the observations used, in data order.
"""
modelmatrix(m::ConditionalLogitModel) = m.X

"""
    response(m)

The event indicator (`true` = event) of the observations used, in data order.
"""
response(m::ConditionalLogitModel) = m.y
weights(m::ConditionalLogitModel) = fweights(m.freq)

"""
    fitted(m)

Within-stratum conditional probability that each observation is an event, given its
stratum's number of events: π_i = P(i ∈ S). It is the softmax for 1:M matched sets
(m_k = 1), and Σ_i f_i π_i = m_k within each stratum. Observations in uninformative
strata get 1 (all events) or 0 (no events).

It is computed on the first call and cached (a fit does not pay for it). The cost is
O(Σ_k n_k m_k) time and O(rows_k · m_k) memory per stratum, where n_k counts FREQ.
"""
function fitted(m::ConditionalLogitModel)
    lock(m.fittedlock) do
        v = m.fittedcache[]
        if v === nothing
            v = compute_fitted(m)
            m.fittedcache[] = v
        end
        return v
    end
end

"""
    predict(m)
    predict(m, newdata; strata, offset = nothing, type = :prob)        # formula models
    predict(m, Xnew; strata, offset = nothing, type = :prob)           # matrix models

Without new data, `predict(m) == fitted(m)`.

With new data:
- `type = :linpred` gives the linear predictor x'β̂ (+ offset). It has no intercept, so it
  is identified only up to a constant within each stratum.
- `type = :prob` gives the within-stratum choice probabilities exp(η_i) / Σ_{j ∈ k} exp(η_j),
  the probability that row i is *the* event of its stratum (McFadden's conditional logit
  for new choice sets: one event per stratum).

`strata` is a column name (or names) of `newdata`, or a vector of stratum values. Rows
with missing covariates, strata or offset get `missing`.

Residuals are not defined for this model (`residuals` gives a MethodError).
"""
predict(m::ConditionalLogitModel) = fitted(m)

function predict(m::ConditionalLogitModel, newdata; strata, offset = nothing,
                 type::Symbol = :prob)
    type in (:prob, :linpred) ||
        throw(ArgumentError("type must be :prob or :linpred, got :$type"))
    m.formula === nothing &&
        throw(ArgumentError("this model was fitted from a matrix; call predict(m, Xnew; strata)"))
    Tables.istable(newdata) || throw(ArgumentError("newdata must be a Tables.jl table"))
    cols = Tables.columns(newdata)
    cnames = Symbol.(collect(Tables.columnnames(cols)))
    n = length(Tables.getcolumn(cols, first(cnames)))
    vars = collect(Symbol, termvars(m.formula.rhs))
    vcols = AbstractVector[_column(cols, cnames, v, "formula") for v in vars]
    skeys, scols = _predict_strata(cols, cnames, strata, n)
    oraw = offset isa Union{Symbol,AbstractString} ?
           _column(cols, cnames, Symbol(offset), "offset") : offset
    mask = trues(n)
    foreach(c -> _mark_missing!(mask, c), vcols)
    foreach(c -> _mark_missing!(mask, c), scols)
    oraw === nothing || _mark_missing!(mask, oraw)
    rows = findall(mask)
    sub = NamedTuple{Tuple(vars)}(Tuple(_take_rows(c, rows, false) for c in vcols))
    Xm = modelcols(m.formula.rhs, sub)
    X = Matrix{Float64}(Xm isa AbstractVector ? reshape(Xm, :, 1) : Xm)
    o = oraw === nothing ? zeros(length(rows)) : Vector{Float64}(_take_rows(oraw, rows, false))
    return _predict_rows(m, X, skeys[rows], o, rows, n, type)
end

function predict(m::ConditionalLogitModel, Xnew::AbstractMatrix{<:Union{Missing,Real}};
                 strata::AbstractVector, offset = nothing, type::Symbol = :prob)
    type in (:prob, :linpred) ||
        throw(ArgumentError("type must be :prob or :linpred, got :$type"))
    n, p = size(Xnew)
    p == length(m.coef) ||
        throw(DimensionMismatch("Xnew has $p columns, the model has $(length(m.coef))"))
    length(strata) == n ||
        throw(DimensionMismatch("Xnew has $n rows but strata has $(length(strata)) elements"))
    offset === nothing || length(offset) == n ||
        throw(DimensionMismatch("offset has length $(length(offset)), expected $n"))
    mask = trues(n)
    for j in 1:p
        _mark_missing!(mask, view(Xnew, :, j))
    end
    _mark_missing!(mask, strata)
    offset === nothing || _mark_missing!(mask, offset)
    rows = findall(mask)
    X = Matrix{Float64}(Xnew[rows, :])
    o = offset === nothing ? zeros(length(rows)) : Vector{Float64}(offset[rows])
    return _predict_rows(m, X, strata[rows], o, rows, n, type)
end

function _predict_strata(cols, cnames, strata, n)
    if strata isa Union{Symbol,AbstractString} ||
       (strata isa Union{AbstractVector,Tuple} && !isempty(strata) &&
        all(x -> x isa Symbol, strata))
        scols = [_column(cols, cnames, s, "strata") for s in _symlist(strata)]
        return _combine_keys(scols...), scols
    elseif strata isa AbstractVector
        length(strata) == n ||
            throw(DimensionMismatch("strata has $(length(strata)) elements, newdata has $n rows"))
        return strata, [strata]
    end
    throw(ArgumentError("strata must be a column name, a vector of column names or a " *
                        "vector of stratum values"))
end

function _predict_rows(m::ConditionalLogitModel, X::Matrix{Float64}, skeys, o, rows, n, type)
    β = Float64.(m.coef)          # aliased coefficients are 0
    η = X * β .+ o
    if type === :prob
        codes, K = strata_codes(skeys)
        emax = fill(-Inf, K)
        for (i, k) in enumerate(codes)
            emax[k] = max(emax[k], η[i])
        end
        S = zeros(K)
        for (i, k) in enumerate(codes)
            η[i] = exp(η[i] - emax[k]); S[k] += η[i]
        end
        for (i, k) in enumerate(codes)
            η[i] /= S[k]
        end
    end
    length(rows) == n && return η
    out = Vector{Union{Missing,Float64}}(missing, n)
    out[rows] .= η
    return out
end

"""
    confint(m; level = 0.95)

Wald confidence limits β̂ ± z·SE (SAS `CLPARM=WALD`).
"""
function confint(m::ConditionalLogitModel; level::Real = 0.95)
    _checklevel(level)
    z = norminvcdf(1 - (1 - level) / 2)
    se = stderror(m)
    return hcat(m.coef .- z .* se, m.coef .+ z .* se)
end

"""
    coeftable(m; level = 0.95)

Standard `StatsBase.CoefTable` with estimate, standard error, z, p-value and Wald
confidence limits. The SAS "Wald Chi-Square" is z², with the same p-value.
"""
function coeftable(m::ConditionalLogitModel; level::Real = 0.95)
    β = m.coef
    se = stderror(m)
    z = β ./ se
    pv = [isnan(x) ? NaN : chisqccdf(1, x^2) for x in z]
    ci = confint(m; level)
    lv = round(100 * level; digits = 1)
    lvs = isinteger(lv) ? string(Int(lv)) : string(lv)
    return CoefTable(hcat(β, se, z, pv, ci[:, 1], ci[:, 2]),
                     ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $lvs%", "Upper $lvs%"],
                     copy(m.coefnames), 4, 3)
end
