# SAS-style result tables: Model Fit Statistics, Testing Global Null Hypothesis,
# Odds Ratio Estimates. Each is a small struct with a `show` method and a Tables.jl
# (column-access) interface, so `DataFrame(oddsratios(m))` works.

Tables.istable(::Type{<:ResultTable}) = true
Tables.columnaccess(::Type{<:ResultTable}) = true
Tables.columns(t::ResultTable) = _columns(t)
Tables.columnnames(t::ResultTable) = collect(keys(_columns(t)))
Tables.getcolumn(t::ResultTable, i::Int) = _columns(t)[i]
Tables.getcolumn(t::ResultTable, nm::Symbol) = _columns(t)[nm]

"""
    GlobalTests

SAS "Testing Global Null Hypothesis: BETA=0". Fields `lr`, `score`, `wald`, each a
`NamedTuple` `(chisq, df, pvalue)`.
"""
struct GlobalTests <: ResultTable
    lr::NamedTuple{(:chisq, :df, :pvalue),Tuple{Float64,Int,Float64}}
    score::NamedTuple{(:chisq, :df, :pvalue),Tuple{Float64,Int,Float64}}
    wald::NamedTuple{(:chisq, :df, :pvalue),Tuple{Float64,Int,Float64}}
end
_columns(t::GlobalTests) = (test = ["Likelihood Ratio", "Score", "Wald"],
                            chisq = [t.lr.chisq, t.score.chisq, t.wald.chisq],
                            df = [t.lr.df, t.score.df, t.wald.df],
                            pvalue = [t.lr.pvalue, t.score.pvalue, t.wald.pvalue])

_columns(s::StrataSummary) = (pattern = collect(1:length(s.patterns)),
                              events = [p.nevents for p in s.patterns],
                              nonevents = [p.nnonevents for p in s.patterns],
                              nstrata = [p.nstrata for p in s.patterns],
                              frequency = [p.freq for p in s.patterns],
                              informative = [p.informative for p in s.patterns])

function _checklevel(level::Real)
    0 < level < 1 ||
        throw(ArgumentError("level must be strictly between 0 and 1 (e.g. 0.95 for 95% " *
                            "limits), got $level"))
    return level
end

_test(x, df) = (chisq = Float64(x), df = df, pvalue = df > 0 ? chisqccdf(df, max(x, 0.0)) : NaN)

"""
    globaltests(m) -> GlobalTests

The likelihood ratio, score and Wald tests of the global null hypothesis β = 0, each
with `df = dof(m)` (non-aliased parameters), as in SAS:

- Likelihood Ratio = 2(ℓ(β̂) − ℓ(0))
- Score = U(0)' I(0)⁻¹ U(0), evaluated at β = 0 (with the offset, if any)
- Wald = β̂' I(β̂) β̂ = β̂' V⁻¹ β̂
"""
function globaltests(m::ConditionalLogitModel)
    k = .!m.aliased
    df = count(k)
    b = m.coef[k]
    wald = dot(b, m.information[k, k] * b)
    return GlobalTests(_test(2 * (m.loglik - m.loglik0), df), _test(m.score_chisq, df),
                       _test(wald, df))
end

"""
    FitStatistics

SAS "Model Fit Statistics". Fields `aic`, `sc`, `m2logl`, each a `NamedTuple`
`(without, with)` for the models without and with covariates.
"""
struct FitStatistics <: ResultTable
    aic::NamedTuple{(:without, :with),Tuple{Float64,Float64}}
    sc::NamedTuple{(:without, :with),Tuple{Float64,Float64}}
    m2logl::NamedTuple{(:without, :with),Tuple{Float64,Float64}}
end
_columns(t::FitStatistics) = (criterion = ["AIC", "SC", "-2 Log L"],
                              without = [t.aic.without, t.sc.without, t.m2logl.without],
                              with = [t.aic.with, t.sc.with, t.m2logl.with])

"""
    fitstatistics(m) -> FitStatistics

SAS "Model Fit Statistics" for a conditional analysis:

- -2 Log L = −2ℓ. "Without Covariates" is −2ℓ(0) = 2Σ_k log C(n_k, m_k) when there is no
  offset.
- AIC = −2 Log L + 2p
- SC = −2 Log L + p·log(n), with `n = nobs(m)`, the number of observations (sum of FREQ),
  not the number of strata. This was checked against the SAS LOGIEX11 output. By
  default (`sc_n = :informative`) n counts only observations in informative strata;
  whether SAS also counts uninformative strata is unverified (`sc_n = :all` does). The
  text display prints the n used whenever it differs from the observations used.

Here p = `dof(m)`; "Without Covariates" has p = 0. `aic(m)` and `bic(m)` return the
"With Covariates" AIC and SC.
"""
function fitstatistics(m::ConditionalLogitModel)
    l0 = -2 * m.loglik0
    l1 = -2 * m.loglik
    p = dof(m)
    return FitStatistics((without = l0, with = l1 + 2p), (without = l0, with = l1 + p * log(nobs(m))),
                         (without = l0, with = l1))
end

"""
    OddsRatioTable

SAS "Odds Ratio Estimates": columns `effect`, `estimate`, `lower`, `upper` (Wald
confidence limits at level `level`).
"""
struct OddsRatioTable <: ResultTable
    effect::Vector{String}
    estimate::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    level::Float64
end
_columns(t::OddsRatioTable) = (effect = t.effect, estimate = t.estimate, lower = t.lower,
                               upper = t.upper)

"""
    oddsratio(m, c::AbstractVector; level = 0.95) -> (estimate, lower, upper)

Odds ratio `exp(c'β̂)` for a general contrast `c` (length `p`), with Wald limits
`exp(c'β̂ ± z·√(c'Vc))`. Aliased coefficients must have zero weight.
"""
function oddsratio(m::ConditionalLogitModel, c::AbstractVector{<:Real}; level::Real = 0.95)
    length(c) == length(m.coef) ||
        throw(DimensionMismatch("contrast has length $(length(c)), expected $(length(m.coef))"))
    any(c[m.aliased] .!= 0) &&
        throw(ArgumentError("the contrast puts weight on aliased coefficients"))
    _checklevel(level)
    k = .!m.aliased
    ck = c[k]
    est = dot(ck, m.coef[k])
    se = sqrt(max(dot(ck, m.vcov[k, k] * ck), 0.0))
    z = norminvcdf(1 - (1 - level) / 2)
    return (estimate = exp(est), lower = exp(est - z * se), upper = exp(est + z * se))
end

"""
    oddsratios(m; level = 0.95) -> OddsRatioTable

Odds ratio estimates with Wald confidence limits (SAS `CLODDS=WALD`, the default table).

- A continuous main effect gets `exp(β)`, for a one-unit increase.
- A categorical main effect gets one row per non-reference level, labelled
  "`var level vs base`". The estimate is the level-versus-reference contrast computed from
  the term's contrast matrix, so it is right under effect coding too (SAS `PARAM=EFFECT`),
  where `exp(β)` is *not* an odds ratio.
- Terms that appear in interactions are omitted, as in SAS.
- Models fitted from a matrix get `exp(β)` for each non-aliased coefficient.
"""
function oddsratios(m::ConditionalLogitModel; level::Real = 0.95)
    _checklevel(level)
    eff = String[]; est = Float64[]; lo = Float64[]; hi = Float64[]
    p = length(m.coef)
    function add!(label, c)
        any(c[m.aliased] .!= 0) && return
        r = oddsratio(m, c; level)
        push!(eff, label); push!(est, r.estimate); push!(lo, r.lower); push!(hi, r.upper)
    end
    unit(j) = (c = zeros(p); c[j] = 1.0; c)
    f = m.formula
    if f === nothing
        for j in 1:p
            add!(m.coefnames[j], unit(j))
        end
    else
        ts = _rhs_terms(f.rhs)
        invars = Set{Symbol}()
        for t in ts
            t isa InteractionTerm && union!(invars, termvars(t))
        end
        col = 0
        for t in ts
            w = StatsModels.width(t)
            if t isa InteractionTerm || any(v -> v in invars, termvars(t))
                # omitted, as in SAS
            elseif t isa CategoricalTerm
                cm = t.contrasts
                levs = cm.levels
                bl = StatsModels.baselevel(cm.contrasts)
                bi = bl === nothing ? 1 : findfirst(isequal(bl), levs)
                M = cm.matrix
                if bi !== nothing && size(M, 1) == size(M, 2) + 1
                    for li in eachindex(levs)
                        li == bi && continue
                        c = zeros(p)
                        c[col+1:col+w] .= M[li, :] .- M[bi, :]
                        add!("$(t.sym) $(levs[li]) vs $(levs[bi])", c)
                    end
                end
            elseif w == 1
                add!(m.coefnames[col+1], unit(col + 1))
            end
            col += w
        end
    end
    return OddsRatioTable(eff, est, lo, hi, Float64(level))
end

_rhs_terms(t::MatrixTerm) = collect(t.terms)
_rhs_terms(t::Tuple) = collect(t)
_rhs_terms(t::AbstractTerm) = [t]
