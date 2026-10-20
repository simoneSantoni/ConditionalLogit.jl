# Core types: fitting options, strata summary, iteration history and the model.

"""
    FitOptions

Options for the Newton–Raphson fit. The defaults are the SAS PROC LOGISTIC
defaults for a conditional (STRATA) analysis. Every field can be passed as a
keyword to [`clogit`](@ref) / `fit`.

- `maxiter = 25`: maximum number of Newton–Raphson iterations (SAS `MAXITER=`).
- `gconv = 1e-8`: relative gradient criterion `g'I⁻¹g / (|ℓ| + 1e-6)` (SAS `GCONV=`).
- `fconv`, `absfconv`, `xconv` (all `0.0`, which means off): the SAS `FCONV=`, `ABSFCONV=`
  and `XCONV=` criteria. As SAS does for STRATA models, an *exactly* zero change in the
  log likelihood or in the parameters between iterations also stops the iterations.
- `maxstephalving = 50`: maximum number of step-halvings when a Newton step decreases ℓ.
  Halving also stops once the step is negligible relative to β.
- `singular = 1e7 * eps() ≈ 2.2e-9`: tolerance of the in-order collinearity sweep. The
  default is the SAS `SINGULAR=` default. SAS tests the sweep pivot of the Hessian; this
  package tests 1 − R² of each covariate on the earlier ones in the within-stratum
  centred (FREQ-weighted) Gram matrix. Both have the same null space, but borderline
  cases can be decided differently.
- `dropcollinear = true`: set aliased coefficients to 0 (SAS behaviour); `false` throws.
- `ntasks = Threads.nthreads()`: number of tasks for the per-stratum work (≥ 1). The
  results are bitwise identical for any value. Fits may themselves run concurrently
  (e.g. inside `Threads.@threads` or `Threads.@spawn`).
- `sc_n = :informative`: which `n` enters SC/BIC = -2LogL + p·log(n). `:informative`
  counts observations (sum of FREQ) in informative strata, and `:all` counts every
  observation used. They differ only when there are uninformative strata.
- `start = nothing`: starting values (default 0, as in SAS). They must be finite. If
  ℓ(start) < ℓ(0), the start is discarded with a warning and the fit starts from 0.
- `strict = false`: throw a `ConvergenceError` instead of warning when the fit did not
  converge or the data are separated.
- `verbose = false`: print the iteration history while fitting.
"""
Base.@kwdef struct FitOptions
    maxiter::Int = 25
    gconv::Float64 = 1e-8
    fconv::Float64 = 0.0
    absfconv::Float64 = 0.0
    xconv::Float64 = 0.0
    maxstephalving::Int = 50
    singular::Float64 = 1e7 * eps(Float64)
    dropcollinear::Bool = true
    ntasks::Int = Threads.nthreads()
    sc_n::Symbol = :informative
    start::Union{Nothing,Vector{Float64}} = nothing
    strict::Bool = false
    verbose::Bool = false
end

"""
    ConvergenceError(msg)

Thrown when `strict = true` and the Newton–Raphson iterations did not converge, or the
maximum likelihood estimate does not exist (separation).
"""
struct ConvergenceError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ConvergenceError) = print(io, "ConvergenceError: ", e.msg)

"""
    ResultTable

Supertype of the small result tables (strata summary, fit statistics, global tests, odds
ratios). Each implements the Tables.jl column interface, so `DataFrame(t)` works.
"""
abstract type ResultTable end

"""
    StrataPattern

One row of the SAS "Strata Summary" table: strata with `nevents` events and
`nnonevents` non-events (both counted with FREQ).
"""
struct StrataPattern
    nevents::Int
    nnonevents::Int
    nstrata::Int
    freq::Int
    informative::Bool
end

"""
    StrataSummary

Counts of strata by response pattern (SAS "Strata Summary"). Strata in which every
observation is an event, or none is, are *uninformative*. Their conditional likelihood
is identically 1, so they are reported here but not used in the analysis.

It is a Tables.jl table with one row per response pattern and columns `pattern`,
`events`, `nonevents`, `nstrata`, `frequency` and `informative`.
"""
struct StrataSummary <: ResultTable
    patterns::Vector{StrataPattern}
    nstrata::Int
    ninformative::Int
    nuninformative::Int
    freq_informative::Int
    freq_uninformative::Int
    rows_informative::Int
end

"""
    IterationRecord

One Newton–Raphson iteration: iteration number, log-likelihood and relative gradient
criterion at the iterate, the number of step-halvings used to reach it, whether the
information matrix had to be ridged, and the parameter vector (full length, in the order
of `coefnames`, with 0 for aliased coefficients).
"""
struct IterationRecord
    iter::Int
    loglik::Float64
    criterion::Float64
    stephalvings::Int
    ridged::Bool
    beta::Vector{Float64}
end

"""
    ConditionalLogitModel

A fitted conditional logit model. It is created by [`clogit`](@ref) or
`fit(ConditionalLogitModel, ...)`.
"""
struct ConditionalLogitModel{T<:AbstractFloat} <: RegressionModel
    formula::Union{Nothing,FormulaTerm}
    coefnames::Vector{String}
    responsename::String
    eventlabel::String
    stratalabel::String
    coef::Vector{T}
    vcov::Matrix{T}
    information::Matrix{T}
    gradient::Vector{T}
    aliased::BitVector
    loglik::T
    loglik0::T
    score_chisq::T
    iterations::Int
    converged::Bool
    status::Symbol
    criterion::T
    convergence_message::String
    history::Vector{IterationRecord}
    summary::StrataSummary
    nobs_read::Int
    nobs_used::Int
    freq_read::Int
    freq_used::Int
    X::Matrix{T}
    y::BitVector
    freq::Vector{Int}
    offset::Vector{T}
    strata::Vector{Int}
    fittedcache::Base.RefValue{Union{Nothing,Vector{T}}}   # computed on first use
    fittedlock::ReentrantLock
    options::FitOptions
end
