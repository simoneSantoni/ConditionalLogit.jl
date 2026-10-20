# The type-stable core shared by the matrix and formula front ends.

"""
    fit_core(X, isevent, codes, K, freq, offset, opts; names...) -> ConditionalLogitModel

`X` (n×p) and the vectors hold only the rows used (no missing values, FREQ ≥ 1);
`codes` are dense stratum codes in 1:K.
"""
function fit_core(X::Matrix{T}, isevent::BitVector, codes::Vector{Int}, K::Int,
                  freq::Vector{Int}, offset::Vector{T}, opts::FitOptions;
                  coefnames::Vector{String}, responsename::String, eventlabel::String,
                  stratalabel::String, formula::Union{Nothing,FormulaTerm},
                  nobs_read::Int, freq_read::Int) where {T<:AbstractFloat}
    n, p = size(X)
    p >= 1 || throw(ArgumentError("the model has no covariates (p = 0)"))
    length(coefnames) == p ||
        throw(DimensionMismatch("coefnames has $(length(coefnames)) entries for $p columns"))
    opts.sc_n in (:informative, :all) ||
        throw(ArgumentError("sc_n must be :informative or :all, got :$(opts.sc_n)"))
    for j in 1:p, i in 1:n
        isfinite(X[i, j]) ||
            throw(ArgumentError("covariate $(coefnames[j]) contains non-finite value $(X[i, j]) at row $i"))
    end
    all(isfinite, offset) || throw(ArgumentError("offset contains non-finite values"))

    n >= 1 || throw(ArgumentError("no observations to fit"))
    dfull, summary = build_design(X, isevent, codes, K, freq, offset; ntasks = opts.ntasks)
    summary.ninformative > 0 ||
        throw(ArgumentError("no informative strata: each of the $(summary.nstrata) strata has " *
                            "all or none of its observations as events, so the conditional " *
                            "likelihood is constant"))

    aliased = alias_sweep(dfull, opts.singular)
    if any(aliased)
        names = join(coefnames[aliased], ", ")
        opts.dropcollinear ||
            throw(ArgumentError("covariates $names are linear combinations of other covariates " *
                                "within strata (or constant within every stratum); " *
                                "use dropcollinear = true to set them to 0"))
        @warn "Coefficients for $names set to 0: linear combination of other covariates " *
              "within strata, or constant within every stratum (conditioned out with the strata)."
    end
    keep = findall(!, aliased)
    isempty(keep) && throw(ArgumentError("all covariates are aliased with the strata"))
    d = length(keep) == p ? dfull : subset_columns(dfull, keep)

    start = zeros(T, length(keep))
    if opts.start !== nothing
        length(opts.start) == p ||
            throw(DimensionMismatch("start has length $(length(opts.start)), expected $p"))
        all(isfinite, opts.start) ||
            throw(ArgumentError("start values must be finite; got $(opts.start)"))
        start .= opts.start[keep]
    end

    ev = Evaluator(d, opts.ntasks)
    res = newton(ev, opts, start)

    # separation diagnostics
    status = res.status
    converged = res.converged
    message = res.message
    sep = diagnose_separation(d, res, ev)
    if sep !== :none
        converged = false
        status = sep === :complete ? :complete_separation : :quasi_separation
        message = sep === :complete ?
            "Complete separation of data points detected. The maximum likelihood estimate does not exist." :
            "Quasi-complete separation of data points detected. The maximum likelihood estimate may not exist."
    end
    if !converged
        msg = message * " Results shown are based on the last maximum likelihood iteration. " *
              "Validity of the model fit is questionable."
        opts.strict && throw(ConvergenceError(msg))
        @warn msg
    end

    # covariance at the reported estimate
    F = cholesky(Symmetric(res.I), check = false)
    Vk = issuccess(F) ? inv(F) : pinv(Symmetric(res.I))
    Vk = (Vk + Vk') / 2

    β = zeros(T, p); β[keep] .= res.β
    V = fill(T(NaN), p, p); V[keep, keep] .= Vk
    Ifull = zeros(T, p, p); Ifull[keep, keep] .= res.I
    gfull = zeros(T, p); gfull[keep] .= res.g

    # iteration history with full-length parameter vectors (0 for aliased)
    history = map(res.history) do r
        b = zeros(Float64, p); b[keep] .= r.beta
        IterationRecord(r.iter, r.loglik, r.criterion, r.stephalvings, r.ridged, b)
    end
    # `fitted` is computed on first use (see `fitted(m)`), not here
    return ConditionalLogitModel{T}(formula, coefnames, responsename, eventlabel, stratalabel,
                                    β, V, Ifull, gfull, aliased, res.loglik, res.loglik0,
                                    res.score0, res.iterations, converged, status,
                                    res.crit, message, history, summary, nobs_read, n,
                                    freq_read, sum(freq), X, isevent, freq, offset, codes,
                                    Ref{Union{Nothing,Vector{T}}}(nothing), ReentrantLock(),
                                    opts)
end

"""
    compute_fitted(m) -> Vector

Within-stratum inclusion probabilities of the used rows, in data order. The design is
rebuilt from the stored data, so a fit never pays for this unless `fitted` is called.
"""
function compute_fitted(m::ConditionalLogitModel{T}) where {T}
    n = length(m.y)
    K = n == 0 ? 0 : maximum(m.strata)
    d, _ = build_design(m.X, m.y, m.strata, K, m.freq, m.offset; ntasks = m.options.ntasks)
    keep = findall(!, m.aliased)
    length(keep) == size(m.X, 2) || (d = subset_columns(d, keep))
    πd = inclusion_probabilities(d, m.coef[keep]; ntasks = m.options.ntasks)
    fitted = zeros(T, n)
    inf_row = falses(n)
    for (j, i) in enumerate(d.rows)
        fitted[i] = πd[j]
        inf_row[i] = true
    end
    # uninformative strata: deterministic (all events -> 1, no events -> 0)
    for i in 1:n
        inf_row[i] || (fitted[i] = m.y[i] ? one(T) : zero(T))
    end
    return fitted
end
