# Front ends: matrix interface and StatsModels formula / Tables.jl interface.
# Both end in `fit_core`. Missing values are handled here by listwise deletion over
# response, covariates, strata, FREQ and offset (the SAS default).

StatsModels.drop_intercept(::Type{<:ConditionalLogitModel}) = true

const _OPTION_NAMES = fieldnames(FitOptions)
const _FRONTEND_KEYWORDS = "strata, event, freq, weights, offset, strata_missing, " *
                           "contrasts (formula interface), coefnames, responsename, " *
                           "stratalabel (matrix interface)"

function _options(kwargs)
    for k in keys(kwargs)
        k in _OPTION_NAMES ||
            throw(ArgumentError("unknown keyword argument `$k`. Valid keywords: " *
                                _FRONTEND_KEYWORDS * "; fitting options: " *
                                join(_OPTION_NAMES, ", ")))
    end
    nt = NamedTuple(kwargs)
    if haskey(nt, :start) && nt.start !== nothing
        st = Vector{Float64}(nt.start)
        all(isfinite, st) || throw(ArgumentError("start values must be finite; got $st"))
        nt = merge(nt, (start = st,))
    end
    if haskey(nt, :sc_n)
        nt.sc_n isa Union{Symbol,AbstractString} ||
            throw(ArgumentError("sc_n must be :informative or :all, got $(repr(nt.sc_n))"))
        nt = merge(nt, (sc_n = Symbol(nt.sc_n),))
    end
    o = FitOptions(; nt...)
    o.sc_n in (:informative, :all) ||
        throw(ArgumentError("sc_n must be :informative or :all, got :$(o.sc_n)"))
    o.ntasks >= 1 || throw(ArgumentError("ntasks must be ≥ 1, got $(o.ntasks)"))
    o.maxiter >= 0 || throw(ArgumentError("maxiter must be ≥ 0, got $(o.maxiter)"))
    o.maxstephalving >= 0 ||
        throw(ArgumentError("maxstephalving must be ≥ 0, got $(o.maxstephalving)"))
    for (nm, v) in ((:gconv, o.gconv), (:fconv, o.fconv), (:absfconv, o.absfconv),
                    (:xconv, o.xconv))
        v >= 0 || throw(ArgumentError("$nm must be ≥ 0 (0 switches it off), got $v"))
    end
    0 <= o.singular < 1 || throw(ArgumentError("singular must be in [0, 1), got $(o.singular)"))
    return o
end

# Function barriers: the columns of a generic table are not inferable, so the loops over
# their elements live in functions that specialise on the concrete column type.
function _mark_missing!(mask::BitVector, c::AbstractVector)
    Missing <: eltype(c) || return mask
    @inbounds for i in eachindex(c, mask)
        ismissing(c[i]) && (mask[i] = false)
    end
    return mask
end

_take_rows(c::AbstractVector, rows, allrows::Bool) =
    allrows ? (Missing <: eltype(c) ? disallowmissing(c) : c) : disallowmissing(c[rows])

# ---- response / event coding ------------------------------------------------------

function event_indicator(y::AbstractVector, event, responsename::AbstractString)
    n = length(y)
    if event === nothing
        if all(v -> v isa Bool, y)
            return BitVector(y), "$responsename = true"
        end
        for (i, v) in enumerate(y)
            (v isa Real && (v == 0 || v == 1)) ||
                throw(ArgumentError("response must be Bool or numeric 0/1 (found $(repr(v)) at " *
                                    "row $i). Pass `event = <level>` to choose the event level " *
                                    "for other codings"))
        end
        return BitVector(v == 1 for v in y), "$responsename = 1"
    end
    lev = unique(y)
    any(l -> l == event, lev) ||
        throw(ArgumentError("event = $(repr(event)) does not occur in the response; observed " *
                            "levels: $(repr(sort!(map(_plain, lev); by = string)))"))
    length(lev) <= 2 ||
        throw(ArgumentError("the response has $(length(lev)) levels; a binary response is " *
                            "required"))
    ev = BitVector(undef, n)
    for i in 1:n
        ev[i] = y[i] == event
    end
    return ev, "$responsename = $(_plain(event))"
end
_plain(x) = x isa AbstractString ? String(x) : x
_plain(x::Real) = x

# ---- FREQ ---------------------------------------------------------------------------

function _resolve_freq(freq, weights)
    weights === nothing && return freq
    freq === nothing ||
        throw(ArgumentError("give either `freq` or `weights = fweights(...)`, not both"))
    weights isa FrequencyWeights && return weights.values
    throw(ArgumentError("only frequency weights (`weights = fweights(v)`, SAS FREQ) are " *
                        "supported. SAS PROC LOGISTIC does not allow WEIGHT with STRATA, and " *
                        "the exact conditional likelihood has no sampling-weight analogue"))
end

# Integer FREQ values: SAS truncates non-integers and drops values < 1.
function _freq_values(fraw::AbstractVector, n::Int)
    length(fraw) == n || throw(DimensionMismatch("freq has length $(length(fraw)), expected $n"))
    f = zeros(Int, n)
    ok = trues(n)
    warned = false
    for i in 1:n
        v = fraw[i]
        if v === missing
            ok[i] = false
            continue
        end
        v isa Real && isfinite(v) || throw(ArgumentError("freq must be numeric; found $(repr(v))"))
        if v < 1
            # dropped (after truncation the value is < 1)
            t = 0
            ok[i] = false
            if v > 0 && !warned
                @warn "Non-integer FREQ values are truncated to integers (as in SAS)."
                warned = true
            end
            continue
        end
        v < 2.0^53 || throw(ArgumentError("freq value $v at row $i is too large (must be < 2^53)"))
        t = trunc(Int, v)
        if t != v && !warned
            @warn "Non-integer FREQ values are truncated to integers (as in SAS)."
            warned = true
        end
        f[i] = t
        t >= 1 || (ok[i] = false)
    end
    return f, ok
end

# ---- shared finishing step --------------------------------------------------------

function _finish(X::Matrix{Float64}, y::AbstractVector, skeys::AbstractVector,
                 freq::Vector{Int}, offset::Vector{Float64}, event, opts::FitOptions;
                 coefnames, responsename, stratalabel, formula, nobs_read, freq_read)
    isempty(y) &&
        throw(ArgumentError("no observations left after removing rows with missing values " *
                            "in the response, covariates, strata or offset, or with FREQ < 1"))
    isevent, eventlabel = event_indicator(y, event, responsename)
    codes, K = strata_codes(skeys)
    return fit_core(X, isevent, codes, K, freq, offset, opts; coefnames = String.(coefnames),
                    responsename = String(responsename), eventlabel = eventlabel,
                    stratalabel = String(stratalabel), formula = formula,
                    nobs_read = nobs_read, freq_read = freq_read)
end

_freq_read(f::Vector{Int}, fraw) = fraw === nothing ? length(f) :
    sum((max(f[i], 0) for i in eachindex(f) if fraw[i] !== missing); init = 0)

# ---- matrix interface -------------------------------------------------------------

"""
    fit(ConditionalLogitModel, X, y, strata; kwargs...)
    clogit(X, y, strata; kwargs...)

Fit a conditional logit model from a covariate matrix `X` (n×p, no intercept column),
a binary response `y` and a vector of stratum identifiers `strata` (any hashable
values). Rows with `missing` in `X`, `y`, `strata`, `freq` or `offset` are dropped.

`strata` may also be given as a keyword: `clogit(X, y; strata = s)`.

Keywords: `event`, `freq`, `weights` (only `fweights`), `offset`, `strata_missing`,
`coefnames`, `responsename`, plus the fitting options in [`FitOptions`](@ref).
See [`clogit`](@ref).
"""
function fit(::Type{<:ConditionalLogitModel}, X::AbstractMatrix{<:Union{Missing,Real}},
             y::AbstractVector, strata::AbstractVector;
             event = nothing, freq = nothing, weights = nothing, offset = nothing,
             strata_missing::Bool = false, coefnames = nothing, responsename = "y",
             stratalabel = "strata", kwargs...)
    opts = _options(kwargs)
    n, p = size(X)
    length(y) == n || throw(DimensionMismatch("X has $n rows but y has $(length(y)) elements"))
    length(strata) == n ||
        throw(DimensionMismatch("X has $n rows but strata has $(length(strata)) elements"))
    cn = coefnames === nothing ? ["x$j" for j in 1:p] : String.(collect(coefnames))
    fraw = _resolve_freq(freq, weights)
    fvals, fok = fraw === nothing ? (ones(Int, n), trues(n)) : _freq_values(fraw, n)
    ovals = offset === nothing ? nothing : offset
    ovals === nothing || length(ovals) == n ||
        throw(DimensionMismatch("offset has length $(length(ovals)), expected $n"))
    mask = copy(fok)
    _mark_missing!(mask, y)
    strata_missing || _mark_missing!(mask, strata)
    ovals === nothing || _mark_missing!(mask, ovals)
    if Missing <: eltype(X)
        for j in 1:p
            _mark_missing!(mask, view(X, :, j))
        end
    end
    rows = findall(mask)
    Xu = eltype(X) === Float64 ? X[rows, :] : Matrix{Float64}(X[rows, :])   # one copy
    ou = ovals === nothing ? zeros(length(rows)) : Vector{Float64}(ovals[rows])
    return _finish(Xu, disallowmissing(y[rows]), strata[rows], fvals[rows], ou, event, opts;
                   coefnames = cn, responsename = responsename, stratalabel = stratalabel,
                   formula = nothing, nobs_read = n, freq_read = _freq_read(fvals, fraw))
end

fit(::Type{<:ConditionalLogitModel}, X::AbstractMatrix{<:Union{Missing,Real}},
    y::AbstractVector; strata::AbstractVector, kwargs...) =
    fit(ConditionalLogitModel, X, y, strata; kwargs...)

# ---- formula interface ------------------------------------------------------------

_symlist(s::Symbol) = [s]
_symlist(s::AbstractString) = [Symbol(s)]
_symlist(s::Union{AbstractVector,Tuple}) = [Symbol(x) for x in s]

function _column(cols, names, name::Symbol, what)
    name in names ||
        throw(ArgumentError("$what column :$name not found; available columns: " *
                            join(string.(names), ", ")))
    return Tables.getcolumn(cols, name)
end

_combine_keys(c::AbstractVector) = c
_combine_keys(cs::AbstractVector...) = collect(zip(cs...))

"""
    fit(ConditionalLogitModel, formula, data; strata, kwargs...)
    clogit(formula, data; strata, kwargs...)

Fit a conditional logit model by exact conditional maximum likelihood, the model fitted
by SAS `PROC LOGISTIC` with a `STRATA` statement.

- `formula`: a StatsModels `@formula(y ~ x1 + x2 + ...)`. There is no intercept: it is
  conditioned out with the strata, so `y ~ x` and `y ~ 0 + x` are the same model and
  `y ~ 1 + x` is an error. Categorical covariates get full-rank (k−1) contrasts.
- `data`: any Tables.jl table.
- `strata` (required): the column name(s) identifying the strata (matched sets), as a
  `Symbol`, a `String`, or a vector of `Symbol`s or `String`s (combined key, like SAS
  `strata a b;`). It may also be a vector of stratum values (one per row of `data`).
  A vector of `Symbol`s is always read as column names; a vector of `String`s is read as
  column names when every element is a column name and its length differs from the
  number of rows. `strata` may also be passed positionally:
  `fit(ConditionalLogitModel, formula, data, strata)`.

Keywords:
- `event`: the response level that is the event. The default is `true` / `1` for Bool or
  0/1 responses; any other two-level response requires it. SAS without `EVENT=` or
  `DESCENDING` models the *lower* value (0); use `event = 0` to reproduce such output.
- `freq`: FREQ column name or vector. Each row counts as `freq` identical observations.
  Non-integers are truncated; rows with `freq < 1` are dropped. `weights = fweights(v)`
  is accepted as an alias; other weights are rejected.
- `offset`: offset column name or vector (SAS `OFFSET=`).
- `contrasts`: a `Dict` of StatsModels contrasts (e.g. `EffectsCoding(base = last)` to
  reproduce SAS `PARAM=EFFECT`).
- `strata_missing = false`: treat a missing stratum value as a valid stratum (SAS
  `STRATA ... / MISSING`). By default such rows are dropped.
- Fitting options: `maxiter`, `gconv`, `fconv`, `absfconv`, `xconv`, `maxstephalving`,
  `singular`, `dropcollinear`, `ntasks`, `sc_n`, `start`, `strict`, `verbose`; see
  [`FitOptions`](@ref).

Only the columns named in the formula, `strata`, `freq` and `offset` are read from
`data`. Rows with a missing value in any of them are dropped (listwise deletion).
"""
function fit(::Type{<:ConditionalLogitModel}, f::FormulaTerm, data;
             strata, event = nothing, freq = nothing, weights = nothing, offset = nothing,
             contrasts::AbstractDict = Dict{Symbol,Any}(), strata_missing::Bool = false,
             kwargs...)
    opts = _options(kwargs)
    Tables.istable(data) || throw(ArgumentError("data must be a Tables.jl-compatible table"))
    cols = Tables.columns(data)
    cnames = Symbol.(collect(Tables.columnnames(cols)))
    isempty(cnames) && throw(ArgumentError("data has no columns"))
    n = length(Tables.getcolumn(cols, first(cnames)))
    vars = collect(Symbol, termvars(f))
    vcols = AbstractVector[_column(cols, cnames, v, "formula") for v in vars]
    # strata
    isnames = strata isa Union{Symbol,AbstractString} ||
              (strata isa Union{AbstractVector,Tuple} && !isempty(strata) &&
               (all(x -> x isa Symbol, strata) ||
                (all(x -> x isa AbstractString, strata) && length(strata) != n &&
                 all(x -> Symbol(x) in cnames, strata))))
    if isnames
        snames = _symlist(strata)
        scols = [_column(cols, cnames, s, "strata") for s in snames]
        skeys_all = _combine_keys(scols...)
        stratalabel = join(string.(snames), " ")
    elseif strata isa AbstractVector
        length(strata) == n ||
            throw(DimensionMismatch("strata has $(length(strata)) elements, data has $n rows " *
                                    "(to name columns, pass Symbols, e.g. strata = [:a, :b])"))
        scols = [strata]
        skeys_all = strata
        stratalabel = "strata"
    else
        throw(ArgumentError("strata must be a column name (Symbol/String), a vector of " *
                            "column names, or a vector of stratum values"))
    end
    # freq / offset
    fsrc = _resolve_freq(freq, weights)
    fraw = fsrc isa Union{Symbol,AbstractString} ?
           _column(cols, cnames, Symbol(fsrc), "freq") : fsrc
    fvals, fok = fraw === nothing ? (ones(Int, n), trues(n)) : _freq_values(fraw, n)
    oraw = offset isa Union{Symbol,AbstractString} ?
           _column(cols, cnames, Symbol(offset), "offset") : offset
    oraw === nothing || length(oraw) == n ||
        throw(DimensionMismatch("offset has length $(length(oraw)), expected $n"))
    # listwise deletion
    mask = copy(fok)
    for c in vcols
        _mark_missing!(mask, c)
    end
    if !strata_missing
        for c in scols
            _mark_missing!(mask, c)
        end
    end
    oraw === nothing || _mark_missing!(mask, oraw)
    rows = findall(mask)
    isempty(rows) &&
        throw(ArgumentError("no observations left after removing rows with missing values " *
                            "in the response, covariates, strata or offset, or with FREQ < 1"))
    allrows = length(rows) == n
    sub = NamedTuple{Tuple(vars)}(Tuple(_take_rows(c, rows, allrows) for c in vcols))
    f2 = try
        sch = schema(f, sub, contrasts)
        apply_schema(f, sch, ConditionalLogitModel)
    catch e
        if e isa ArgumentError && occursin("intercept", e.msg)
            throw(ArgumentError("conditional logit has no intercept: it is absorbed by the " *
                                "strata (as in SAS PROC LOGISTIC with STRATA). Remove `1 +` " *
                                "from the formula $f"))
        end
        rethrow()
    end
    Xm = modelcols(f2.rhs, sub)
    # copy unless modelcols already produced a fresh Matrix{Float64} (never alias the data)
    X = Xm isa Matrix{Float64} ? Xm :
        Matrix{Float64}(Xm isa AbstractVector ? reshape(Xm, :, 1) : Xm)
    cn = StatsModels.coefnames(f2.rhs)
    cn = cn isa AbstractString ? [String(cn)] : String.(collect(cn))
    if f.lhs isa Term
        y = sub[f.lhs.sym]
        rname = string(f.lhs.sym)
    else
        y = modelcols(f2.lhs, sub)
        rname = string(f.lhs)
    end
    ou = oraw === nothing ? zeros(length(rows)) : Vector{Float64}(_take_rows(oraw, rows, allrows))
    skeys = allrows ? skeys_all : skeys_all[rows]
    return _finish(X, y, skeys, fvals[rows], ou, event, opts;
                   coefnames = cn, responsename = rname, stratalabel = stratalabel,
                   formula = f2, nobs_read = n, freq_read = _freq_read(fvals, fraw))
end

# `strata` given positionally (instead of falling through to StatsModels' generic
# `fit(T, f, data, args...)`, which would wrap the model in a TableRegressionModel).
fit(::Type{<:ConditionalLogitModel}, f::FormulaTerm, data, strata; kwargs...) =
    fit(ConditionalLogitModel, f, data; strata = strata, kwargs...)

"""
    clogit(formula, data; strata, kwargs...)
    clogit(X, y, strata; kwargs...)

Fit a conditional logit model (conditional logistic regression for matched sets).
This is the same as `fit(ConditionalLogitModel, ...)`; see its docstrings for the
keywords.

```julia
m = clogit(@formula(case ~ spontaneous + induced), infert; strata = :stratum)
```
"""
clogit(f::FormulaTerm, data; kwargs...) = fit(ConditionalLogitModel, f, data; kwargs...)
clogit(X::AbstractMatrix{<:Union{Missing,Real}}, y::AbstractVector, strata::AbstractVector;
       kwargs...) = fit(ConditionalLogitModel, X, y, strata; kwargs...)
clogit(X::AbstractMatrix{<:Union{Missing,Real}}, y::AbstractVector; strata::AbstractVector,
       kwargs...) = fit(ConditionalLogitModel, X, y, strata; kwargs...)
