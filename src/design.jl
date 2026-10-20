# Data layout used by the likelihood kernels, built once per fit.
#
# * Only informative strata (0 < m_k < n_k) are stored; the others contribute exactly 0.
# * Rows are grouped so that each stratum is a contiguous column range of `Xt`.
# * `Xt` is p × N (transposed), so one observation's covariates are contiguous.
# * Covariates are centred within each stratum (FREQ-weighted mean). The conditional
#   likelihood, score and information are exactly invariant to this shift. Centring
#   removes the cancellation that otherwise ruins the information matrix when a
#   covariate has a large stratum-level mean.
# * When m_k > n_k - m_k the kernels work on the complement (the non-events) with
#   x → -x, η → -η, so the cost is O(n_k · min(m_k, n_k - m_k)).

struct StratifiedDesign{T<:AbstractFloat}
    Xt::Matrix{T}              # p × N, centred within stratum
    offset::Vector{T}          # N
    freq::Vector{Int}          # N
    isevent::BitVector         # N
    rows::Vector{Int}          # N: row index in the (post-missing-deletion) data
    starts::Vector{Int}        # K+1: stratum k is columns starts[k]:starts[k+1]-1
    nk::Vector{Int}            # stratum size (sum of FREQ)
    mk::Vector{Int}            # number of events (sum of FREQ)
    meff::Vector{Int}          # min(m_k, n_k - m_k)
    flip::BitVector            # complement used
    tsum::Vector{T}            # Σ_events f x̃  (constant in β)
    osum::T                    # Σ_events f o
    rawss::Vector{T}           # Σ f x² per column before centring (scale for aliasing)
    chunks::Vector{UnitRange{Int}}   # partition of 1:K, a function of the data only
    chunkorder::Vector{Int}    # chunks by decreasing estimated cost (scheduling order)
    maxrows::Int               # max rows (not FREQ) in a stratum
    maxmeff::Int
end

nstrata(d::StratifiedDesign) = length(d.nk)
npar(d::StratifiedDesign) = size(d.Xt, 1)

"""
    strata_codes(keys) -> (codes::Vector{Int}, K::Int)

Dense integer codes 1..K for arbitrary hashable stratum keys, in order of first
appearance. `missing` is an ordinary key here; callers drop it beforehand unless
`strata_missing = true`.
"""
function strata_codes(keys::AbstractVector)
    if eltype(keys) <: Integer && !isempty(keys)
        # dense fast path for integer ids with a moderate range (same codes as below)
        lo, hi = extrema(keys)
        if hi - lo < 4 * length(keys) + 1024
            map = zeros(Int, hi - lo + 1)
            codes = Vector{Int}(undef, length(keys))
            K = 0
            @inbounds for (i, k) in enumerate(keys)
                c = map[k-lo+1]
                c == 0 && (K += 1; c = K; map[k-lo+1] = K)
                codes[i] = c
            end
            return codes, K
        end
    end
    d = Dict{eltype(keys),Int}()
    codes = Vector{Int}(undef, length(keys))
    @inbounds for (i, k) in enumerate(keys)
        codes[i] = get!(d, k, length(d) + 1)
    end
    return codes, length(d)
end

function _summary(nk, mk, nrows)
    pat = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    ninf = 0; finf = 0; funinf = 0; rinf = 0
    for k in eachindex(nk)
        key = (mk[k], nk[k] - mk[k])
        c, f = get(pat, key, (0, 0))
        pat[key] = (c + 1, f + nk[k])
        if 0 < mk[k] < nk[k]
            ninf += 1; finf += nk[k]; rinf += nrows[k]
        else
            funinf += nk[k]
        end
    end
    ks = sort!(collect(keys(pat)); by = x -> (!(0 < x[1] && x[2] > 0), x[1], x[2]))
    pats = [StrataPattern(k[1], k[2], pat[k][1], pat[k][2], k[1] > 0 && k[2] > 0) for k in ks]
    return StrataSummary(pats, length(nk), ninf, length(nk) - ninf, finf, funinf, rinf)
end

"""
    build_design(X, isevent, codes, K, freq, offset; ntasks = 1) -> (design, summary)

Group rows by stratum, drop uninformative strata, centre and transpose. `X` is the n×p
model matrix of the rows used; `codes` are the dense stratum codes. The gather/centre
pass uses up to `ntasks` tasks (the result does not depend on `ntasks`).
"""
function build_design(X::AbstractMatrix{T}, isevent::AbstractVector{Bool},
                      codes::Vector{Int}, K::Int, freq::Vector{Int},
                      offset::Vector{T}; ntasks::Integer = 1) where {T<:AbstractFloat}
    n, p = size(X)
    nk = zeros(Int, K); mk = zeros(Int, K); nrows = zeros(Int, K)
    @inbounds for i in 1:n
        c = codes[i]
        nk[c] += freq[i]
        nrows[c] += 1
        isevent[i] && (mk[c] += freq[i])
    end
    summary = _summary(nk, mk, nrows)
    inf = [0 < mk[c] < nk[c] for c in 1:K]
    # new index for informative strata, in order of first appearance
    newk = zeros(Int, K)
    Kinf = 0
    for c in 1:K
        inf[c] && (Kinf += 1; newk[c] = Kinf)
    end
    starts = ones(Int, Kinf + 1)
    for c in 1:K
        inf[c] && (starts[newk[c]+1] = nrows[c])
    end
    for k in 1:Kinf
        starts[k+1] += starts[k]
    end
    N = starts[end] - 1
    # counting sort of the informative rows (stable: original order within a stratum)
    pos = starts[1:Kinf]
    rows = Vector{Int}(undef, N)
    @inbounds for i in 1:n
        k = newk[codes[i]]
        k == 0 && continue
        rows[pos[k]] = i
        pos[k] += 1
    end
    Xt = Matrix{T}(undef, p, N)
    off = Vector{T}(undef, N)
    fr = Vector{Int}(undef, N)
    ev = BitVector(undef, N)
    nkI = Vector{Int}(undef, Kinf); mkI = Vector{Int}(undef, Kinf)
    for c in 1:K
        inf[c] || continue
        nkI[newk[c]] = nk[c]; mkI[newk[c]] = mk[c]
    end
    @inbounds for j in 1:N
        i = rows[j]
        off[j] = offset[i]; fr[j] = freq[i]; ev[j] = isevent[i]
    end
    # gather, transpose and centre within stratum (FREQ-weighted mean). Strata are
    # independent, so the task-parallel loop is deterministic.
    _gather_centre!(Xt, X, rows, starts, fr, nkI, ntasks)
    rawss = zeros(T, p)
    @inbounds for r in 1:p
        acc = zero(T)
        for j in 1:N
            i = rows[j]
            acc += fr[j] * X[i, r]^2
        end
        rawss[r] = acc
    end
    meff = [min(mkI[k], nkI[k] - mkI[k]) for k in 1:Kinf]
    flip = BitVector([mkI[k] > nkI[k] - mkI[k] for k in 1:Kinf])
    tsum = zeros(T, p); osum = zero(T)
    @inbounds for j in 1:N
        if ev[j]
            for r in 1:p
                tsum[r] += fr[j] * Xt[r, j]
            end
            osum += fr[j] * off[j]
        end
    end
    maxrows = Kinf == 0 ? 0 : maximum(starts[k+1] - starts[k] for k in 1:Kinf)
    maxmeff = Kinf == 0 ? 0 : maximum(meff)
    chunks, order = make_chunks(starts, nkI, meff, p)
    d = StratifiedDesign{T}(Xt, off, fr, ev, rows, starts, nkI, mkI, meff, flip, tsum,
                            osum, rawss, chunks, order, maxrows, maxmeff)
    return d, summary
end

# Task-based (not `@threads :static`), so it works when the fit itself runs inside a
# threaded loop or a spawned task. Block t handles strata t, t+nb, ...; every stratum
# is written by exactly one block, so the result does not depend on nb.
function _gather_centre!(Xt::Matrix{T}, X::AbstractMatrix{T}, rows::Vector{Int},
                         starts::Vector{Int}, fr::Vector{Int}, nkI::Vector{Int},
                         ntasks::Integer) where {T}
    p = size(Xt, 1)
    Kinf = length(nkI)
    nb = clamp(Int(ntasks), 1, max(Kinf, 1))
    if nb == 1
        _gather_centre_block!(Xt, X, rows, starts, fr, nkI, 1, 1, p)
    else
        @sync for t in 1:nb
            Threads.@spawn _gather_centre_block!(Xt, X, rows, starts, fr, nkI, t, nb, p)
        end
    end
    return Xt
end

function _gather_centre_block!(Xt::Matrix{T}, X::AbstractMatrix{T}, rows, starts, fr, nkI,
                               t::Int, nb::Int, p::Int) where {T}
    xbar = zeros(T, p)
    @inbounds for k in t:nb:length(nkI)
        fill!(xbar, zero(T))
        for j in starts[k]:starts[k+1]-1
            i = rows[j]
            for r in 1:p
                Xt[r, j] = X[i, r]
                xbar[r] += fr[j] * X[i, r]
            end
        end
        xbar ./= nkI[k]
        for j in starts[k]:starts[k+1]-1, r in 1:p
            Xt[r, j] -= xbar[r]
        end
    end
    return nothing
end

"""
    make_chunks(starts, nk, meff, p) -> (chunks, order)

Partition the strata into contiguous chunks of roughly equal estimated cost. A stratum
whose cost alone reaches the per-chunk target gets a chunk of its own. `order` lists the
chunks by decreasing cost (longest-processing-time-first scheduling). Everything depends only on the data (never on the
number of threads); this is what makes threaded results bitwise reproducible. The chunk
count is capped so that the per-chunk p×p result slots stay below about 64 MiB.
"""
function make_chunks(starts, nk, meff, p)
    K = length(nk)
    K == 0 && return UnitRange{Int}[], Int[]
    cost = Vector{Float64}(undef, K)
    pp = p * (p + 1) / 2 + p + 1
    for k in 1:K
        cost[k] = (starts[k+1] - starts[k]) * p + nk[k] * max(meff[k], 1) * pp
    end
    total = sum(cost)
    kmax = clamp(floor(Int, 32 * 2^20 / (8 * max(p, 1)^2)), 1, 256)
    nchunks = clamp(ceil(Int, total / 2^15), 1, min(kmax, K))
    target = total / nchunks
    chunks = UnitRange{Int}[]
    lo = 1; acc = 0.0
    for k in 1:K
        if cost[k] >= target && k > lo          # heavy stratum: close the open chunk
            push!(chunks, lo:k-1); lo = k; acc = 0.0
        end
        acc += cost[k]
        if acc >= target
            push!(chunks, lo:k); lo = k + 1; acc = 0.0
        end
    end
    lo <= K && push!(chunks, lo:K)
    ccost = [sum(view(cost, c)) for c in chunks]
    order = sortperm(ccost; rev = true, alg = Base.Sort.DEFAULT_STABLE)
    return chunks, order
end

"""
    alias_sweep(d, singular) -> BitVector

Identify covariates that are linear combinations of earlier covariates *within strata*.
This includes covariates that are constant within every informative stratum, which are
collinear with the conditioned-out stratum intercepts. The sweep runs in column order,
so the later column of a dependent set is dropped, as SAS and R do. It works on the
FREQ-weighted Gram matrix of the within-stratum-centred design. That matrix has the same
null space as the conditional information matrix at any β.
"""
function alias_sweep(d::StratifiedDesign{T}, singular::Real) where {T}
    p = npar(d)
    A = weighted_gram(d)
    Xt = d.Xt
    aliased = falses(p)
    keep = Int[]
    Lf = zeros(T, p, p)   # lower Cholesky factor of A[keep, keep], grown column by column
    z = zeros(T, p)
    for j in 1:p
        ajj = A[j, j]
        # exactly constant within strata: centred SS is at rounding level of raw SS
        if !(ajj > (1e3 * eps(T))^2 * max(d.rawss[j], floatmin(T)))
            aliased[j] = true
            continue
        end
        q = length(keep)
        for (t, kt) in enumerate(keep)
            s = A[min(kt, j), max(kt, j)]
            for u in 1:t-1
                s -= Lf[t, u] * z[u]
            end
            z[t] = s / Lf[t, t]
        end
        r = ajj - sum(abs2, view(z, 1:q))
        if r <= singular * ajj
            aliased[j] = true
        else
            push!(keep, j)
            for u in 1:q
                Lf[q+1, u] = z[u]
            end
            Lf[q+1, q+1] = sqrt(r)
        end
    end
    return aliased
end

"""
    weighted_gram(d) -> A

Upper triangle of Σ_i f_i x̃_i x̃_i' (the FREQ-weighted Gram matrix of the centred
design), accumulated in fixed-size column blocks with a rank-k update (BLAS `syrk` for
BLAS element types). It is called outside the threaded kernels.
"""
function weighted_gram(d::StratifiedDesign{T}) where {T}
    p = npar(d)
    Xt = d.Xt
    N = size(Xt, 2)
    A = zeros(T, p, p)
    N == 0 && return A
    B = min(N, max(64, 2^18 ÷ max(p, 1)))
    W = Matrix{T}(undef, p, B)
    for lo in 1:B:N
        hi = min(lo + B - 1, N)
        nb = hi - lo + 1
        @inbounds for j in lo:hi
            sf = sqrt(T(d.freq[j]))
            @simd for r in 1:p
                W[r, j-lo+1] = sf * Xt[r, j]
            end
        end
        Wv = view(W, :, 1:nb)
        if T <: LinearAlgebra.BlasFloat
            BLAS.syrk!('U', 'N', one(T), Wv, one(T), A)
        else
            mul!(A, Wv, transpose(Wv), true, true)
        end
    end
    return A
end

"""
    subset_columns(d, keep)

The same design restricted to the covariates `keep` (used after the aliasing sweep).
"""
function subset_columns(d::StratifiedDesign{T}, keep::AbstractVector{Int}) where {T}
    Xt = d.Xt[keep, :]
    chunks, order = make_chunks(d.starts, d.nk, d.meff, length(keep))
    return StratifiedDesign{T}(Xt, d.offset, d.freq, d.isevent, d.rows, d.starts, d.nk,
                               d.mk, d.meff, d.flip, d.tsum[keep], d.osum, d.rawss[keep],
                               chunks, order, d.maxrows, d.maxmeff)
end
