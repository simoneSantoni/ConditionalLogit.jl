# Exact conditional log-likelihood with analytic score and information.
#
# For an informative stratum with rows R (row i repeated f_i times), m events and
# linear predictor η_i = x_i'β + o_i, the conditional likelihood is
#
#     L_k(β) = exp(Σ_{events} f_i η_i) / B,   B = Σ_{S ⊂ R, |S| = m} exp(Σ_{j ∈ S} η_j),
#
# the likelihood that SAS PROC LOGISTIC maximises when a STRATA statement is present.
# (It equals the Cox partial likelihood with "exact"/"discrete" ties.) Then
#
#     ℓ_k = Σ_events f η − log B,   ∇ℓ_k = Σ_events f x − E[x_S],   -∇²ℓ_k = Var[x_S],
#
# where S is drawn with P(S) ∝ exp(η_S) over m-subsets.
#
# B is computed with the Gail–Lubin–Rubinstein (1981) recursion B(i,j) = B(i-1,j) +
# e^{η_i} B(i-1,j-1). Here it runs in a normalised "mixture" form that carries
# L = log B(i,j), M = E[x_S] and C = Var[x_S] for the first i items and subset size j.
# Adding item i mixes "i ∉ S" (weight B(i-1,j)) with "i ∈ S" (weight e^{η_i}B(i-1,j-1)):
#
#     L' = logaddexp(L(i-1,j), L(i-1,j-1) + η_i),   r = exp(L(i-1,j-1) + η_i - L')
#     d  = M(i-1,j-1) + x_i - M(i-1,j)
#     M' = M(i-1,j) + r d
#     C' = (1-r) C(i-1,j) + r C(i-1,j-1) + r(1-r) d d'
#
# Only convex combinations and a PSD rank-one term appear, so nothing can overflow and
# the information is positive semidefinite by construction. Cost is O(n m p²) for
# the Hessian, O(n m p) for the score and O(n m) for the value. m = 1 strata use a
# two-pass centred softmax (O(n p²)), which is McFadden's conditional logit.

# Branch-consistent logaddexp: the same branch is taken for ties, so ForwardDiff
# derivatives are right at β = 0, where all η are tied.
@inline function logaddexp_(a::Real, b::Real)
    return a >= b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))
end
# -Inf-safe variant (used outside the hot path)
@inline function logaddexp_safe(a::Real, b::Real)
    a == -Inf && return b
    b == -Inf && return a
    return logaddexp_(a, b)
end

struct Workspace{T}
    η::Vector{T}       # maxrows
    w::Vector{T}       # maxrows (softmax weights)
    L::Vector{T}       # maxmeff + 1
    M::Matrix{T}       # p × (maxmeff + 1)
    C::Array{T,3}      # p × p × (capC + 1), upper triangle used
    xs::Vector{T}      # p: sign-adjusted x_i
    dv::Vector{T}      # p
    mu::Vector{T}      # p: E[x_S] of the current stratum
    colsum::Vector{T}  # p
    g::Vector{T}       # p: chunk accumulator of E[x_S]
    H::Matrix{T}       # p × p: chunk accumulator of Var[x_S] (upper)
    capC::Int          # largest meff the per-task C tensor can hold
    nsplit::Int        # tasks available to split one heavy stratum (see _glr_split!)
end

function Workspace{T}(p::Int, maxrows::Int, maxmeff::Int, capC::Int = maxmeff,
                      nsplit::Int = 1) where {T}
    mC = capC >= 2 ? capC + 1 : 0
    return Workspace{T}(zeros(T, maxrows), zeros(T, maxrows), zeros(T, maxmeff + 1),
                        zeros(T, p, maxmeff + 1), zeros(T, p, p, mC), zeros(T, p),
                        zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p), zeros(T, p, p),
                        capC, nsplit)
end

# A per-task C tensor is p × p × (meff + 1). It is sized for the largest stratum whose
# tensor fits in _C_BYTES_PER_TASK; larger strata ("heavy") are evaluated by
# `_glr_split!`, which allocates the tensor per call and splits its columns over tasks.
# So per-task memory no longer grows with ntasks × p² × (largest meff).
const _C_BYTES_PER_TASK = 8 * 2^20

function _c_capacity(::Type{T}, p::Int, meff::AbstractVector{Int}) where {T}
    lim = _C_BYTES_PER_TASK ÷ (sizeof(T) * max(p, 1)^2) - 1
    cap = 0
    for m in meff
        m <= lim && (cap = max(cap, m))
    end
    return cap
end

# Work (flops) above which one stratum's information is split over tasks, when p is
# large enough for the O(n m p²) part to dominate the O(n m p) recomputation per task.
const _SPLIT_FLOPS = 2^25

"""
    stratum!(ws, d, k, β, Val(ORD)) -> log B_k

Evaluate one informative stratum. Returns log B_k (the log denominator). With
ORD ≥ 1 it adds E[x_S] to `ws.g`; with ORD = 2 it also adds Var[x_S] to the upper
triangle of `ws.H`. This function does not allocate.
"""
function stratum!(ws::Workspace{T}, d::StratifiedDesign{T}, k::Int, β::AbstractVector{T},
                  ::Val{ORD}) where {T,ORD}
    Xt = d.Xt
    p = size(Xt, 1)
    a = d.starts[k]; b = d.starts[k+1] - 1
    flip = d.flip[k]
    s = flip ? -one(T) : one(T)
    η = ws.η
    etaall = zero(T)
    @inbounds for i in a:b
        acc = d.offset[i]
        @simd for r in 1:p
            acc = muladd(Xt[r, i], β[r], acc)
        end
        η[i-a+1] = acc
        etaall += d.freq[i] * acc
    end
    meff = d.meff[k]
    logB = if meff == 1
        _softmax!(ws, d, a, b, s, Val(ORD))
    elseif ORD >= 2 && (meff > ws.capC ||
                        (ws.nsplit > 1 && p >= 8 && d.nk[k] * meff * p^2 >= _SPLIT_FLOPS))
        _glr_split!(ws, d, a, b, s, meff, d.nk[k])
    else
        _glr!(ws, d, a, b, s, meff, d.nk[k], Val(ORD))
    end
    if flip
        # B_m(η) = exp(Σ_all f η) · B_{n-m}(-η);  E[x_S] = Σ_all f x - E[x_T]
        logB += etaall
        if ORD >= 1
            cs = ws.colsum
            fill!(cs, zero(T))
            @inbounds for i in a:b
                f = d.freq[i]
                @simd for r in 1:p
                    cs[r] = muladd(f, Xt[r, i], cs[r])
                end
            end
            ws.mu .+= cs        # ws.mu held E[Σ_T (-x)]
        end
    end
    if ORD >= 1
        ws.g .+= ws.mu
    end
    return logB
end

# m_eff = 1: softmax over rows with weights f_i e^{s η_i}.
function _softmax!(ws::Workspace{T}, d::StratifiedDesign{T}, a::Int, b::Int, s::T,
                   ::Val{ORD}) where {T,ORD}
    Xt = d.Xt
    p = size(Xt, 1)
    η = ws.η; w = ws.w
    nr = b - a + 1
    emax = typemin(T)
    @inbounds for i in 1:nr
        emax = max(emax, s * η[i])
    end
    S = zero(T)
    @inbounds for i in 1:nr
        w[i] = d.freq[a+i-1] * exp(s * η[i] - emax)
        S += w[i]
    end
    logB = emax + log(S)
    ORD == 0 && return logB
    mu = ws.mu
    fill!(mu, zero(T))
    invS = inv(S)
    @inbounds for i in 1:nr
        w[i] *= invS
        wi = w[i] * s
        @simd for r in 1:p
            mu[r] = muladd(wi, Xt[r, a+i-1], mu[r])
        end
    end
    if ORD >= 2
        z = ws.xs; H = ws.H
        @inbounds for i in 1:nr
            wi = w[i]
            for r in 1:p
                z[r] = s * Xt[r, a+i-1] - mu[r]
            end
            for c in 1:p
                zc = wi * z[c]
                @simd for r in 1:c
                    H[r, c] = muladd(z[r], zc, H[r, c])
                end
            end
        end
    end
    return logB
end

# General m_eff ≥ 2: normalised GLR mixture recursion (see the file header).
function _glr!(ws::Workspace{T}, d::StratifiedDesign{T}, a::Int, b::Int, s::T, m::Int,
               n::Int, ::Val{ORD}) where {T,ORD}
    p = size(d.Xt, 1)
    logB = _glr_core!(ws.L, ws.M, ws.C, ws.xs, ws.dv, ws.η, d, a, b, s, m, n, Base.OneTo(p),
                      Val(ORD))
    if ORD >= 1
        mu = ws.mu; M = ws.M
        @inbounds for r in 1:p
            mu[r] = M[r, m+1]
        end
    end
    if ORD >= 2
        H = ws.H; C = ws.C
        @inbounds for c in 1:p
            @simd for r in 1:c
                H[r, c] += C[r, c, m+1]
            end
        end
    end
    return logB
end

# The recursion. With ORD = 2 it updates only the columns `cols` of the (upper
# triangular) covariance, stored as C[r, c - first(cols) + 1, j]. The value of every
# entry depends only on (r, c, j) and the L/M recursion, never on `cols`, so splitting
# the columns over tasks gives bitwise the same result as one pass.
function _glr_core!(L::AbstractVector{T}, M::AbstractMatrix{T}, C::AbstractArray{T,3},
                    xs::Vector{T}, dv::Vector{T}, η::Vector{T}, d::StratifiedDesign{T},
                    a::Int, b::Int, s::T, m::Int, n::Int, cols::AbstractUnitRange{Int},
                    ::Val{ORD}) where {T,ORD}
    Xt = d.Xt
    p = size(Xt, 1)
    c0 = first(cols) - 1
    L[1] = zero(T)
    if ORD >= 1
        @inbounds for r in 1:p
            M[r, 1] = zero(T)
        end
    end
    if ORD >= 2
        @inbounds for c in cols, r in 1:c
            C[r, c-c0, 1] = zero(T)
        end
    end
    cnt = 0
    @inbounds for i in a:b
        ηi = s * η[i-a+1]
        if ORD >= 1
            for r in 1:p
                xs[r] = s * Xt[r, i]
            end
        end
        for _ in 1:d.freq[i]
            cnt += 1
            jtop = min(cnt, m)
            jlow = max(1, m - (n - cnt))
            for j in jtop:-1:jlow
                if j == cnt
                    # first time subset size j is reachable: item is necessarily in S
                    L[j+1] = L[j] + ηi
                    if ORD >= 1
                        @simd for r in 1:p
                            M[r, j+1] = M[r, j] + xs[r]
                        end
                    end
                    if ORD >= 2
                        for c in cols
                            cc = c - c0
                            @simd for r in 1:c
                                C[r, cc, j+1] = C[r, cc, j]
                            end
                        end
                    end
                else
                    bb = L[j] + ηi
                    Lnew = logaddexp_(L[j+1], bb)
                    rr = exp(bb - Lnew)
                    L[j+1] = Lnew
                    if ORD >= 1
                        @simd for r in 1:p
                            dv[r] = M[r, j] + xs[r] - M[r, j+1]
                        end
                        @simd for r in 1:p
                            M[r, j+1] = muladd(rr, dv[r], M[r, j+1])
                        end
                    end
                    if ORD >= 2
                        q = 1 - rr
                        rq = rr * q
                        for c in cols
                            cc = c - c0
                            dc = rq * dv[c]
                            @simd for r in 1:c
                                C[r, cc, j+1] = muladd(dv[r], dc,
                                                       muladd(q, C[r, cc, j+1], rr * C[r, cc, j]))
                            end
                        end
                    end
                end
            end
        end
    end
    return L[m+1]
end

# Column blocks of the upper triangle with roughly equal work (column c has c entries).
function _col_blocks(p::Int, nb::Int)
    nb = clamp(nb, 1, p)
    total = p * (p + 1) / 2
    blocks = UnitRange{Int}[]
    lo = 1; acc = 0.0
    for c in 1:p
        acc += c
        if acc >= total * length(blocks) / nb + total / nb - 1e-9 && length(blocks) < nb - 1
            push!(blocks, lo:c); lo = c + 1
        end
    end
    lo <= p && push!(blocks, lo:p)
    return blocks
end

# ORD = 2 for one heavy stratum: the columns of C are split over up to `ws.nsplit` tasks.
# Each task reruns the cheap L/M recursion (O(n m p)) and updates its own column block
# of C (O(n m p²) in total); the tensors are allocated per call, so per-task workspaces
# need not be sized for the largest stratum. Bitwise identical to `_glr!`.
function _glr_split!(ws::Workspace{T}, d::StratifiedDesign{T}, a::Int, b::Int, s::T,
                     m::Int, n::Int)::T where {T}
    p = size(d.Xt, 1)
    blocks = _col_blocks(p, ws.nsplit)
    η = ws.η
    if length(blocks) == 1
        lb, Mb, Cb = _glr_block(η, d, a, b, s, m, n, blocks[1])
        _addH!(ws.H, blocks[1], Cb, m)
    else
        tasks = [Threads.@spawn _glr_block(η, d, a, b, s, m, n, cols) for cols in blocks]
        R = Tuple{T,Matrix{T},Array{T,3}}
        lb, Mb, Cb = fetch(tasks[1])::R
        _addH!(ws.H, blocks[1], Cb, m)
        for t in 2:length(blocks)
            Ct = (fetch(tasks[t])::R)[3]
            _addH!(ws.H, blocks[t], Ct, m)
        end
    end
    mu = ws.mu
    @inbounds for r in 1:p
        mu[r] = Mb[r, m+1]
    end
    return lb
end

function _glr_block(η::Vector{T}, d::StratifiedDesign{T}, a::Int, b::Int, s::T, m::Int,
                    n::Int, cols::UnitRange{Int}) where {T}
    p = size(d.Xt, 1)
    L = Vector{T}(undef, m + 1)
    M = Matrix{T}(undef, p, m + 1)
    C = Array{T,3}(undef, p, length(cols), m + 1)
    xs = Vector{T}(undef, p); dv = Vector{T}(undef, p)
    logB = _glr_core!(L, M, C, xs, dv, η, d, a, b, s, m, n, cols, Val(2))
    return logB, M, C
end

function _addH!(H::Matrix{T}, cols::UnitRange{Int}, C::Array{T,3}, m::Int) where {T}
    c0 = first(cols) - 1
    @inbounds for c in cols
        @simd for r in 1:c
            H[r, c] += C[r, c-c0, m+1]
        end
    end
    return H
end

# ---------------------------------------------------------------------------------
# Evaluator: threaded, deterministic accumulation over strata

struct Evaluator{T}
    d::StratifiedDesign{T}
    ws::Vector{Workspace{T}}
    Lc::Vector{T}         # per-chunk Σ log B
    Gc::Matrix{T}         # p × nchunks
    Hc::Array{T,3}        # p × p × nchunks (upper)
    g::Vector{T}          # score
    I::Matrix{T}          # information (full symmetric)
end

function Evaluator(d::StratifiedDesign{T}, ntasks::Int) where {T}
    p = npar(d)
    nc = length(d.chunks)
    nt = clamp(ntasks, 1, max(nc, 1))
    capC = _c_capacity(T, p, d.meff)
    nsplit = clamp(ntasks, 1, max(p ÷ 4, 1))     # each block also reruns the O(nmp) recursion
    ws = [Workspace{T}(p, d.maxrows, d.maxmeff, capC, nsplit) for _ in 1:nt]
    return Evaluator{T}(d, ws, zeros(T, nc), zeros(T, p, nc), zeros(T, p, p, nc),
                        zeros(T, p), zeros(T, p, p))
end

function _chunk!(ev::Evaluator{T}, ws::Workspace{T}, c::Int, β, ::Val{ORD}) where {T,ORD}
    p = npar(ev.d)
    ORD >= 1 && fill!(ws.g, zero(T))
    ORD >= 2 && fill!(ws.H, zero(T))
    lc = zero(T)
    for k in ev.d.chunks[c]
        lc += stratum!(ws, ev.d, k, β, Val(ORD))
    end
    ev.Lc[c] = lc
    if ORD >= 1
        @inbounds for r in 1:p
            ev.Gc[r, c] = ws.g[r]
        end
    end
    if ORD >= 2
        @inbounds for cc in 1:p, r in 1:cc
            ev.Hc[r, cc, c] = ws.H[r, cc]
        end
    end
    return nothing
end

function _worker!(ev::Evaluator, ws::Workspace, counter::Atomic{Int}, nc::Int, β,
                  ::Val{ORD}) where {ORD}
    order = ev.d.chunkorder
    while true
        t = atomic_add!(counter, 1)
        t > nc && break
        _chunk!(ev, ws, order[t], β, Val(ORD))
    end
    return nothing
end

"""
    evaluate!(ev, β, Val(ORD)) -> ℓ

Exact conditional log-likelihood at β. With ORD ≥ 1 it fills `ev.g` with the score;
with ORD = 2 it also fills `ev.I` with the information (minus the Hessian). Chunks of
strata are processed by up to `length(ev.ws)` tasks with dynamic scheduling (most
expensive chunks first), and the per-chunk results are reduced in a fixed order. The result is therefore bitwise
identical for any number of tasks or threads.
"""
function evaluate!(ev::Evaluator{T}, β::AbstractVector{T}, ::Val{ORD}) where {T,ORD}
    d = ev.d
    nc = length(d.chunks)
    nt = min(length(ev.ws), nc)
    if nt <= 1
        for c in 1:nc
            _chunk!(ev, ev.ws[1], c, β, Val(ORD))
        end
    else
        counter = Atomic{Int}(1)
        @sync for t in 1:nt
            Threads.@spawn _worker!(ev, ev.ws[t], counter, nc, β, Val(ORD))
        end
    end
    p = npar(d)
    ℓ = d.osum + dot(d.tsum, β)
    for c in 1:nc
        ℓ -= ev.Lc[c]
    end
    if ORD >= 1
        g = ev.g
        copyto!(g, d.tsum)
        @inbounds for c in 1:nc, r in 1:p
            g[r] -= ev.Gc[r, c]
        end
    end
    if ORD >= 2
        I = ev.I
        fill!(I, zero(T))
        @inbounds for c in 1:nc, cc in 1:p, r in 1:cc
            I[r, cc] += ev.Hc[r, cc, c]
        end
        @inbounds for cc in 1:p, r in 1:cc-1
            I[cc, r] = I[r, cc]
        end
    end
    return ℓ
end

# ---------------------------------------------------------------------------------
# Generic (allocating) value-only path. It does not use the complement trick or the
# softmax path, and it accepts any Real element type (ForwardDiff.Dual, BigFloat).
# Tests use it as an independent oracle for the analytic score and information.

"""
    logB_generic(η, f, m)

`log Σ_{|S|=m} exp(η_S)` over the multiset in which item i appears `f[i]` times.
"""
function logB_generic(η::AbstractVector{S}, f::AbstractVector{<:Integer}, m::Integer) where {S<:Real}
    n = sum(f)
    L = Vector{S}(undef, m + 1)
    L[1] = zero(S)
    cnt = 0
    for i in eachindex(η), _ in 1:f[i]
        cnt += 1
        for j in min(cnt, m):-1:max(1, m - (n - cnt))
            L[j+1] = j == cnt ? L[j] + η[i] : logaddexp_(L[j+1], L[j] + η[i])
        end
    end
    return L[m+1]
end

"""
    loglik_generic(d, β)

Exact conditional log-likelihood of design `d` at `β` by the generic recursion
(no complement trick, no fast paths). For testing.
"""
function loglik_generic(d::StratifiedDesign, β::AbstractVector{S}) where {S<:Real}
    ℓ = zero(S)
    for k in 1:nstrata(d)
        rr = d.starts[k]:d.starts[k+1]-1
        η = [d.offset[i] + sum(d.Xt[r, i] * β[r] for r in eachindex(β)) for i in rr]
        ev = sum(d.freq[i] * η[t] for (t, i) in enumerate(rr) if d.isevent[i])
        ℓ += ev - logB_generic(η, d.freq[rr], d.mk[k])
    end
    return ℓ
end

# ---------------------------------------------------------------------------------
# Within-stratum inclusion probabilities π_i = P(a given copy of row i ∈ S), by a
# log-space forward–backward pass over *rows*. Forward and backward tables are stored
# only at row boundaries, so memory is O(rows · m) per stratum rather than
# O(Σ FREQ · m); the f_i copies of a row are folded in one at a time (O(f_i m) time,
# the same as the likelihood kernel). Used for `fitted` only, on demand.

# In-place: e[l] ← log e_l after appending `cnt` copies of value η (e holds log
# elementary symmetric sums of degrees 0..m; -Inf for unreachable degrees).
@inline function _append_copies!(e::AbstractVector{T}, η::T, cnt::Int, m::Int) where {T}
    @inbounds for _ in 1:cnt
        for l in m:-1:1
            e[l+1] = logaddexp_safe(e[l+1], e[l] + η)
        end
    end
    return e
end

function _stratum_inclusion!(π::Vector{T}, d::StratifiedDesign{T}, β::AbstractVector{T},
                             k::Int, buf::Vector{T}) where {T}
    rr = d.starts[k]:d.starts[k+1]-1
    flip = d.flip[k]
    s = flip ? -one(T) : one(T)
    m = d.meff[k]
    R = length(rr)
    # η (sign-adjusted) of each row, stored in π temporarily
    @inbounds for i in rr
        acc = d.offset[i]
        for r in eachindex(β)
            acc = muladd(d.Xt[r, i], β[r], acc)
        end
        π[i] = s * acc
    end
    if m == 1
        # softmax: P(copy of i ∈ S) = e^{sη_i} / Σ_j f_j e^{sη_j}
        emax = typemin(T); S = zero(T)
        @inbounds for i in rr
            emax = max(emax, π[i])
        end
        @inbounds for i in rr
            π[i] = exp(π[i] - emax)
            S += d.freq[i] * π[i]
        end
        @inbounds for i in rr
            pr = π[i] / S
            π[i] = flip ? 1 - pr : pr
        end
        return π
    end
    w = m + 1
    need = w * (R + 1) + w * (R + 2) + w
    length(buf) < need && resize!(buf, need)
    F = reshape(view(buf, 1:w*(R+1)), w, R + 1)           # F[:, r+1]: first r rows
    B = reshape(view(buf, w*(R+1)+1:w*(2R+3)), w, R + 2)   # B[:, r]: rows r..R
    G = view(buf, w*(2R+3)+1:w*(2R+4))
    fill!(F, T(-Inf)); fill!(B, T(-Inf))
    F[1, 1] = zero(T); B[1, R+1] = zero(T)
    @inbounds for (t, i) in enumerate(rr)
        copyto!(view(F, :, t + 1), view(F, :, t))
        _append_copies!(view(F, :, t + 1), π[i], d.freq[i], m)
    end
    @inbounds for t in R:-1:1
        i = rr[t]
        copyto!(view(B, :, t), view(B, :, t + 1))
        _append_copies!(view(B, :, t), π[i], d.freq[i], m)
    end
    logB = F[m+1, R+1]
    @inbounds for (t, i) in enumerate(rr)
        # rows other than i, plus f_i - 1 copies of i: [t^{m-1}] of F_{t-1} ⊗ (f_i - 1 copies) ⊗ B_{t+1}
        copyto!(G, view(F, :, t))
        _append_copies!(G, π[i], d.freq[i] - 1, m)
        acc = T(-Inf)
        for l in 0:m-1
            acc = logaddexp_safe(acc, G[l+1] + B[m-l, t+1])
        end
        pr = exp(π[i] + acc - logB)
        π[i] = flip ? 1 - pr : pr
    end
    return π
end

"""
    inclusion_probabilities(d, β; ntasks = 1) -> π

Per-copy inclusion probability of every design column (`Σ_i f_i π_i = m_k` within each
stratum). Strata are distributed over up to `ntasks` tasks; each stratum is written by
one task only, so the result does not depend on `ntasks`.
"""
function inclusion_probabilities(d::StratifiedDesign{T}, β::AbstractVector{T};
                                 ntasks::Integer = 1) where {T}
    N = size(d.Xt, 2)
    π = zeros(T, N)
    K = nstrata(d)
    nt = clamp(Int(ntasks), 1, max(length(d.chunks), 1))
    if nt == 1
        buf = T[]
        for k in 1:K
            _stratum_inclusion!(π, d, β, k, buf)
        end
    else
        counter = Atomic{Int}(1)
        @sync for _ in 1:nt
            Threads.@spawn _inclusion_worker!(π, d, β, counter)
        end
    end
    return π
end

# One task: its own scratch buffer; strata are written by exactly one task.
function _inclusion_worker!(π::Vector{T}, d::StratifiedDesign{T}, β::AbstractVector{T},
                            counter::Atomic{Int}) where {T}
    buf = T[]
    nc = length(d.chunks)
    while true
        t = atomic_add!(counter, 1)
        t > nc && break
        for k in d.chunks[d.chunkorder[t]]
            _stratum_inclusion!(π, d, β, k, buf)
        end
    end
    return nothing
end
