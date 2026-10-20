# Newton–Raphson with SAS PROC LOGISTIC (STRATA) convergence semantics.
#
# * Start at β = 0 (or `start`).
# * At each iterate β_i compute ℓ, g, I; δ = I⁻¹g; crit = g'δ / (|ℓ| + 1e-6) (GCONV).
#   If crit < gconv, stop and report β_i itself. The extra step is not taken, as in
#   SAS, so the reported estimate is the iterate at which the criterion was met.
# * FCONV/ABSFCONV/XCONV are optional. As SAS does for STRATA, an exactly zero change in
#   ℓ or β also counts as convergence.
# * Step control: the full step is evaluated with score and information at once (it is
#   almost always accepted). If it lowers ℓ, halve it, using value-only evaluations
#   (cheap), until ℓ does not decrease, the step is negligible relative to β, or
#   `maxstephalving` halvings were made. If I is not positive definite, ridge it:
#   I + λ·diag(I), with λ growing ×10. SAS's NRRIDG technique instead increases a ridge
#   when a step fails, so on hard problems (near-separation, poor starts) the iterates,
#   and hence the last digits of the reported GCONV iterate, can differ from SAS. When
#   the full step is accepted (the usual case) the two coincide.
# * A user `start` with ℓ(start) < ℓ(0) is replaced by 0 (with a warning): GCONV is
#   relative to |ℓ|, so an absurd start could otherwise "converge" immediately.
# * Stop after `maxiter` iterations without convergence.

struct NewtonResult{T}
    β::Vector{T}
    loglik::T
    g::Vector{T}
    I::Matrix{T}
    loglik0::T
    score0::T
    iterations::Int
    converged::Bool
    status::Symbol
    crit::T
    message::String
    history::Vector{IterationRecord}
    laststep::Vector{T}
end

# Solve I δ = g; ridge I if it is not positive definite.
function newton_direction(I::Matrix{T}, g::Vector{T}) where {T}
    F = cholesky(Symmetric(I), check = false)
    issuccess(F) && return F \ g, false
    D = [max(I[j, j], sqrt(eps(T))) for j in axes(I, 1)]
    λ = T(1e-4)
    for _ in 1:30
        F = cholesky(Symmetric(I + λ * Diagonal(D)), check = false)
        issuccess(F) && return F \ g, true
        λ *= 10
    end
    return zeros(T, length(g)), true
end

"""
    quadform_inv(I, g)

`g' I⁻¹ g`, using a Cholesky factorization with a pseudo-inverse fallback.
"""
function quadform_inv(I::Matrix{T}, g::Vector{T}) where {T}
    isempty(g) && return zero(T)
    F = cholesky(Symmetric(I), check = false)
    issuccess(F) && return dot(g, F \ g)
    return dot(g, pinv(Symmetric(I)) * g)
end

function fmt_crit(x)
    s = @sprintf("%.0E", x)
    return replace(s, r"E([+-])0*(\d)" => s"E\1\2", "E+" => "E")
end

function newton(ev::Evaluator{T}, opts::FitOptions, start::Vector{T}) where {T}
    p = npar(ev.d)
    β0 = zeros(T, p)
    ℓ0 = evaluate!(ev, β0, Val(2))
    score0 = quadform_inv(ev.I, ev.g)
    g = copy(ev.g); I = copy(ev.I)
    β = copy(start)
    ℓ = ℓ0
    if !iszero(β)
        ℓs = evaluate!(ev, β, Val(2))
        if ℓs >= ℓ0
            ℓ = ℓs
            copyto!(g, ev.g); copyto!(I, ev.I)
        else
            @warn "The log likelihood at the supplied start values ($(@sprintf("%.6g", ℓs))) " *
                  "is below its value at β = 0 ($(@sprintf("%.6g", ℓ0))); starting from β = 0 instead."
            fill!(β, zero(T))
        end
    end
    history = IterationRecord[]
    iter = 0
    converged = false
    status = :maxiter
    message = ""
    crit = T(NaN)
    ℓprev = T(NaN)
    βprev = fill(T(NaN), p)
    laststep = zeros(T, p)
    halvings = 0
    ridged = false
    while true
        δ, ridged = newton_direction(I, g)
        crit = p == 0 ? zero(T) : dot(g, δ) / (abs(ℓ) + T(1e-6))
        push!(history, IterationRecord(iter, ℓ, crit, halvings, ridged, Vector{Float64}(β)))
        opts.verbose && @printf("Iteration %3d  -2 Log L = %.6f  crit = %.3e  halvings = %d%s\n",
                                iter, -2ℓ, crit, halvings, ridged ? "  (ridged)" : "")
        if crit < opts.gconv
            converged = true; status = :converged
            message = "Convergence criterion (GCONV=$(fmt_crit(opts.gconv))) satisfied."
            break
        end
        if iter >= 1
            Δℓ = abs(ℓ - ℓprev)
            relx = zero(T)
            for j in 1:p
                δj = β[j] - βprev[j]
                relx = max(relx, abs(βprev[j]) < 0.01 ? abs(δj) : abs(δj / βprev[j]))
            end
            if opts.fconv > 0 && Δℓ / (abs(ℓprev) + 1e-6) < opts.fconv
                converged = true; status = :converged
                message = "Convergence criterion (FCONV=$(fmt_crit(opts.fconv))) satisfied."
                break
            elseif opts.absfconv > 0 && Δℓ < opts.absfconv
                converged = true; status = :converged
                message = "Convergence criterion (ABSFCONV=$(fmt_crit(opts.absfconv))) satisfied."
                break
            elseif opts.xconv > 0 && relx < opts.xconv
                converged = true; status = :converged
                message = "Convergence criterion (XCONV=$(fmt_crit(opts.xconv))) satisfied."
                break
            elseif Δℓ == 0 || relx == 0
                converged = true; status = :converged
                message = Δℓ == 0 ? "Convergence criterion (FCONV=0) satisfied." :
                                    "Convergence criterion (XCONV=0) satisfied."
                break
            end
        end
        if iter >= opts.maxiter
            status = :maxiter
            message = "Convergence was not attained in $(opts.maxiter) iterations."
            break
        end
        # full step with score and information; halve with value-only evaluations if needed
        βt = β .+ δ
        ℓt = evaluate!(ev, βt, Val(2))
        halvings = 0
        tol = T(1e-12) * (1 + abs(ℓ))
        if !(ℓt >= ℓ - tol)
            βscale = 1 + maximum(abs, β; init = zero(T))
            while !(ℓt >= ℓ - tol) && halvings < opts.maxstephalving &&
                  maximum(abs, δ; init = zero(T)) > eps(T) * βscale
                δ ./= 2
                βt .= β .+ δ
                ℓt = evaluate!(ev, βt, Val(0))
                halvings += 1
            end
            if !(ℓt >= ℓ - tol)
                status = :stalled
                message = "The log likelihood could not be increased along the Newton direction " *
                          "(relative gradient criterion $(@sprintf("%.3g", crit)))."
                break
            end
            ℓt = evaluate!(ev, βt, Val(2))
        end
        copyto!(laststep, δ)
        copyto!(βprev, β); ℓprev = ℓ
        copyto!(β, βt)
        ℓ = ℓt
        copyto!(g, ev.g); copyto!(I, ev.I)
        iter += 1
    end
    return NewtonResult{T}(β, ℓ, g, I, ℓ0, score0, iter, converged, status, crit, message,
                           history, laststep)
end

"""
    separation_check(d, dir) -> :none | :quasi | :complete

Check whether `dir` certifies that the conditional MLE does not exist. This is the
conditional analogue of Albert & Anderson (1984). If, in every informative stratum,
every event has `x'dir` at least as large as every non-event, then ℓ(β + t·dir) is
non-decreasing in t. If at least one stratum separates strictly, ℓ keeps increasing as
t → ∞. `:complete` means all strata separate strictly; `:quasi` means some are ties.

A stratum counts as a tie only when its gap is within rounding error of zero *relative
to that stratum's own values*; a single outlying
covariate value elsewhere cannot turn a clearly negative gap into a tie.
"""
function separation_check(d::StratifiedDesign{T}, dir::AbstractVector) where {T}
    p = npar(d)
    (p == 0 || !all(isfinite, dir) || iszero(dir)) && return :none
    return _separation_dir(d, convert(Vector{T}, dir))
end

# Tie tolerance of one stratum: a small multiple of the rounding error of the values
# compared (x̃_i'dir is a p-term dot product). Never relative to the global maximum.
@inline _tie_tol(::Type{T}, p::Int, scale) where {T} = T(max(64, 4 * (p + 2))) * eps(T) * scale

# One fused pass: v_i = x̃_i'dir and its rounding scale Σ_r |x̃_ri dir_r| are computed per
# row; the pass stops at the first stratum whose gap is clearly negative.
function _separation_dir(d::StratifiedDesign{T}, dir::Vector{T}) where {T}
    p = npar(d)
    Xt = d.Xt
    nstrict = 0; nties = 0
    @inbounds for k in 1:nstrata(d)
        minev = T(Inf); maxne = T(-Inf); smax = zero(T)
        for i in d.starts[k]:d.starts[k+1]-1
            v = zero(T); sc = zero(T)
            @simd for r in 1:p
                t = Xt[r, i] * dir[r]
                v += t; sc += abs(t)
            end
            if d.isevent[i]
                minev = min(minev, v)
            else
                maxne = max(maxne, v)
            end
            smax = max(smax, sc)
        end
        gap = minev - maxne
        tol = _tie_tol(T, p, smax)
        if gap < -tol
            return :none
        elseif gap > tol
            nstrict += 1
        else
            nties += 1
        end
    end
    nstrict == 0 && return :none
    return nties == 0 ? :complete : :quasi
end

# Numerical confirmation of a certificate: ℓ must not decrease along `dir` from β̂
# (it is non-decreasing for a valid certificate; for a finite MLE every direction
# lowers ℓ). Steps are scaled so that the largest |Δη| is 1, 10 and 100.
function _confirm_direction(ev::Evaluator{T}, β::Vector{T}, ℓ::T, dir::AbstractVector) where {T}
    vmax = maximum(abs, ev.d.Xt' * dir; init = zero(T))
    (vmax > 0 && isfinite(vmax) && isfinite(ℓ)) || return false
    tol = T(1e-9) * (1 + abs(ℓ))
    for t in (1, 10, 100)
        ℓt = evaluate!(ev, β .+ (T(t) / vmax) .* dir, Val(0))
        ℓt >= ℓ - tol || return false
    end
    return true
end

"""
    diagnose_separation(d, res, ev) -> :none | :quasi | :complete

Try candidate directions as separation certificates (see [`separation_check`](@ref)):
the estimate β̂, the last Newton step, the eigenvectors of the information matrix with
relatively tiny eigenvalues (both signs), and the coordinate axes (both signs). Each
passing certificate is confirmed numerically by evaluating ℓ along it. A confirmed
certificate shows that the MLE does not exist. A failing search does not prove that it
exists: an exact check would need a linear program.
"""
function diagnose_separation(d::StratifiedDesign{T}, res::NewtonResult{T},
                             ev::Evaluator{T}) where {T}
    p = npar(d)
    cands = Vector{Vector{T}}()
    push!(cands, res.β)
    res.iterations > 0 && push!(cands, res.laststep)
    if all(isfinite, res.I)
        E = eigen(Symmetric(res.I))
        λmax = maximum(abs, E.values)
        for (i, λ) in enumerate(E.values)
            if λ <= 1e-6 * λmax
                v = E.vectors[:, i]
                push!(cands, v, -v)
            end
        end
    end
    found = :none
    for c in cands
        r = separation_check(d, c)
        r === :none && continue
        _confirm_direction(ev, res.β, res.loglik, c) || continue
        r === :complete && return :complete
        found = :quasi
    end
    for (j, sgn, r) in _coordinate_separation(d)
        e = zeros(T, p); e[j] = sgn
        _confirm_direction(ev, res.β, res.loglik, e) || continue
        r === :complete && return :complete
        found = :quasi
    end
    return found
end

# Certificates along the coordinate axes ±e_j for all j, in one column-major pass over
# the data, with per-stratum, per-column tie tolerances. Returns the passing
# certificates as (j, ±1, :complete | :quasi).
function _coordinate_separation(d::StratifiedDesign{T}) where {T}
    p = npar(d)
    Xt = d.Xt
    alive = trues(2, p)          # row 1: +e_j, row 2: -e_j
    nstrict = zeros(Int, 2, p); nties = zeros(Int, 2, p)
    minev = zeros(T, p); maxev = zeros(T, p); minne = zeros(T, p); maxne = zeros(T, p)
    amax = zeros(T, p)
    out = Tuple{Int,Int,Symbol}[]
    @inbounds for k in 1:nstrata(d)
        fill!(minev, T(Inf)); fill!(minne, T(Inf)); fill!(maxev, T(-Inf)); fill!(maxne, T(-Inf))
        fill!(amax, zero(T))
        for i in d.starts[k]:d.starts[k+1]-1
            if d.isevent[i]
                for j in 1:p
                    x = Xt[j, i]
                    minev[j] = min(minev[j], x); maxev[j] = max(maxev[j], x)
                    amax[j] = max(amax[j], abs(x))
                end
            else
                for j in 1:p
                    x = Xt[j, i]
                    minne[j] = min(minne[j], x); maxne[j] = max(maxne[j], x)
                    amax[j] = max(amax[j], abs(x))
                end
            end
        end
        for j in 1:p
            tol = _tie_tol(T, 1, amax[j])
            for (s, gap) in ((1, minev[j] - maxne[j]), (2, minne[j] - maxev[j]))
                alive[s, j] || continue
                if gap < -tol
                    alive[s, j] = false
                elseif gap > tol
                    nstrict[s, j] += 1
                else
                    nties[s, j] += 1
                end
            end
        end
        any(alive) || return out
    end
    for j in 1:p, s in 1:2
        (alive[s, j] && nstrict[s, j] > 0) || continue
        push!(out, (j, s == 1 ? 1 : -1, nties[s, j] == 0 ? :complete : :quasi))
    end
    return out
end
