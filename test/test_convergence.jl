# Convergence semantics and separation.

@testset "SAS GCONV semantics on infert" begin
    df, _ = load_fixture("infert")
    m = clogit(@formula(case ~ spontaneous + induced), df; strata = :stratum)
    h = iterationhistory(m)
    crit = [r.criterion for r in h]
    # relative gradient criterion at iterations 0..4; the estimate is reported at the
    # first iterate with crit < 1e-8 (no extra Newton step).
    @test crit ≈ [0.534, 0.0316, 9.0e-4, 1.04e-6, 1.46e-12] rtol = 0.02
    @test iterations(m) == 4
    @test converged(m)
    @test m.criterion < 1e-8
    @test coef(m) == h[end].beta
    # ℓ0 = -Σ log C(n_k, m_k)
    @test nullloglikelihood(m) ≈ -(82 * log(3) + log(2)) rtol = 1e-14
    # MAXITER
    m1 = @test_logs (:warn, r"not attained in 1 iterations") clogit(
        @formula(case ~ spontaneous + induced), df; strata = :stratum, maxiter = 1)
    @test !converged(m1)
    @test convergencestatus(m1) === :maxiter
    @test iterations(m1) == 1
    @test_throws CL.ConvergenceError clogit(@formula(case ~ spontaneous + induced), df;
                                            strata = :stratum, maxiter = 1, strict = true)
    # FCONV / XCONV / ABSFCONV options
    mf = clogit(@formula(case ~ spontaneous + induced), df; strata = :stratum, gconv = 0.0,
                fconv = 1e-10)
    @test converged(mf) && occursin("FCONV", mf.convergence_message)
    mx = clogit(@formula(case ~ spontaneous + induced), df; strata = :stratum, gconv = 0.0,
                xconv = 1e-6)
    @test converged(mx) && occursin("XCONV", mx.convergence_message)
    # start values
    ms = clogit(@formula(case ~ spontaneous + induced), df; strata = :stratum,
                start = coef(m), gconv = 1e-16)
    @test coef(ms) ≈ coef(m) rtol = 1e-5
end

mkpairs(xcase, xctrl) = DataFrame(id = repeat(1:length(xcase); inner = 2),
                                y = repeat([1, 0]; outer = length(xcase)),
                                x = vec(permutedims(hcat(xcase, xctrl))))

@testset "complete separation" begin
    rng = StableRNG(4)
    xc = randn(rng, 20)
    d = mkpairs(xc .+ 0.5 .+ rand(rng, 20), xc)           # case always larger
    m = @test_logs (:warn, r"Complete separation") match_mode = :any clogit(
        @formula(y ~ x), d; strata = :id)
    @test !converged(m)
    @test convergencestatus(m) === :complete_separation
    @test all(isfinite, coef(m))
    @test_throws CL.ConvergenceError clogit(@formula(y ~ x), d; strata = :id, strict = true)
end

@testset "quasi-complete separation" begin
    # binary x: discordant pairs all favour the case; concordant pairs tie
    d = mkpairs([1.0, 1, 1, 1, 0, 1, 0, 1], [0.0, 0, 0, 0, 0, 1, 0, 1])
    d.z = randn(StableRNG(2), nrow(d))
    m = @test_logs (:warn, r"separation") match_mode = :any clogit(
        @formula(y ~ x + z), d; strata = :id)
    @test !converged(m)
    @test convergencestatus(m) in (:quasi_separation, :complete_separation)
    out = sprint(show, MIME"text/plain"(), m)
    @test occursin("separation", out)
    @test test_report_layout(out) == 6
    @test !any(t -> occursin("separation", t.title) ||
                    any(l -> occursin("separation", l), t.body), report_tables(out))
end

@testset "no false separation alarms" begin
    for name in ("la", "infert", "synth_mk", "synth_big")
        df, refs = load_fixture(name)
        f = name == "la" ? @formula(y ~ gall + hyper) :
            name == "infert" ? @formula(case ~ spontaneous + induced) :
            name == "synth_mk" ? @formula(y ~ x1 + x2 + x3) : @formula(y ~ x1 + x2)
        s = name == "la" ? :s : name == "infert" ? :stratum : :id
        m = @test_logs clogit(f, df; strata = s)
        @test convergencestatus(m) === :converged
    end
end

@testset "no false separation from an outlying covariate value" begin
    # statistics reviewer: 60 pairs, one case exposure of 1e7. The MLE is finite (the
    # profile likelihood peaks at β ≈ 0.5559) and must not be flagged.
    rng = StableRNG(1)
    K = 60
    y = repeat([1, 0], K)
    x = randn(rng, 2K) .+ 0.5 .* y
    x[1] = 1e7
    d = DataFrame(id = repeat(1:K, inner = 2), y = y, x = x)
    m = @test_logs clogit(@formula(y ~ x), d; strata = :id)
    @test convergencestatus(m) === :converged && converged(m)
    @test coef(clogit(@formula(y ~ x), d; strata = :id, strict = true)) == coef(m)
    # swe reviewer: 40 pairs with case 0.4 above control, 20 with 0.4 below, one 1e6 value
    base = randn(StableRNG(3), K)
    xc = [k <= 40 ? base[k] + 0.4 : base[k] - 0.4 for k in 1:K]
    xc[1] = 1e6
    d2 = mkpairs(xc, base)
    m2 = @test_logs clogit(@formula(y ~ x), d2; strata = :id)
    @test convergencestatus(m2) === :converged
    @test 0 < coef(m2)[1] < 5 && isfinite(stderror(m2)[1])
end

@testset "heavy-tailed exposure: no false separation flags" begin
    rng = StableRNG(11)
    nexist = 0; nflag = 0
    for σ in (5.0, 6.0), rep in 1:40
        K = 50
        X = zeros(2K, 1)
        for k in 1:K
            a, c = exp(σ * randn(rng)), exp(σ * randn(rng))
            pr = 1 / (1 + exp(-(log(a) - log(c))))
            X[2k-1, 1], X[2k, 1] = rand(rng) < pr ? (a, c) : (c, a)
        end
        exists = any(X[2k-1, 1] < X[2k, 1] for k in 1:K)   # p = 1: some pair discordant
        m = quietly(() -> clogit(X, repeat([1, 0], K), repeat(1:K, inner = 2)))
        if exists
            nexist += 1
            nflag += convergencestatus(m) in (:quasi_separation, :complete_separation)
        else
            @test convergencestatus(m) in (:quasi_separation, :complete_separation)
        end
    end
    @test nexist > 70
    @test nflag == 0
end

@testset "start values" begin
    df, _ = load_fixture("infert")
    f = @formula(case ~ spontaneous + induced)
    m = clogit(f, df; strata = :stratum)
    @test_throws ArgumentError clogit(f, df; strata = :stratum, start = [NaN, 0.0])
    @test_throws ArgumentError clogit(f, df; strata = :stratum, start = [Inf, 0.0])
    @test_throws DimensionMismatch clogit(f, df; strata = :stratum, start = [0.0])
    # an absurd start must not "converge" at iteration 0 (GCONV is relative to |ℓ|)
    for st in ([1e200, 0.0], [30.0, 0.0])
        ms = @test_logs (:warn, r"start values") clogit(f, df; strata = :stratum, start = st)
        @test converged(ms)
        @test coef(ms) == coef(m)
    end
    # a sensible start is used as given
    ms = @test_logs clogit(f, df; strata = :stratum, start = [1.0, 1.0])
    @test converged(ms)
    @test iterationhistory(ms)[1].beta == [1.0, 1.0]
    @test coef(ms) ≈ coef(m) rtol = 1e-3
end

@testset "step halving" begin
    # From start (2.5, 0) on infert (ℓ(start) > ℓ(0), so it is used) the first full Newton
    # step overshoots: one halving is needed. Without halving the fit stalls.
    df, _ = load_fixture("infert")
    f = @formula(case ~ spontaneous + induced)
    m = clogit(f, df; strata = :stratum)
    md = @test_logs clogit(f, df; strata = :stratum, start = [2.5, 0.0])
    h = iterationhistory(md)
    @test h[1].beta == [2.5, 0.0]
    @test h[2].stephalvings == 1
    @test converged(md)
    @test coef(md) ≈ coef(m) rtol = 1e-3
    m0 = quietly(() -> clogit(f, df; strata = :stratum, start = [2.5, 0.0], maxstephalving = 0))
    @test convergencestatus(m0) === :stalled
end
