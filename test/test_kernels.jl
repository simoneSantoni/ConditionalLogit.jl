# Likelihood kernels vs brute-force enumeration, ForwardDiff and BigFloat.

@testset "brute force + ForwardDiff on 200 random problems" begin
    rng = StableRNG(2024)
    ntested = 0
    for rep in 1:200
        X, y, codes, f, off = random_problem(rng)
        d, s = design_of(X, y, codes, f, off)
        CL.nstrata(d) == 0 && continue
        ntested += 1
        p = size(X, 2)
        β = 0.7 .* randn(rng, p)
        ℓ, g, I = eval_design(d, β)
        ℓb, gb, Hb = brute(X, y, codes, f, off, β)
        @test ℓ ≈ ℓb rtol = 1e-11 atol = 1e-11
        @test g ≈ gb rtol = 1e-9 atol = 1e-10
        @test I ≈ -Hb rtol = 1e-9 atol = 1e-10
        # generic path (no complement, no softmax fast path) + automatic differentiation
        lg(b) = CL.loglik_generic(d, b)
        @test lg(β) ≈ ℓ rtol = 1e-12 atol = 1e-12
        @test ForwardDiff.gradient(lg, β) ≈ g rtol = 1e-9 atol = 1e-10
        @test -ForwardDiff.hessian(lg, β) ≈ I rtol = 1e-9 atol = 1e-10
        # information is symmetric PSD
        @test I == I'
        @test minimum(eigvals(Symmetric(I))) > -1e-10
        # inclusion probabilities sum to m_k within each stratum
        π = CL.inclusion_probabilities(d, β)
        for k in 1:CL.nstrata(d)
            r = d.starts[k]:d.starts[k+1]-1
            @test sum(d.freq[r] .* π[r]) ≈ d.mk[k] rtol = 1e-10
        end
    end
    @test ntested > 150
end

@testset "AD at β = 0 (all η tied)" begin
    rng = StableRNG(5)
    X, y, codes, f, off = random_problem(rng; K = 5, nmax = 7, p = 3, withoffset = false)
    d, _ = design_of(X, y, codes, f, off)
    β = zeros(3)
    ℓ, g, I = eval_design(d, β)
    lg(b) = CL.loglik_generic(d, b)
    @test -ForwardDiff.hessian(lg, β) ≈ I rtol = 1e-10
    @test ForwardDiff.gradient(lg, β) ≈ g rtol = 1e-10 atol = 1e-12
    # ℓ(0) = -Σ log C(n_k, m_k)
    @test ℓ ≈ -sum(logbinom(d.nk[k], d.mk[k]) for k in 1:CL.nstrata(d)) rtol = 1e-13
end

@testset "complement and softmax paths" begin
    rng = StableRNG(11)
    # one stratum, n = 9, every m: m = 1 / n-1 use softmax, m > n/2 uses the complement
    X = randn(rng, 9, 2)
    for m in 1:8
        y = BitVector([trues(m); falses(9 - m)])
        d, _ = design_of(X, y, ones(Int, 9), ones(Int, 9), zeros(9))
        @test d.flip[1] == (m > 9 - m)
        β = [0.4, -1.3]
        ℓ, g, I = eval_design(d, β)
        ℓb, gb, Hb = brute(X, y, ones(Int, 9), ones(Int, 9), zeros(9), β)
        @test ℓ ≈ ℓb rtol = 1e-12
        @test g ≈ gb rtol = 1e-10 atol = 1e-12
        @test I ≈ -Hb rtol = 1e-10
    end
end

@testset "large balanced strata do not overflow" begin
    for (n, m) in ((1100, 550), (3000, 1500))
        rng = StableRNG(n)
        X = randn(rng, n, 1)
        y = BitVector([trues(m); falses(n - m)])
        d, _ = design_of(X, y, ones(Int, n), ones(Int, n), zeros(n))
        ℓ, g, I = eval_design(d, [0.0])
        @test isfinite(ℓ) && all(isfinite, g) && all(isfinite, I)
        @test ℓ ≈ -logbinom(n, m) rtol = 1e-12
        # large β: η range of about ±1000
        ℓ2, g2, I2 = eval_design(d, [300.0])
        @test isfinite(ℓ2) && all(isfinite, g2) && all(isfinite, I2)
        @test I2[1, 1] >= 0
    end
end

@testset "BigFloat oracle with extreme η" begin
    rng = StableRNG(17)
    n, m = 40, 17
    X = 100 .* randn(rng, n, 2) .+ 1e4         # huge stratum-level mean, η range ~±1000
    y = BitVector(shuffle(rng, [trues(m); falses(n - m)]))
    d, _ = design_of(X, y, ones(Int, n), ones(Int, n), zeros(n))
    β = [3.0, -2.0]
    ℓ, g, I = eval_design(d, β)
    setprecision(BigFloat, 512) do
        lg(b) = CL.loglik_generic(d, b)
        @test ℓ ≈ Float64(lg(big.(β))) rtol = 1e-12
    end
    lg(b) = CL.loglik_generic(d, b)
    @test ForwardDiff.gradient(lg, β) ≈ g rtol = 1e-8
    @test -ForwardDiff.hessian(lg, β) ≈ I rtol = 1e-7
end

@testset "kernel does not allocate" begin
    rng = StableRNG(3)
    X, y, codes, f, off = random_problem(rng; K = 30, nmax = 9, p = 3)
    d, _ = design_of(X, y, codes, f, off)
    ev = CL.Evaluator(d, 1)
    β = [0.1, 0.2, -0.3]
    allocs(ev, β) = (CL.evaluate!(ev, β, Val(2)); @allocated CL.evaluate!(ev, β, Val(2)))
    @test allocs(ev, β) == 0
end

@testset "inclusion probabilities: FREQ rows equal expanded copies" begin
    rng = StableRNG(31)
    for rep in 1:30
        X, y, codes, f, off = random_problem(rng; nmax = 7, withfreq = true)
        f .= rand(rng, 1:4, length(f))
        d, _ = design_of(X, y, codes, f, off)
        CL.nstrata(d) == 0 && continue
        β = 0.6 .* randn(rng, size(X, 2))
        π = CL.inclusion_probabilities(d, β)
        # expanded data: every copy is its own row
        idx = reduce(vcat, [fill(i, f[i]) for i in eachindex(f)])
        de, _ = design_of(X[idx, :], y[idx], codes[idx], ones(Int, length(idx)), off[idx])
        πe = CL.inclusion_probabilities(de, β)
        # map design rows back to data rows and compare per-copy probabilities
        pr = Dict(d.rows[j] => π[j] for j in eachindex(π))
        for (j, r) in enumerate(de.rows)
            @test πe[j] ≈ pr[idx[r]] rtol = 1e-10 atol = 1e-14
        end
        @test CL.inclusion_probabilities(d, β; ntasks = 3) == π
    end
end
