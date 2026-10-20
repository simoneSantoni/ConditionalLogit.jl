# Results are bitwise identical for any number of tasks / threads.

df, _ = load_fixture("synth_mk")
f = @formula(y ~ x1 + x2 + x3 + g)

@testset "evaluate! is bitwise independent of ntasks" begin
    rng = StableRNG(99)
    X, y, codes, fr, off = random_problem(rng; K = 3000, nmax = 12, p = 4)
    d, _ = design_of(X, y, codes, fr, off)
    @test length(d.chunks) > 8
    β = randn(rng, 4) ./ 4
    ref = eval_design(d, β; ntasks = 1)
    for nt in (2, 3, 7, 32)
        @test eval_design(d, β; ntasks = nt) == ref     # exact equality
    end
end

@testset "fit is bitwise independent of ntasks" begin
    m1 = clogit(f, df; strata = :id, ntasks = 1)
    for nt in (2, 5, 16)
        mt = clogit(f, df; strata = :id, ntasks = nt)
        @test coef(mt) == coef(m1)
        @test vcov(mt) == vcov(m1)
        @test loglikelihood(mt) == loglikelihood(m1)
    end
end

@testset "subprocess with julia -t 4 gives identical bits" begin
    code = """
    using ConditionalLogit, CSV, DataFrames
    df = CSV.read($(repr(joinpath(REFDIR, "synth_mk.csv"))), DataFrame; missingstring = "NA")
    m = clogit(@formula(y ~ x1 + x2 + x3 + g), df; strata = :id)
    print(Threads.nthreads(), " ", join(reinterpret(UInt64, coef(m)), ","), " ",
          reinterpret(UInt64, loglikelihood(m)))
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no -t 4 -e $code`
    env = copy(ENV)
    env["JULIA_LOAD_PATH"] = join(LOAD_PATH, Sys.iswindows() ? ";" : ":")
    env["JULIA_PROJECT"] = Base.active_project()
    # Skip only if Julia itself cannot be launched; any failure of the threaded run
    # (load error, crash) must fail the test.
    canrun = try
        success(`$(Base.julia_cmd()) --startup-file=no -e 0`)
    catch
        false
    end
    if canrun
        out = read(setenv(cmd, env), String)       # throws (test error) on failure
        m = clogit(f, df; strata = :id)
        nt, bits, lbits = split(out)
        @test nt == "4"
        @test bits == join(reinterpret(UInt64, coef(m)), ",")
        @test lbits == string(reinterpret(UInt64, loglikelihood(m)))
    else
        @test_skip "cannot launch a julia subprocess"
    end
end

@testset "concurrent fits (inside @threads and @spawn) match the serial fit" begin
    ref = clogit(f, df; strata = :id, ntasks = 1)
    X = Matrix(df[:, [:x1, :x2, :x3]])
    refx = clogit(X, df.y, df.id; ntasks = 1)
    nfit = 16
    res = Vector{Any}(undef, nfit)
    Threads.@threads for i in 1:nfit
        res[i] = clogit(f, df; strata = :id, ntasks = isodd(i) ? 1 : 3)
    end
    @test all(m -> coef(m) == coef(ref) && vcov(m) == vcov(ref), res)
    tasks = [Threads.@spawn clogit(X, df.y, df.id; ntasks = 2) for _ in 1:nfit]
    @test all(t -> coef(fetch(t)) == coef(refx), tasks)
    # the default ntasks inside a threaded loop
    Threads.@threads for i in 1:4
        res[i] = clogit(X, df.y, df.id)
    end
    @test all(i -> coef(res[i]) == coef(refx), 1:4)
end

@testset "heavy stratum: column-split information is bitwise identical" begin
    rng = StableRNG(21)
    p = 12
    N = 1500 + 400
    X = randn(rng, N, p)
    codes = vcat(fill(1, 1500), repeat(2:101, inner = 4))
    y = falses(N); y[1:600] .= true
    for k in 0:99
        y[1501+4k] = true; y[1502+4k] = true
    end
    d, _ = design_of(X, y, codes, ones(Int, N), zeros(N))
    β = randn(rng, p) ./ 10
    ref = eval_design(d, β; ntasks = 1)
    for nt in (2, 8)
        @test eval_design(d, β; ntasks = nt) == ref
    end
    # forced split (capacity 0, several blocks) equals the plain recursion
    ws1 = CL.Workspace{Float64}(p, d.maxrows, d.maxmeff, d.maxmeff, 1)
    ws2 = CL.Workspace{Float64}(p, d.maxrows, d.maxmeff, 0, 3)
    l1 = CL.stratum!(ws1, d, 1, β, Val(2))
    l2 = CL.stratum!(ws2, d, 1, β, Val(2))
    @test l1 == l2 && ws1.H == ws2.H && ws1.g == ws2.g
    @test CL._col_blocks(12, 3) == [1:7, 8:10, 11:12]
    @test reduce(vcat, collect.(CL._col_blocks(17, 5))) == 1:17
end

@testset "per-task workspaces are not sized for the largest stratum" begin
    p = 40
    rng = StableRNG(8)
    N = 2000 + 800
    X = randn(rng, N, p)
    codes = vcat(repeat(1:400, inner = 2), fill(401, 2000))
    y = falses(N); y[1:2:800] .= true; y[801:1800] .= true
    d, _ = design_of(X, y, codes, ones(Int, N), zeros(N))
    @test d.maxmeff == 1000
    ev = CL.Evaluator(d, 8)
    @test all(ws -> size(ws.C, 3) <= 1, ev.ws)            # only m_k = 1 strata besides
    @test Base.summarysize(ev) < 16 * 2^20
    ℓ1 = eval_design(d, zeros(p); ntasks = 1)
    @test eval_design(d, zeros(p); ntasks = 8) == ℓ1
    @test ℓ1[1] ≈ -(logbinom(2000, 1000) + 400 * log(2)) rtol = 1e-12
end
