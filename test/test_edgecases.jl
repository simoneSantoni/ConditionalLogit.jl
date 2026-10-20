# Front end: tables, response coding, FREQ, missing values, errors.

df, _ = load_fixture("synth_mk")
f = @formula(y ~ x1 + x2 + x3)
m = clogit(f, df; strata = :id)

@testset "formula vs matrix interface, table types" begin
    X = Matrix(df[:, [:x1, :x2, :x3]])
    mx = clogit(X, df.y, df.id; coefnames = ["x1", "x2", "x3"])
    @test coef(mx) == coef(m)
    @test vcov(mx) == vcov(m)
    mf = fit(ConditionalLogitModel, f, df; strata = :id)
    @test coef(mf) == coef(m)
    nt = (y = df.y, x1 = df.x1, x2 = df.x2, x3 = df.x3, id = df.id)
    @test coef(clogit(f, nt; strata = :id)) == coef(m)
    csvf = CSV.File(joinpath(REFDIR, "synth_mk.csv"); missingstring = "NA")
    @test coef(clogit(f, csvf; strata = :id)) == coef(m)
    @test coefnames(mx) == ["x1", "x2", "x3"]
    @test coefnames(clogit(X, df.y, df.id)) == ["x1", "x2", "x3"]
    # Float32 input is promoted
    @test coef(clogit(Float32.(X), df.y, df.id)) ≈ coef(m) rtol = 1e-5
end

@testset "intercept policy" begin
    @test_throws ArgumentError clogit(@formula(y ~ 1 + x1), df; strata = :id)
    @test coef(clogit(@formula(y ~ 0 + x1 + x2 + x3), df; strata = :id)) == coef(m)
    err = try
        clogit(@formula(y ~ 1 + x1), df; strata = :id)
    catch e
        e
    end
    @test occursin("no intercept", err.msg)
end

@testset "response coding" begin
    d2 = copy(df)
    d2.yb = d2.y .== 1
    d2.yf = Float64.(d2.y)
    d2.ys = ifelse.(d2.y .== 1, "case", "control")
    d2.yc = categorical(d2.ys)
    for (col, ev) in ((:yb, nothing), (:yf, nothing), (:ys, "case"), (:yc, "case"))
        mm = clogit(Term(col) ~ Term(:x1) + Term(:x2) + Term(:x3), d2; strata = :id, event = ev)
        @test coef(mm) ≈ coef(m) rtol = 1e-13
    end
    mm = clogit(@formula(ys ~ x1 + x2 + x3), d2; strata = :id, event = "control")
    @test coef(mm) ≈ -coef(m) rtol = 1e-6
    @test occursin("control", mm.eventlabel)
    @test_throws ArgumentError clogit(@formula(ys ~ x1), d2; strata = :id)   # event required
    @test_throws ArgumentError clogit(@formula(ys ~ x1), d2; strata = :id, event = "Case")
    d2.y3 = d2.y .+ (1:nrow(d2) .== 5)
    @test_throws ArgumentError clogit(@formula(y3 ~ x1), d2; strata = :id)
    d2.y4 = [i == 3 ? "maybe" : s for (i, s) in enumerate(d2.ys)]
    @test_throws ArgumentError clogit(@formula(y4 ~ x1), d2; strata = :id, event = "case")
end

@testset "FREQ semantics" begin
    rng = StableRNG(8)
    d2 = copy(df)
    d2.f = rand(rng, 1:3, nrow(d2))
    mf = clogit(f, d2; strata = :id, freq = :f)
    ex = d2[reduce(vcat, [fill(i, d2.f[i]) for i in 1:nrow(d2)]), :]
    me = clogit(f, ex; strata = :id)
    @test coef(mf) ≈ coef(me) rtol = 1e-10
    @test mf.freq_used == sum(d2.f)
    @test mf.nobs_used == nrow(d2)
    # truncation and dropping
    d2.ff = Float64.(d2.f) .+ 0.7
    d2.ff[1] = 0.4                                  # truncated to 0 -> dropped
    mt = @test_logs (:warn, r"truncated") clogit(f, d2; strata = :id, freq = :ff)
    d3 = copy(d2); d3.f[1] = 0
    mz = clogit(f, d3; strata = :id, freq = :f)
    @test coef(mt) ≈ coef(mz) rtol = 1e-13
    @test mz.nobs_used == nrow(d2) - 1
    @test_throws ArgumentError clogit(f, d2; strata = :id, weights = pweights(d2.f))
    @test_throws ArgumentError clogit(f, d2; strata = :id, freq = :f, weights = fweights(d2.f))
end

@testset "missing strata and strata_missing" begin
    d2 = allowmissing(copy(df))
    d2.id[1:3] .= missing
    mm = clogit(f, d2; strata = :id)
    @test mm.nobs_used == nrow(df) - 3
    mm2 = clogit(f, d2; strata = :id, strata_missing = true)
    @test mm2.nobs_used == nrow(df)
    @test stratasummary(mm2).nstrata == stratasummary(mm).nstrata + 1
end

@testset "errors" begin
    @test_throws ArgumentError clogit(f, df; strata = :nope)
    @test_throws ArgumentError clogit(@formula(y ~ nope), df; strata = :id)
    @test_throws UndefKeywordError clogit(f, df)
    @test_throws ArgumentError clogit(f, df; strata = :id, method = :efron)
    @test_throws DimensionMismatch clogit(rand(5, 2), [1, 0, 1, 0], [1, 1, 2, 2, 3])
    # no informative strata
    @test_throws ArgumentError clogit(rand(4, 1), [1, 1, 0, 0], [1, 1, 2, 2])
    # non-finite covariate
    X = Matrix(df[:, [:x1, :x2]]); X[5, 1] = Inf
    @test_throws ArgumentError clogit(X, df.y, df.id)
    # aliased with dropcollinear = false
    X = hcat(df.x1, df.x2, df.x1 .+ df.x2)
    @test_throws ArgumentError clogit(X, df.y, df.id; dropcollinear = false)
    @test_throws ArgumentError clogit(f, df; strata = :id, sc_n = :strata)
end

@testset "aliasing: linear combinations and stratum-constant covariates" begin
    d2 = copy(df)
    d2.x4 = d2.x1 .- 2 .* d2.x2                 # linear combination
    d2.sc = [parse(Int, s[2:end]) for s in d2.id]   # constant within strata
    mm = @test_logs (:warn, r"x4, sc") clogit(@formula(y ~ x1 + x2 + x3 + x4 + sc), d2; strata = :id)
    @test aliased(mm) == [false, false, false, true, true]
    @test coef(mm)[1:3] ≈ coef(m) rtol = 1e-12
    @test coef(mm)[4:5] == [0.0, 0.0]
    @test all(isnan, stderror(mm)[4:5])
    @test dof(mm) == 3
    @test globaltests(mm).lr.df == 3
    @test aic(mm) ≈ aic(m)
    out = sprint(show, MIME"text/plain"(), mm)
    @test occursin("linear combinations", out)
    @test test_report_layout(out) == 6
    est = table_rows(out, "Analysis of Conditional Maximum Likelihood Estimates")
    @test est[4] == ["x4", "0", "0", ".", ".", "."]
    @test est[5] == ["sc", "0", "0", ".", ".", "."]
    @test occursin("\n\nNote: parameters with DF 0", out)    # a text line between tables
    @test length(oddsratios(mm).effect) == 3
end

@testset "categorical covariates and odds ratios" begin
    md = clogit(@formula(y ~ x1 + g), df; strata = :id)
    me = clogit(@formula(y ~ x1 + g), df; strata = :id,
                contrasts = Dict(:g => EffectsCoding(base = "c")))
    mr = clogit(@formula(y ~ x1 + g), df; strata = :id,
                contrasts = Dict(:g => DummyCoding(base = "c")))
    @test loglikelihood(md) ≈ loglikelihood(me) rtol = 1e-12
    ord = oddsratios(md)
    @test ord.effect == ["x1", "g b vs a", "g c vs a"]
    @test ord.estimate[2] ≈ exp(coef(md)[2]) rtol = 1e-12
    # effect coding (SAS PARAM=EFFECT): exp(β) is not the OR; contrasts give the same
    # level-vs-reference ORs as reference coding with the same base
    ore = oddsratios(me); orr = oddsratios(mr)
    @test ore.effect == orr.effect == ["x1", "g a vs c", "g b vs c"]
    @test ore.estimate ≈ orr.estimate rtol = 1e-6
    @test ore.lower ≈ orr.lower rtol = 1e-6
    @test ore.upper ≈ orr.upper rtol = 1e-6
    # interactions: terms involved are omitted from the OR table
    mi = clogit(@formula(y ~ x1 + x2 + x1 & x2), df; strata = :id)
    @test isempty(oddsratios(mi).effect)
    mi2 = clogit(@formula(y ~ x3 + x1 * x2), df; strata = :id)
    @test oddsratios(mi2).effect == ["x3"]
    # general contrast
    r = oddsratio(md, [0.0, 1.0, -1.0])
    @test r.estimate ≈ exp(coef(md)[2] - coef(md)[3])
    @test r.lower < r.estimate < r.upper
    @test DataFrame(ord).effect == ord.effect
end

@testset "strata positionally and as keyword in both interfaces" begin
    mp = fit(ConditionalLogitModel, f, df, :id)
    @test mp isa ConditionalLogitModel
    @test coef(mp) == coef(m)
    @test coef(fit(ConditionalLogitModel, f, df, df.id)) == coef(m)
    X = Matrix(df[:, [:x1, :x2, :x3]])
    @test coef(clogit(X, df.y; strata = df.id)) == coef(m)
    @test coef(fit(ConditionalLogitModel, X, df.y; strata = df.id)) == coef(m)
    # positional strata with missing data still align with the rows
    d2 = allowmissing(copy(df)); d2.x1[4] = missing
    @test coef(fit(ConditionalLogitModel, f, d2, d2.id)) == coef(clogit(f, d2; strata = :id))
    # strata given as String column names
    @test coef(clogit(f, df; strata = ["id"])) == coef(m)
    @test coef(clogit(f, df; strata = "id")) == coef(m)
end

@testset "option validation" begin
    @test_throws ArgumentError clogit(f, df; strata = :id, ntasks = 0)
    @test_throws ArgumentError clogit(f, df; strata = :id, ntasks = -3)
    @test_throws ArgumentError clogit(f, df; strata = :id, gconv = -1)
    @test_throws ArgumentError clogit(f, df; strata = :id, xconv = -1)
    @test_throws ArgumentError clogit(f, df; strata = :id, maxiter = -1)
    @test_throws ArgumentError clogit(f, df; strata = :id, singular = 2.0)
    @test_throws ArgumentError clogit(f, df; strata = :id, sc_n = 3)
    @test coef(clogit(f, df; strata = :id, sc_n = "all")) == coef(m)
    @test clogit(f, df; strata = :id, sc_n = "all").options.sc_n === :all
    err = try clogit(f, df; strata = :id, method = :efron) catch e e end
    @test occursin("offset", err.msg) && occursin("maxstephalving", err.msg)
end

@testset "no usable rows, extreme FREQ" begin
    d2 = allowmissing(copy(df)); d2.x1 .= missing
    @test_throws r"no observations left" clogit(f, d2; strata = :id)
    @test_throws r"no observations left" clogit(f, df; strata = :id, freq = fill(-1, nrow(df)))
    @test_throws r"no observations left" clogit(f, df[1:0, :]; strata = :id)
    X = Matrix(df[:, [:x1, :x2, :x3]])
    @test_throws r"no observations left" clogit(X, df.y, df.id; freq = zeros(Int, nrow(df)))
    @test_throws r"too large" clogit(f, df; strata = :id, freq = fill(1e30, nrow(df)))
end

@testset "collinearity default matches SAS SINGULAR (1e7·eps)" begin
    # x2 = x1 + small noise: 1 - R² ≈ 5e-9, between 1e7·eps ≈ 2.2e-9 and the old 1e-8
    rng = StableRNG(9)
    K = 300
    s = repeat(1:K, inner = 3); y = repeat([1, 0, 0], K)
    x1 = randn(rng, 3K) .+ 0.7 .* y
    x2 = x1 .+ 7e-5 .* randn(rng, 3K) .+ 3e-5 .* y
    d, _ = design_of(hcat(x1, x2), y .== 1, s, ones(Int, 3K), zeros(3K))
    A = CL.weighted_gram(d)
    r2gap = (A[2, 2] - A[1, 2]^2 / A[1, 1]) / A[2, 2]
    @test 1e7 * eps() < r2gap < 1e-8
    mk = @test_logs clogit(hcat(x1, x2), y, s)
    @test aliased(mk) == [false, false]
    mo = quietly(() -> clogit(hcat(x1, x2), y, s; singular = 1e-8))
    @test aliased(mo) == [false, true]
    @test loglikelihood(mk) > loglikelihood(mo)
    @test CL.FitOptions().singular == 1e7 * eps()
end

@testset "formula path does not read unused columns" begin
    wide = copy(df)
    for j in 1:50
        wide[!, "junk$j"] = fill("z", nrow(df))
    end
    wide.bad = Vector{Any}(fill(nothing, nrow(df)))
    @test coef(clogit(f, wide; strata = :id)) == coef(m)
end
