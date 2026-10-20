# Invariances of the conditional likelihood.

df, _ = load_fixture("synth_mk")
f = @formula(y ~ x1 + x2 + x3)
tight = (gconv = 1e-16, maxiter = 50)
m = clogit(f, df; strata = :id, tight...)

@testset "row permutation" begin
    rng = StableRNG(1)
    dp = df[randperm(rng, nrow(df)), :]
    mp = clogit(f, dp; strata = :id, tight...)
    @test coef(mp) ≈ coef(m) rtol = 1e-12
    @test vcov(mp) ≈ vcov(m) rtol = 1e-10
    @test loglikelihood(mp) ≈ loglikelihood(m) rtol = 1e-14
end

@testset "stratum relabelling and id types" begin
    d2 = copy(df)
    d2.id2 = [hash(s) % 100003 for s in df.id]          # integer ids
    d2.id3 = categorical(df.id)
    d2.part1 = first.(df.id, 2); d2.part2 = last.(df.id, 2)   # two-column key
    for (kw, label) in ((:id2, "int"), (:id3, "categorical"))
        mm = clogit(f, d2; strata = kw, tight...)
        @test coef(mm) ≈ coef(m) rtol = 1e-12
    end
    mm = clogit(f, d2; strata = [:part1, :part2], tight...)
    @test coef(mm) ≈ coef(m) rtol = 1e-12
    mm = clogit(f, d2; strata = collect(df.id), tight...)
    @test coef(mm) ≈ coef(m) rtol = 1e-12
    mm = clogit(f, d2; strata = "id", tight...)
    @test coef(mm) ≈ coef(m) rtol = 1e-12
end

@testset "within-stratum shift of a covariate" begin
    d2 = copy(df)
    shift = Dict(s => 1e4 * randn(StableRNG(hash(s))) for s in unique(df.id))
    d2.x1 = df.x1 .+ [shift[s] for s in df.id]
    mm = clogit(f, d2; strata = :id, tight...)
    @test coef(mm) ≈ coef(m) rtol = 1e-9
    @test stderror(mm) ≈ stderror(m) rtol = 1e-8
    @test loglikelihood(mm) ≈ loglikelihood(m) rtol = 1e-12
end

@testset "scaling" begin
    d2 = copy(df); d2.x1 = 10 .* df.x1
    mm = clogit(f, d2; strata = :id, tight...)
    @test coef(mm) ≈ coef(m) ./ [10, 1, 1] rtol = 1e-10
    @test loglikelihood(mm) ≈ loglikelihood(m) rtol = 1e-13
end

@testset "duplicating the data under new stratum ids" begin
    d2 = copy(df); d2.id = df.id .* "_copy"
    mm = clogit(f, vcat(df, d2); strata = :id, tight...)
    @test coef(mm) ≈ coef(m) rtol = 1e-10
    @test vcov(mm) ≈ vcov(m) ./ 2 rtol = 1e-9
    @test loglikelihood(mm) ≈ 2loglikelihood(m) rtol = 1e-13
end

@testset "uninformative strata do not change the fit" begin
    d2 = filter(r -> begin
        sub = df.y[df.id .== r.id]
        0 < sum(sub) < length(sub)
    end, df)
    mm = clogit(f, d2; strata = :id, tight...)
    @test coef(mm) ≈ coef(m) rtol = 1e-12
    @test stratasummary(mm).nuninformative == 0
    @test nobs(mm) == nobs(m)                      # sc_n = :informative
    @test loglikelihood(mm) ≈ loglikelihood(m) rtol = 1e-14
    ma = clogit(f, df; strata = :id, sc_n = :all, tight...)
    @test nobs(ma) == nrow(df)
    @test bic(ma) ≈ -2loglikelihood(m) + 3log(nrow(df))
end

@testset "matched pairs: closed forms (McNemar)" begin
    # b discordant pairs with the case exposed, c with the control exposed
    b, c, conc = 23, 9, 30
    rows = NamedTuple[]
    id = 0
    for _ in 1:b; id += 1; push!(rows, (id = id, y = 1, x = 1.0), (id = id, y = 0, x = 0.0)); end
    for _ in 1:c; id += 1; push!(rows, (id = id, y = 1, x = 0.0), (id = id, y = 0, x = 1.0)); end
    for t in 1:conc
        id += 1; v = Float64(t % 2)
        push!(rows, (id = id, y = 1, x = v), (id = id, y = 0, x = v))
    end
    dd = DataFrame(rows)
    mm = clogit(@formula(y ~ x), dd; strata = :id, tight...)
    @test coef(mm)[1] ≈ log(b / c) rtol = 1e-12
    @test stderror(mm)[1] ≈ sqrt(1 / b + 1 / c) rtol = 1e-10
    @test globaltests(mm).score.chisq ≈ (b - c)^2 / (b + c) rtol = 1e-12
end
