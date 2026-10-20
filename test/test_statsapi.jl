# StatsAPI interface and display.

df, refs = load_fixture("infert")
m = clogit(@formula(case ~ spontaneous + induced), df; strata = :stratum)

@testset "accessors" begin
    @test coefnames(m) == ["spontaneous", "induced"]
    @test responsename(m) == "case"
    @test dof(m) == 2
    @test nobs(m) == 248
    @test deviance(m) ≈ -2loglikelihood(m)
    @test nulldeviance(m) ≈ -2nullloglikelihood(m)
    @test aic(m) ≈ -2loglikelihood(m) + 4
    @test bic(m) ≈ -2loglikelihood(m) + 2log(248)
    @test fitstatistics(m).sc.with ≈ bic(m)
    @test stderror(m) ≈ sqrt.(diag(vcov(m)))
    @test inv(informationmatrix(m)) ≈ vcov(m)
    @test norm(score(m)) < 1e-3
    @test !islinear(m)
    @test size(modelmatrix(m)) == (248, 2)
    @test response(m) == (df.case .== 1)
    @test weights(m) == fweights(ones(Int, 248))
    @test CL.StatsAPI.r2(m, :McFadden) ≈ 1 - loglikelihood(m) / nullloglikelihood(m)
    ci = confint(m)
    @test ci[:, 1] ≈ coef(m) .- 1.959963984540054 .* stderror(m)
    ct = coeftable(m)
    @test ct.rownms == coefnames(m)
    @test ct.cols[3] ≈ coef(m) ./ stderror(m)
    @test ct.cols[4] ≈ [CL.chisqccdf(1, z^2) for z in ct.cols[3]]
    @test StatsModels.formula(m) isa StatsModels.FormulaTerm
end

@testset "fitted = within-stratum inclusion probabilities" begin
    π = fitted(m)
    @test length(π) == 248
    for s in unique(df.stratum)
        idx = findall(==(s), df.stratum)
        @test sum(π[idx]) ≈ 1 rtol = 1e-12                       # m_k = 1: softmax
        η = modelmatrix(m)[idx, :] * coef(m)
        @test π[idx] ≈ exp.(η) ./ sum(exp.(η)) rtol = 1e-12
    end
    @test predict(m) == π
end

@testset "result tables are Tables.jl tables" begin
    gt = DataFrame(globaltests(m))
    @test gt.test == ["Likelihood Ratio", "Score", "Wald"]
    @test gt.chisq[2] ≈ refs["infert_si"]["score"] rtol = 1e-11
    fs = DataFrame(fitstatistics(m))
    @test fs.criterion == ["AIC", "SC", "-2 Log L"]
    @test DataFrame(oddsratios(m; level = 0.9)).lower[1] ≈
          exp(coef(m)[1] - 1.6448536269514722 * stderror(m)[1])
end

@testset "display" begin
    out = sprint(show, MIME"text/plain"(), m)
    for s in ("Model Information", "Strata Summary", "Model Fit Statistics",
              "Testing Global Null Hypothesis: BETA=0",
              "Analysis of Conditional Maximum Likelihood Estimates", "Odds Ratio Estimates",
              "spontaneous", "<.0001", "Newton-Raphson")
        @test occursin(s, out)
    end
    @test occursin("ConditionalLogitModel(p=2", sprint(show, m))
    d2, _ = load_fixture("synth_mk")
    mm = clogit(@formula(y ~ x1), d2; strata = :id, freq = ones(Int, nrow(d2)) .* 2)
    out2 = sprint(show, MIME"text/plain"(), mm)
    @test occursin("uninformative", out2)
    @test occursin("Sum of Frequencies Used", out2)
    @test test_report_layout(out) == 6
    @test test_report_layout(out2) == 6
    @test ["Sum of Frequencies Used", string(mm.freq_used)] in
          table_rows(out2, "Model Information")
    est = table_rows(out, "Analysis of Conditional Maximum Likelihood Estimates")
    @test first.(est) == ["spontaneous", "induced"]
    @test est[1][2:4] == ["1", @sprintf("%.4f", coef(m)[1]), @sprintf("%.4f", stderror(m)[1])]
    @test est[1][end] == "<.0001"
    @test first.(table_rows(out, "Odds Ratio Estimates")) == ["spontaneous", "induced"]
    # the one-line show is unchanged and has no table rules
    @test !occursin('─', sprint(show, m)) && !occursin('\n', sprint(show, m))
    # the result tables and the strata summary display on their own with the same layout
    @test test_report_layout(sprint(show, MIME"text/plain"(), fitstatistics(m))) == 1
    @test test_report_layout(sprint(show, MIME"text/plain"(), globaltests(m))) == 1
    @test test_report_layout(sprint(show, MIME"text/plain"(), oddsratios(m))) == 1
    @test test_report_layout(sprint(show, MIME"text/plain"(), stratasummary(m))) == 1
end

@testset "display: long and wide-character names keep the tables aligned" begin
    long = "spontaneous_abortions_with_a_very_long_parameter_name_αβ"
    wide = "自然流产"                   # textwidth 8, length 4
    d = rename(copy(df), :spontaneous => long, :induced => wide)
    ml = clogit(Term(:case) ~ Term(Symbol(long)) + Term(Symbol(wide)), d; strata = :stratum)
    out = sprint(show, MIME"text/plain"(), ml)
    @test test_report_layout(out) == 6
    est = table_rows(out, "Analysis of Conditional Maximum Likelihood Estimates")
    @test first.(est) == [long, wide]
    @test est[1][3] == @sprintf("%.4f", coef(m)[1])
    @test est[2][3] == @sprintf("%.4f", coef(m)[2])
    t = only(filter(t -> t.title == "Odds Ratio Estimates", report_tables(out)))
    # the label column is widened to the longest name (+ indent and gap)
    @test dispcol(t.rules[1], '┬') == 2 + textwidth(long) + 2
end

@testset "lrtest, isnested, dof_residual" begin
    la, _ = load_fixture("la")
    m1 = clogit(@formula(y ~ gall), la; strata = :s)
    m2 = clogit(@formula(y ~ gall + hyper), la; strata = :s)
    @test dof_residual(m1) == nobs(m1) - 1
    @test StatsModels.isnested(m1, m2)
    @test !StatsModels.isnested(m2, m1)
    t = @test_logs StatsModels.lrtest(m1, m2)
    @test t.pval[2] ≈ CL.chisqccdf(1, 2 * (loglikelihood(m2) - loglikelihood(m1))) rtol = 1e-12
    @test t.deviance == (deviance(m1), deviance(m2))
    # different observations are not nested
    m3 = clogit(@formula(y ~ gall + hyper), la[3:end, :]; strata = :s)
    @test !StatsModels.isnested(m1, m3)
    # a reparametrisation of the same column space is nested both ways
    la2 = copy(la); la2.g2 = 2 .* la2.gall .+ la2.hyper
    m4 = clogit(@formula(y ~ g2 + hyper), la2; strata = :s)
    @test StatsModels.isnested(m2, m4) && StatsModels.isnested(m4, m2)
end

@testset "predict on new data" begin
    p1 = predict(m, df; strata = :stratum)
    @test p1 ≈ fitted(m) rtol = 1e-12          # m_k = 1: within-stratum softmax
    lp = predict(m, df; strata = :stratum, type = :linpred)
    @test lp ≈ modelmatrix(m) * coef(m)
    # new choice sets, given as a vector of stratum values
    nd = DataFrame(spontaneous = [0, 1, 2, 0], induced = [0, 0, 1, 2], set = [1, 1, 2, 2])
    pr = predict(m, nd; strata = nd.set)
    η = Matrix(nd[:, [:spontaneous, :induced]]) * coef(m)
    @test pr[1:2] ≈ exp.(η[1:2]) ./ sum(exp.(η[1:2]))
    @test pr[3:4] ≈ exp.(η[3:4]) ./ sum(exp.(η[3:4]))
    # missing covariates give missing
    nd2 = allowmissing(nd); nd2.induced[2] = missing
    pm = predict(m, nd2; strata = :set)
    @test ismissing(pm[2]) && pm[1] ≈ 1.0 && pm[3:4] ≈ pr[3:4]
    # matrix models
    mx = clogit(modelmatrix(m), response(m), df.stratum)
    @test predict(mx, Matrix{Float64}(nd[:, 1:2]); strata = nd.set) ≈ pr
    @test_throws ArgumentError predict(mx, nd; strata = :set)
    @test_throws ArgumentError predict(m, nd; strata = :set, type = :response)
    @test_throws UndefKeywordError predict(m, nd)
end

@testset "result objects" begin
    ss = DataFrame(stratasummary(m))
    @test names(ss) == ["pattern", "events", "nonevents", "nstrata", "frequency", "informative"]
    @test ss.nstrata == [p.nstrata for p in stratasummary(m).patterns]
    @test isdefined(ConditionalLogit, :ConvergenceError) &&
          Base.isexported(ConditionalLogit, :ConvergenceError)
    # iteration history holds full-length parameter vectors, 0 for aliased
    d2 = copy(df); d2.sp2 = 2 .* d2.spontaneous
    ma = quietly(() -> clogit(@formula(case ~ spontaneous + sp2 + induced), d2; strata = :stratum))
    @test aliased(ma) == [false, true, false]
    h = iterationhistory(ma)
    @test all(r -> length(r.beta) == 3 && r.beta[2] == 0, h)
    @test h[end].beta == coef(ma)
    # confidence level validation
    for f in (l -> confint(m; level = l), l -> coeftable(m; level = l),
              l -> oddsratios(m; level = l), l -> oddsratio(m, [1.0, 0.0]; level = l))
        @test_throws ArgumentError f(95)
        @test_throws ArgumentError f(0.0)
    end
end

@testset "fitted is lazy and FREQ-aware" begin
    m2 = clogit(@formula(case ~ spontaneous + induced), df; strata = :stratum)
    @test m2.fittedcache[] === nothing
    π = fitted(m2)
    @test m2.fittedcache[] === π                 # cached
    @test fitted(m2) === π
    # aggregated 2×2 tables per stratum (SAS FREQ): memory is O(rows · m), not O(Σf · m)
    X = Float64[]; y = Int[]; s = Int[]; fr = Int[]
    for k in 1:3, (xx, yy, ff) in ((1, 1, 1000), (1, 0, 500), (0, 1, 500), (0, 0, 1000))
        push!(X, xx); push!(y, yy); push!(s, k); push!(fr, ff)
    end
    mf = clogit(reshape(X, :, 1), y, s; freq = fr)
    fitted(clogit(reshape(X[1:4], :, 1), y[1:4], s[1:4]; freq = [3, 2, 2, 3]))   # compile
    a = @allocated(fitted(mf))
    @test a < 4 * 2^20          # the copy-expanded tables would need 2·1501·3001·8 B per stratum
    πf = fitted(mf)
    @test πf == CL.compute_fitted(clogit(reshape(X, :, 1), y, s; freq = fr, ntasks = 1))
    for k in 1:3
        idx = findall(==(k), s)
        @test sum(fr[idx] .* πf[idx]) ≈ 1500 rtol = 1e-9
    end
end

@testset "SC note when n differs from the observations used" begin
    d2, _ = load_fixture("synth_mk")
    mm = clogit(@formula(y ~ x1), d2; strata = :id)
    out = sprint(show, MIME"text/plain"(), mm)
    if nobs(mm) != mm.freq_used
        @test occursin("SC uses n = $(nobs(mm))", out)
    end
    out1 = sprint(show, MIME"text/plain"(), m)
    @test !occursin("SC uses n", out1)
    @test occursin("Newton-Raphson (step-halving", out1)
end
