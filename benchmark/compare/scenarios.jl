# Cross-software benchmark scenarios, shared by the Julia, R and Stata scripts through
# CSV files in benchmark/compare/data/ (columns: s, y, x1..xp).
#
#   julia --project=benchmark/compare benchmark/compare/scenarios.jl

using CSV, DataFrames, Random

const SCENARIOS = [
    # name,          strata, stratum size, events per stratum, covariates
    (name = "pairs",       K = 20_000,  n = 2,   m = :one,   p = 5),
    (name = "match1to4",   K = 20_000,  n = 5,   m = :one,   p = 10),
    (name = "choice",      K = 50_000,  n = 10,  m = :one,   p = 10),
    (name = "multi",       K = 20_000,  n = 10,  m = :mixed, p = 10),
    (name = "panelfe",     K = 10_000,  n = 20,  m = :panel, p = 5),
    (name = "bigstrata",   K = 500,     n = 100, m = :half,  p = 5),
    (name = "choicelarge", K = 100_000, n = 10,  m = :one,   p = 20),
]

const DATADIR = joinpath(@__DIR__, "data")

"""
    generate(sc; seed = 20261020) -> DataFrame

X ~ N(0, 1). With a fixed number of events per stratum (`:one`, `:mixed` = U{1..n-1},
`:half` = n/2), events are drawn by Gumbel top-m sampling from exp(Xβ), which is the
conditional logit model. `:panel` draws y ~ Bernoulli(logistic(α_k + Xβ)) with
α_k ~ N(0, 1), so some strata are all 0 or all 1 (uninformative), as in a fixed-effects
panel. Rows are shuffled so that every tool has to group them.
"""
function generate(sc; seed = 20261020)
    rng = Xoshiro(seed + hash(sc.name) % 1000)
    N = sc.K * sc.n
    X = randn(rng, N, sc.p)
    β = randn(rng, sc.p) ./ sqrt(sc.p)
    η = X * β
    y = zeros(Int, N)
    s = repeat(1:sc.K; inner = sc.n)
    for k in 1:sc.K
        r = (k-1)*sc.n+1:k*sc.n
        if sc.m === :panel
            α = randn(rng)
            for i in r
                y[i] = rand(rng) < 1 / (1 + exp(-(α + η[i])))
            end
        else
            mk = sc.m === :one ? 1 : sc.m === :half ? sc.n ÷ 2 : rand(rng, 1:sc.n-1)
            key = η[r] .- log.(-log.(rand(rng, sc.n)))
            y[r[partialsortperm(key, 1:mk; rev = true)]] .= 1
        end
    end
    perm = randperm(rng, N)
    df = DataFrame(s = s[perm], y = y[perm])
    for j in 1:sc.p
        df[!, "x$j"] = X[perm, j]
    end
    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(DATADIR)
    for sc in SCENARIOS
        df = generate(sc)
        CSV.write(joinpath(DATADIR, sc.name * ".csv"), df)
        println(rpad(sc.name, 12), " rows = ", nrow(df), ", strata = ", sc.K, ", p = ", sc.p)
    end
end
