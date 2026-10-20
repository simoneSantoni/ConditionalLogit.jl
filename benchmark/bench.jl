# Benchmarks for ConditionalLogit.jl
#
#   julia --project=benchmark -t 8 benchmark/bench.jl [mode ...]
#
# Modes: choice (m_k = 1, n_k = 10), mixed (n_k = 10, m_k ~ U(1, 9)), unequal (n_k in
# 1..26), bigstrata (n_k = 200, m_k = 100), offset (covariates + 1e5, checks centring
# cost), stringids (string stratum ids). Default: choice mixed unequal bigstrata.
#
# Reports design build time, one evaluation of ℓ (ORD 0), ℓ+score (ORD 1),
# ℓ+score+information (ORD 2), allocations, and a full fit through the matrix interface
# and through the formula interface (Tables.jl NamedTuple input).

using ConditionalLogit, BenchmarkTools, Random, Printf
const CL = ConditionalLogit

"""
    gen(; n, K, p, mode, seed) -> (X, y, strata)

Synthetic data: X ~ N(0,1) (column 2 Bernoulli); events drawn by Gumbel top-m sampling
from exp(Xβ); rows shuffled so that grouping is exercised.
"""
function gen(; n = 10^6, K = 10^5, p = 20, mode = :choice, seed = 1)
    rng = Xoshiro(seed)
    if mode === :bigstrata
        K = 5000; sizes = fill(200, K)
    elseif mode === :unequal
        sizes = rand(rng, 1:26, K)
    else
        sizes = fill(n ÷ K, K)
    end
    N = sum(sizes)
    X = randn(rng, N, p)
    X[:, 2] .= rand(rng, N) .< 0.3
    mode === :offset && (X .+= 1e5)
    β = randn(rng, p) ./ sqrt(p)
    η = X * β
    y = falses(N)
    strata = Vector{Int}(undef, N)
    pos = 1
    for k in 1:K
        nk = sizes[k]
        r = pos:pos+nk-1
        mk = mode === :choice || mode === :offset || mode === :stringids ? 1 :
             mode === :bigstrata ? 100 : rand(rng, 0:nk)
        key = η[r] .- log.(-log.(rand(rng, nk)))
        y[r[partialsortperm(key, 1:mk; rev = true)]] .= true
        strata[r] .= k
        pos += nk
    end
    perm = randperm(rng, N)
    s = mode === :stringids ? ["stratum_$(strata[i])" for i in perm] : strata[perm]
    return X[perm, :], y[perm], s
end

function run(mode)
    X, y, s = gen(; mode)
    N, p = size(X)
    println("\n== mode = $mode   (N = $N, p = $p, threads = $(Threads.nthreads()))")
    codes, K = CL.strata_codes(s)
    tb = @belapsed CL.build_design($X, $y, $codes, $K, ones(Int, $N), zeros($N);
                                   ntasks = Threads.nthreads()) samples = 3 evals = 1
    d, summ = CL.build_design(X, y, codes, K, ones(Int, N), zeros(N))
    @printf("strata: %d informative, %d uninformative; %d chunks; max min(m, n-m) = %d\n",
            summ.ninformative, summ.nuninformative, length(d.chunks), d.maxmeff)
    @printf("build design            %8.1f ms\n", 1e3tb)
    ev = CL.Evaluator(d, Threads.nthreads())
    β = randn(Xoshiro(2), p) ./ 10
    for ord in 0:2
        t = @belapsed CL.evaluate!($ev, $β, Val($ord)) samples = 5 evals = 1
        a = @allocated CL.evaluate!(ev, β, Val(ord))
        @printf("evaluate ORD=%d          %8.1f ms   (%d bytes allocated)\n", ord, 1e3t, a)
    end
    ev1 = CL.Evaluator(d, 1)
    t1 = @belapsed CL.evaluate!($ev1, $β, Val(2)) samples = 3 evals = 1
    @printf("evaluate ORD=2, 1 task  %8.1f ms\n", 1e3t1)
    Xw, yw, sw = gen(; n = 10^4, K = 10^3, p, mode = mode === :bigstrata ? :mixed : mode)
    clogit(Xw, yw, sw)                              # compile
    tf = @elapsed m = clogit(X, y, s)
    @printf("full fit (matrix)       %8.1f ms   (%d iterations, converged = %s)\n",
            1e3tf, iterations(m), converged(m))
    SM = CL.StatsModels
    f = SM.term(:y) ~ sum(SM.term(Symbol("x$j")) for j in 1:p)
    tbl(X, y, s) = (; y = y, s = s, (Symbol("x$j") => X[:, j] for j in 1:size(X, 2))...)
    clogit(f, tbl(Xw, yw, sw); strata = :s)        # compile
    t = tbl(X, y, s)
    tff = @elapsed mf = clogit(f, t; strata = :s)
    @printf("full fit (formula)      %8.1f ms   (same estimates: %s)\n", 1e3tff,
            coef(mf) == coef(m))
end

if abspath(PROGRAM_FILE) == @__FILE__
    modes = isempty(ARGS) ? [:choice, :mixed, :unequal, :bigstrata] : Symbol.(ARGS)
    foreach(run, modes)
end
