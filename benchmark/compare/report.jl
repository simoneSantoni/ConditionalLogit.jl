# Merges results/*.csv into the Markdown tables used in docs/src/performance.md.
#
#   julia --project=benchmark/compare benchmark/compare/report.jl

using CSV, DataFrames, Printf
include(joinpath(@__DIR__, "scenarios.jl"))
const RES = joinpath(@__DIR__, "results")

load(f) = isfile(joinpath(RES, f)) ? CSV.read(joinpath(RES, f), DataFrame) : nothing

fmt(x) = x === nothing ? "–" : x < 0.1 ? @sprintf("%.1f ms", 1000x) : @sprintf("%.2f s", x)
ratio(a, b) = a === nothing || b === nothing ? "–" : @sprintf("%.0f×", a / b)

function pick(df, sc, col = :seconds; threads = nothing)
    df === nothing && return nothing
    i = findfirst(r -> r.scenario == sc && (threads === nothing || r.threads == threads),
                  eachrow(df))
    return i === nothing ? nothing : df[i, col]
end

"Largest |β_tool − β_julia| (Julia fully converged) over the coefficients of a scenario."
function maxdiff(coefs, jcoef, sc)
    coefs === nothing && return nothing
    a = filter(:scenario => ==(sc), coefs); b = filter(:scenario => ==(sc), jcoef)
    d = innerjoin(a, b; on = :term, makeunique = true)
    return maximum(abs.(d.estimate .- d.estimate_1))
end

function main()
    r = load("r.csv"); st = load("stata.csv"); py = load("statsmodels.csv")
    jt = Dict(t => load("julia_t$t.csv") for t in (1, 8, 24))
    jcoef = load("julia_coef.csv"); rcoef = load("r_coef.csv"); scoef = load("stata_coef.csv")
    pcoef = load("statsmodels_coef.csv")
    smax = st === nothing ? nothing : maximum(st.threads)

    println("| Scenario | Rows | R `clogit` | Stata `clogit`, 1 core | Stata/MP `clogit`, $(smax) cores | statsmodels | Julia, 1 thread | Julia, 8 threads | Julia, 24 threads |")
    println("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    for sc in SCENARIOS
        tr = pick(r, sc.name); t1 = pick(jt[1], sc.name)
        s1 = pick(st, sc.name; threads = 1); sm = pick(st, sc.name; threads = smax)
        println("| `", sc.name, "` | ", sc.K * sc.n, " | ", fmt(tr), " | ", fmt(s1), " | ", fmt(sm),
                " | ", fmt(pick(py, sc.name)), " | ", fmt(t1), " | ", fmt(pick(jt[8], sc.name)),
                " | ", fmt(pick(jt[24], sc.name)), " |")
    end

    println("\nSpeedup of ConditionalLogit.jl (other time / Julia time):\n")
    println("| Scenario | Rows | vs R | vs Stata, 1 core | vs Stata/MP, $(smax) cores | vs statsmodels | vs R (Julia best) | vs Stata/MP (Julia best) | vs statsmodels (Julia best) |")
    println("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    rat(a, b) = a === nothing || b === nothing ? "–" : @sprintf("%.1f×", a / b)
    for sc in SCENARIOS
        t1 = pick(jt[1], sc.name); tb = minimum(pick(jt[t], sc.name) for t in (1, 8, 24))
        tr = pick(r, sc.name); tp = pick(py, sc.name)
        s1 = pick(st, sc.name; threads = 1); sm = pick(st, sc.name; threads = smax)
        println("| `", sc.name, "` | ", sc.K * sc.n, " | ", rat(tr, t1), " | ", rat(s1, t1), " | ",
                rat(sm, t1), " | ", rat(tp, t1), " | ", rat(tr, tb), " | ", rat(sm, tb), " | ",
                rat(tp, tb), " |")
    end

    println("\nFully converged (gconv = 1e-16), 1 thread:\n")
    println("| Scenario | R iterations | Stata iterations | Julia iterations (default / tight) | Julia, tight | R / Julia tight | Stata / Julia tight | statsmodels / Julia tight |")
    println("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for sc in SCENARIOS
        tt = pick(jt[1], sc.name, :seconds_tight)
        println("| `", sc.name, "` | ", pick(r, sc.name, :iterations), " | ",
                something(pick(st, sc.name, :iterations; threads = 1), "–"), " | ",
                pick(jt[1], sc.name, :iterations), " / ", pick(jt[1], sc.name, :iterations_tight),
                " | ", fmt(tt), " | ", ratio(pick(r, sc.name), tt), " | ",
                ratio(pick(st, sc.name; threads = 1), tt), " | ", ratio(pick(py, sc.name), tt), " |")
    end

    println("\nAgreement with ConditionalLogit.jl (gconv = 1e-16), max |Δβ|:\n")
    println("| Scenario | R | Stata | statsmodels |")
    println("|:--|--:|--:|--:|")
    for sc in SCENARIOS
        e(x) = x === nothing ? "–" : @sprintf("%.0e", x)
        println("| `", sc.name, "` | ", e(maxdiff(rcoef, jcoef, sc.name)), " | ",
                e(maxdiff(scoef, jcoef, sc.name)), " | ", e(maxdiff(pcoef, jcoef, sc.name)), " |")
    end
end

main()
