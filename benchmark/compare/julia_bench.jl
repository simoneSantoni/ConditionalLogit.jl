# Times ConditionalLogit.jl on the scenarios in data/ (run scenarios.jl first).
#
#   julia --project=benchmark/compare -t 1 benchmark/compare/julia_bench.jl
#   julia --project=benchmark/compare -t 8 benchmark/compare/julia_bench.jl
#
# Each fit uses the formula interface on a DataFrame read from CSV, with default
# settings (SAS stopping rule, GCONV = 1e-8) and, for a conservative comparison, fully
# converged (gconv = 1e-16), which takes one or two more Newton steps than R or Stata.
# One untimed fit per scenario compiles the code; the reported time is the median of
# `REPS` fits. The tight fit's coefficients are saved for the accuracy check.
# Output: results/julia_t<threads>.csv and results/julia_coef.csv.

using ConditionalLogit, CSV, DataFrames, Printf, Statistics
using StatsModels: term
include(joinpath(@__DIR__, "scenarios.jl"))

const REPS = 5
const RESDIR = joinpath(@__DIR__, "results")

function main()
    mkpath(RESDIR)
    nt = Threads.nthreads()
    rows = DataFrame(scenario = String[], tool = String[], threads = Int[], rows = Int[],
                     seconds = Float64[], iterations = Int[], loglik = Float64[],
                     seconds_tight = Float64[], iterations_tight = Int[])
    coefs = DataFrame(scenario = String[], term = String[], estimate = Float64[])
    for sc in SCENARIOS
        df = CSV.read(joinpath(DATADIR, sc.name * ".csv"), DataFrame)
        f = term(:y) ~ sum(term(Symbol("x$j")) for j in 1:sc.p)
        m = clogit(f, df; strata = :s)                     # compile / warm up
        times = [(@elapsed clogit(f, df; strata = :s)) for _ in 1:REPS]
        t = median(times)
        mt = clogit(f, df; strata = :s, gconv = 1e-16)
        tt = median([(@elapsed clogit(f, df; strata = :s, gconv = 1e-16)) for _ in 1:REPS])
        push!(rows, (sc.name, "ConditionalLogit.jl", nt, nrow(df), t, iterations(m),
                     loglikelihood(m), tt, iterations(mt)))
        @printf("%-12s %9d rows  %8.4f s  (%d iterations)  tight %8.4f s  (%d iterations)  %d threads\n",
                sc.name, nrow(df), t, iterations(m), tt, iterations(mt), nt)
        if nt == 1
            for (nm, b) in zip(coefnames(mt), coef(mt))
                push!(coefs, (sc.name, nm, b))
            end
        end
    end
    CSV.write(joinpath(RESDIR, "julia_t$(nt).csv"), rows)
    nt == 1 && CSV.write(joinpath(RESDIR, "julia_coef.csv"), coefs)
end

main()
