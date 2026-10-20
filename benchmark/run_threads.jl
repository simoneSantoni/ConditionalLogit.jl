# Thread sweep: fit the same data with julia -t t for several t, report the time per
# fit, and assert that the estimates are bitwise identical.
#
#   julia --project=benchmark benchmark/run_threads.jl [mode]

mode = isempty(ARGS) ? "mixed" : ARGS[1]
here = @__DIR__
code = """
include($(repr(joinpath(here, "bench.jl"))))   # defines gen(); runs nothing
X, y, s = gen(; mode = :$mode)
clogit(gen(; n = 10^4, K = 10^3, mode = :mixed)...)
t = @elapsed m = clogit(X, y, s)
print(Threads.nthreads(), " ", t, " ", hash(reinterpret(UInt64, coef(m))), " ",
      hash(reinterpret(UInt64, vec(vcov(m)))))
"""
results = []
for t in (1, 2, 4, 8, 16, 32)
    t > Sys.CPU_THREADS && continue
    out = read(`$(Base.julia_cmd()) --project=$here --startup-file=no -t $t -e $code`, String)
    nt, time, hb, hv = split(out)
    println(rpad("threads = $nt", 14), "fit: ", round(parse(Float64, time); digits = 3), " s   ",
            "hash(β) = ", hb)
    push!(results, (hb, hv))
end
@assert all(==(results[1]), results) "results differ across thread counts"
println("bitwise identical across thread counts: OK")
