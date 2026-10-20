# Shared test helpers.
using ConditionalLogit, Test, LinearAlgebra, Random, StableRNGs
using CSV, DataFrames, JSON, ForwardDiff, CategoricalArrays
using StatsBase: fweights, pweights
using StatsModels: StatsModels, Term, EffectsCoding, DummyCoding
using Printf

const CL = ConditionalLogit
const REFDIR = joinpath(@__DIR__, "reference")

"""Load `test/reference/<name>.csv` and `.json` (fixtures written by generate_reference.R)."""
function load_fixture(name)
    df = CSV.read(joinpath(REFDIR, name * ".csv"), DataFrame; missingstring = "NA")
    js = JSON.parsefile(joinpath(REFDIR, name * ".json"))
    fits = Dict(String(f["name"]) => f for f in js["fits"])
    return df, fits
end

jnum(x) = x === nothing ? NaN : Float64(x)
jvec(v) = Float64[jnum(x) for x in v]
jmat(v) = permutedims(reduce(hcat, [jvec(r) for r in v]))

logbinom(n, m) = (k = min(m, n - m); sum(log((n - k + i) / i) for i in 1:k; init = 0.0))

"""
    brute(X, y, codes, f, off, β) -> (ℓ, g, H)

Exact conditional log-likelihood, score and Hessian by explicit enumeration of all
m-subsets of the FREQ-expanded strata (BigFloat), from the *raw* (uncentred) data.
"""
function brute(X, y, codes, f, off, β)
    p = size(X, 2)
    ℓ = big(0.0); g = zeros(BigFloat, p); H = zeros(BigFloat, p, p)
    for k in unique(codes)
        idx = Int[]
        for i in findall(==(k), codes), _ in 1:f[i]
            push!(idx, i)
        end
        n = length(idx); m = count(i -> y[i], idx)
        (m == 0 || m == n) && continue
        η = [big(off[i]) + sum(big(X[i, r]) * β[r] for r in 1:p) for i in idx]
        Z = big(0.0); S1 = zeros(BigFloat, p); S2 = zeros(BigFloat, p, p)
        for mask in 0:(2^n-1)
            count_ones(mask) == m || continue
            sel = [((mask >> (t - 1)) & 1) == 1 for t in 1:n]
            w = exp(sum(η[sel]))
            xs = [sum(big(X[idx[t], r]) for t in 1:n if sel[t]; init = big(0.0)) for r in 1:p]
            Z += w; S1 .+= w .* xs; S2 .+= w .* (xs * xs')
        end
        E = S1 ./ Z
        ℓ += sum(η[t] for t in 1:n if y[idx[t]]) - log(Z)
        g .+= [sum(big(X[idx[t], r]) for t in 1:n if y[idx[t]]) for r in 1:p] .- E
        H .-= S2 ./ Z .- E * E'
    end
    return Float64(ℓ), Float64.(g), Float64.(H)
end

"""Random small problem: strata of size 1..nmax, random events (some uninformative),
optional FREQ and offset."""
function random_problem(rng; K = rand(rng, 1:6), nmax = 6, p = rand(rng, 1:4),
                        withfreq = rand(rng, Bool), withoffset = rand(rng, Bool), scale = 1.0)
    codes = Int[]; y = Bool[]; f = Int[]
    for k in 1:K
        n = rand(rng, 1:nmax)
        m = rand(rng, 0:n)
        ev = shuffle(rng, [trues(m); falses(n - m)])
        append!(codes, fill(k, n)); append!(y, ev)
        append!(f, withfreq ? rand(rng, 1:2, n) : ones(Int, n))
    end
    N = length(codes)
    X = scale .* randn(rng, N, p) .+ 3 .* randn(rng, 1, p)
    off = withoffset ? 0.5 .* randn(rng, N) : zeros(N)
    perm = randperm(rng, N)
    return X[perm, :], BitVector(y[perm]), codes[perm], f[perm], off[perm]
end

function design_of(X, y, codes, f, off)
    K = maximum(codes)
    d, s = CL.build_design(Matrix{Float64}(X), BitVector(y), codes, K, f, Vector{Float64}(off))
    return d, s
end

function eval_design(d, β; ntasks = 1)
    ev = CL.Evaluator(d, ntasks)
    ℓ = CL.evaluate!(ev, Vector{Float64}(β), Val(2))
    return ℓ, copy(ev.g), copy(ev.I)
end

# quiet fit (suppress expected warnings)
quietly(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

# --- Parsing the text/plain report -------------------------------------------------

const RULE_CHARS = ('─', '┬', '┼', '┴')
isrule(l) = !isempty(l) && all(in(RULE_CHARS), l)

"""Display column (0-based, by `textwidth`) of the first `c` in `l`, or `nothing`."""
function dispcol(l::AbstractString, c::AbstractChar)
    k = findfirst(c, l)
    return k === nothing ? nothing : textwidth(SubString(l, 1, prevind(l, k)))
end

"""
    report_tables(out) -> Vector of (title, rules, header, body, delim)

Split the text/plain report `out` into its tables. A table is a title line followed by
a rule; with a `│` delimiter (top rule has `┬`) it closes at the rule with `┴`, and lines
before the `┼` rule are the header. Without a delimiter it closes at its second rule.
"""
function report_tables(out::AbstractString)
    lines = split(out, '\n')
    tables = []
    i = 2
    while i <= length(lines)
        if isrule(lines[i]) && !isempty(strip(lines[i-1])) && !isrule(lines[i-1])
            delim = occursin('┬', lines[i])
            rules, header, body = [String(lines[i])], String[], String[]
            j = i + 1
            while j <= length(lines)
                l = String(lines[j])
                if isrule(l)
                    push!(rules, l)
                    occursin('┼', l) && (header = body; body = String[])
                    (!delim || occursin('┴', l)) && break
                else
                    push!(body, l)
                end
                j += 1
            end
            push!(tables, (title = String(lines[i-1]), rules = rules, header = header,
                           body = body, delim = delim))
            i = j + 2
        else
            i += 1
        end
    end
    return tables
end

"""The cells of each body row of the table titled `title` in `out`: the label (left of
`│`, or of the first run of 2+ spaces in a key-value block) followed by the
whitespace-separated cells to its right."""
function table_rows(out::AbstractString, title::AbstractString)
    t = only(filter(t -> t.title == title, report_tables(out)))
    return [t.delim ? [String(strip(first(split(l, '│')))); split(last(split(l, '│')))] :
                      String.(split(strip(l), r"\s{2,}")) for l in t.body]
end

"""Test the layout of every table in `out`: all rules have the table's display width,
no line or title is wider, and the `│` delimiter (and `┬ ┼ ┴` junctions) sit at the same
display column on every line of the table. Returns the number of tables."""
function test_report_layout(out::AbstractString)
    tables = report_tables(out)
    for t in tables
        W = textwidth(t.rules[1])
        @test all(r -> textwidth(r) == W, t.rules)
        @test all(l -> textwidth(l) <= W, vcat(t.header, t.body))
        @test textwidth(t.title) <= W
        @test !isempty(t.body)
        if t.delim
            col = dispcol(t.rules[1], '┬')
            @test length(t.rules) == (isempty(t.header) ? 2 : 3)
            @test dispcol(t.rules[end], '┴') == col
            length(t.rules) == 3 && @test dispcol(t.rules[2], '┼') == col
            @test all(r -> count(c -> c in ('┬', '┼', '┴'), r) == 1, t.rules)
            @test all(l -> count(==('│'), l) == 1 && dispcol(l, '│') == col,
                      vcat(t.header, t.body))
        else
            @test length(t.rules) == 2
            @test all(r -> all(==('─'), r), t.rules)
            @test all(l -> !occursin('│', l), t.body)
        end
    end
    return length(tables)
end
