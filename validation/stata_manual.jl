# Reproduces the examples of the Stata [R] clogit manual entry (Stata 19) and checks
# that ConditionalLogit.jl gives every printed number to the digits Stata prints. Run
# validation/stata_manual.R first to download the data.
#
#   julia --project=docs validation/stata_manual.jl
#   GCONV=1e-8 julia --project=docs validation/stata_manual.jl   # SAS stopping rule
#
# If data/stata_full.csv exists (written by Stata running validation/stata_manual.do),
# the script also compares with Stata's results at full precision.
#
# Stata iterates further than the SAS GCONV=1e-8 stopping point, so the comparison uses
# gconv = 1e-16 by default; with the SAS default the last printed digit can differ.

using ConditionalLogit, CSV, DataFrames, StatsModels, Printf

const DATA = joinpath(@__DIR__, "data")
const GCONV = (gconv = get(ENV, "GCONV", "1e-16") |> s -> parse(Float64, s),)

# Stata's printed values: (coef, se, z, lo, hi) per term, then the header.
const STATA = Dict(
    "clogitid" => (terms = Dict("x1" => (.653363, .2875215, 2.27, .0898312, 1.216895),
                                "x2" => (.0659169, .0449555, 1.47, -.0221943, .1540281)),
                   N = 369, lr = 9.07, ll = -123.41386, r2 = 0.0355),
    "lowbirth2" => (terms = Dict("lwt" => (-.0183757, .0100806, -1.82, -.0381333, .0013819),
                                 "smoke" => (1.400656, .6278396, 2.23, .1701131, 2.631199),
                                 "ptd" => (1.808009, .7886502, 2.29, .2622828, 3.353735),
                                 "ht" => (2.361152, 1.086128, 2.17, .2323796, 4.489924),
                                 "ui" => (1.401929, .6961585, 2.01, .0374836, 2.766375),
                                 "race: 2" => (.5713643, .689645, 0.83, -.7803149, 1.923044),
                                 "race: 3" => (-.0253148, .6992044, -0.04, -1.39573, 1.345101)),
                    N = 112, lr = 26.04, ll = -25.794271, r2 = 0.3355),
    "union" => (terms = Dict("age" => (.0170301, .004146, 4.11, .0089042, .0251561),
                             "grade" => (.0853572, .0418781, 2.04, .0032777, .1674368),
                             "not_smsa" => (.0083678, .1127963, 0.07, -.2127088, .2294445),
                             "south" => (-.748023, .1251752, -5.98, -.9933619, -.5026842),
                             "black" => (0.0, NaN, NaN, NaN, NaN)),
                N = 12035, lr = 68.09, ll = -4516.1385, r2 = 0.0075),
)

fits = Dict(
    "clogitid" => df -> clogit(@formula(y ~ x1 + x2), df; strata = :id, GCONV...),
    "lowbirth2" => df -> clogit(@formula(low ~ lwt + smoke + ptd + ht + ui + race), df;
                                strata = :pairid, contrasts = Dict(:race => DummyCoding()), GCONV...),
    "union" => df -> clogit(@formula(union ~ age + grade + not_smsa + south + black), df;
                            strata = :idcode, GCONV...),
)

"Decimals Stata printed for `x` (Stata shows at most 7 significant digits)."
decimals(x) = findfirst(k -> round(x; digits = k) == x, 0:12) - 1

"True when `ours` rounds to Stata's printed value `printed`."
agree(ours, printed) = abs(ours - printed) <= 0.5 * 10.0^-decimals(printed) * (1 + 1e-9)

function main()
    allok = true
    for nm in ("clogitid", "lowbirth2", "union")
        df = CSV.read(joinpath(DATA, nm * ".csv"), DataFrame)
        m = fits[nm](df)
        ref = STATA[nm]
        ci = confint(m)
        println("\n== ", nm)
        nbad = 0
        for (i, t) in enumerate(coefnames(m))
            s = ref.terms[t]
            b, se = coef(m)[i], stderror(m)[i]
            if s[1] == 0
                @printf("  %-9s omitted (aliased = %s)\n", t, aliased(m)[i])
                continue
            end
            ok = agree(b, s[1]) && agree(se, s[2]) && agree(ci[i, 1], s[4]) && agree(ci[i, 2], s[5])
            nbad += !ok
            @printf("  %-9s coef %11.7f (Stata %11.7f)  se %10.7f (Stata %10.7f)  z %6.2f (Stata %6.2f)\n",
                    t, b, s[1], se, s[2], b / se, s[3])
            allok &= round(b / se; digits = 2) == s[3]
        end
        lr = globaltests(m).lr.chisq
        r2 = 1 - loglikelihood(m) / nullloglikelihood(m)
        @printf("  N %d (Stata %d)   LR chi2 %.2f (Stata %.2f)   ll %.6f (Stata %.6f)   pseudo R2 %.4f (Stata %.4f)\n",
                nobs(m), ref.N, lr, ref.lr, loglikelihood(m), ref.ll, r2, ref.r2)
        println("  coefficients, std. errors and CI bounds that differ from Stata's printed digits: ", nbad)
        allok &= nobs(m) == ref.N && round(lr; digits = 2) == ref.lr &&
                 round(r2; digits = 4) == ref.r2 && agree(loglikelihood(m), ref.ll) && nbad == 0
    end
    println(allok ? "\nAll printed Stata values reproduced." : "\nSOME VALUES DIFFER")
    isfile(joinpath(DATA, "stata_full.csv")) && fullprecision()
end

# Full-precision comparison with Stata's own results (validation/stata_manual.do), with
# Stata's default and with tight convergence settings.
const TERMS = Dict("2.race" => "race: 2", "3.race" => "race: 3")

function fullprecision()
    st = CSV.read(joinpath(DATA, "stata_full.csv"), DataFrame)
    println("\nFull precision vs Stata (ConditionalLogit.jl with gconv = $(GCONV.gconv)):")
    @printf("  %-10s %-8s %10s %10s %10s %10s\n", "example", "Stata", "max|Δβ|", "max|Δse|", "|Δℓ|", "|Δℓ₀|")
    for g in groupby(st, [:example, :setting]; sort = false)
        nm = g.example[1]
        m = fits[nm](CSV.read(joinpath(DATA, nm * ".csv"), DataFrame))
        idx = Dict(n => i for (i, n) in enumerate(coefnames(m)))
        i = [idx[get(TERMS, t, t)] for t in g.term]
        @printf("  %-10s %-8s %10.1e %10.1e %10.1e %10.1e\n", nm, g.setting[1],
                maximum(abs.(coef(m)[i] .- g.estimate)), maximum(abs.(stderror(m)[i] .- g.se)),
                abs(loglikelihood(m) - g.ll[1]), abs(nullloglikelihood(m) - g.ll_0[1]))
    end
end

main()
