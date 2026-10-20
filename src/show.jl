# SAS-style text output.

# SAS PVALUE6.4
_pv(p) = isnan(p) ? "." : p < 1e-4 ? "<.0001" : @sprintf("%.4f", p)
_f4(x) = isnan(x) ? "." : @sprintf("%.4f", x)
_f3(x) = isnan(x) ? "." : @sprintf("%.3f", x)
# SAS ODDSRATIO8.3 shows >999.999 / <0.001
_or(x) = isnan(x) ? "." : x > 999.999 ? ">999.999" : x < 0.001 ? "<0.001" : @sprintf("%.3f", x)

# Box-drawing characters for table rules and the label-column delimiter.
const _H = '─'
const _V = '│'

# Print a table: `header` is a vector of header lines (each a vector of cells), `rows`
# a vector of cell vectors. Column 1 (the row label) is left-aligned, the rest
# right-aligned. Layout:
#
#     Title
#     ───────────┬────────────────────────      (top rule)
#       Header   │  Col 1   Col 2              (header lines, if any)
#     ───────────┼────────────────────────      (header/body rule, if a header)
#       label    │    1.0     2.0              (body)
#     ───────────┴────────────────────────      (closing rule)
#
# Every rule spans the table's display width (content plus a 2-character right
# margin, or the title's width if wider); the `│` delimiter separates the label column
# from the others. With `delim = false` (key-value blocks) there is no delimiter and the
# remaining columns are left-aligned.
function _table(io::IO, title::AbstractString, header::Vector{Vector{String}},
                rows::Vector{Vector{String}}; indent::Int = 2, delim::Bool = true)
    allrows = vcat(header, rows)
    ncol = maximum(length, allrows; init = 1)
    w = zeros(Int, ncol)
    for r in allrows, (j, c) in enumerate(r)
        w[j] = max(w[j], textwidth(c))
    end
    gap = 3                                    # spaces between columns
    sep(j) = delim && j == 2 ? 2 : gap         # 2 spaces after the delimiter
    lwidth = indent + w[1] + (delim ? 2 : 0)   # display width before the delimiter
    rwidth = sum((sep(j) + w[j] for j in 2:ncol); init = 0) + 2   # + right margin
    dw = delim ? 1 : 0                         # display width of the delimiter
    total = max(lwidth + dw + rwidth, textwidth(title))
    rwidth = total - lwidth - dw
    rule(c) = delim ? string(_H^lwidth, c, _H^rwidth) : _H^total
    function line(r)
        b = IOBuffer()
        print(b, ' '^indent, rpad(get(r, 1, ""), w[1]))
        delim && print(b, "  ", _V)
        for j in 2:ncol
            c = get(r, j, "")
            print(b, ' '^sep(j), delim ? lpad(c, w[j]) : rpad(c, w[j]))
        end
        return rstrip(String(take!(b)))
    end
    println(io, title)
    println(io, rule('┬'))
    for r in header
        println(io, line(r))
    end
    isempty(header) || println(io, rule('┼'))
    for r in rows
        println(io, line(r))
    end
    println(io, rule('┴'))
    println(io)
end

function Base.show(io::IO, m::ConditionalLogitModel)
    print(io, "ConditionalLogitModel(p=", dof(m), ", strata=", m.summary.ninformative, "/",
          m.summary.nstrata, " informative, n=", nobs(m), ", -2LogL=",
          @sprintf("%.3f", -2 * m.loglik), m.converged ? "" : ", NOT CONVERGED", ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::ConditionalLogitModel)
    s = m.summary
    println(io, "Conditional logistic regression (exact conditional likelihood;")
    println(io, "cf. SAS PROC LOGISTIC with a STRATA statement)")
    println(io)
    info = [["Response Variable", m.responsename],
            ["Probability modeled", m.eventlabel],
            ["Strata Variable", m.stratalabel],
            ["Number of Strata", string(s.nstrata)],
            ["Number of Uninformative Strata", string(s.nuninformative)],
            ["Frequency Uninformative", string(s.freq_uninformative)],
            ["Number of Observations Read", string(m.nobs_read)],
            ["Number of Observations Used", string(m.nobs_used)]]
    if m.freq_read != m.nobs_read || m.freq_used != m.nobs_used
        push!(info, ["Sum of Frequencies Read", string(m.freq_read)])
        push!(info, ["Sum of Frequencies Used", string(m.freq_used)])
    end
    push!(info, ["Optimization Technique", "Newton-Raphson (step-halving; ridge if not PD)"])
    _table(io, "Model Information", Vector{String}[], info; delim = false)
    show(io, MIME"text/plain"(), s; responsename = m.responsename)
    println(io, m.convergence_message)
    println(io)
    show(io, MIME"text/plain"(), fitstatistics(m))
    if nobs(m) != m.freq_used
        println(io, "Note: SC uses n = ", nobs(m), " (sum of frequencies in informative ",
                "strata; sc_n = :", m.options.sc_n, ").\n")
    end
    show(io, MIME"text/plain"(), globaltests(m))
    _estimates(io, m)
    any(m.aliased) && println(io, "Note: parameters with DF 0 are linear combinations of ",
                              "other covariates within strata and were set to 0.\n")
    show(io, MIME"text/plain"(), oddsratios(m))
end

function Base.show(io::IO, ::MIME"text/plain", s::StrataSummary; responsename = "Response")
    rows = [[string(i), string(p.nevents), string(p.nnonevents), string(p.nstrata),
             string(p.freq)] for (i, p) in enumerate(s.patterns)]
    _table(io, "Strata Summary",
           [["Response", "", "", "Number of", ""],
            ["Pattern", "Events", "Non-events", "Strata", "Frequency"]], rows)
    if s.nuninformative > 0
        println(io, "Note: $(s.nuninformative) strata (frequency $(s.freq_uninformative)) ",
                "have only events or only non-events; they are uninformative and were not ",
                "used in the analysis.\n")
    end
end

function Base.show(io::IO, ::MIME"text/plain", t::FitStatistics)
    rows = [["AIC", _f3(t.aic.without), _f3(t.aic.with)],
            ["SC", _f3(t.sc.without), _f3(t.sc.with)],
            ["-2 Log L", _f3(t.m2logl.without), _f3(t.m2logl.with)]]
    _table(io, "Model Fit Statistics",
           [["", "Without", "With"], ["Criterion", "Covariates", "Covariates"]], rows)
end

function Base.show(io::IO, ::MIME"text/plain", t::GlobalTests)
    rows = [[nm, _f4(r.chisq), string(r.df), _pv(r.pvalue)]
            for (nm, r) in (("Likelihood Ratio", t.lr), ("Score", t.score), ("Wald", t.wald))]
    _table(io, "Testing Global Null Hypothesis: BETA=0",
           [["Test", "Chi-Square", "DF", "Pr > ChiSq"]], rows)
end

function _estimates(io::IO, m::ConditionalLogitModel)
    se = stderror(m)
    rows = Vector{String}[]
    for j in eachindex(m.coef)
        if m.aliased[j]
            push!(rows, [m.coefnames[j], "0", "0", ".", ".", "."])
        else
            w = (m.coef[j] / se[j])^2
            push!(rows, [m.coefnames[j], "1", _f4(m.coef[j]), _f4(se[j]), _f4(w),
                         _pv(chisqccdf(1, w))])
        end
    end
    _table(io, "Analysis of Conditional Maximum Likelihood Estimates",
           [["", "", "", "Standard", "Wald", ""],
            ["Parameter", "DF", "Estimate", "Error", "Chi-Square", "Pr > ChiSq"]], rows)
end

function Base.show(io::IO, ::MIME"text/plain", t::OddsRatioTable)
    lv = round(100 * t.level; digits = 1)
    lvs = isinteger(lv) ? string(Int(lv)) : string(lv)
    rows = [[t.effect[i], _or(t.estimate[i]), _or(t.lower[i]), _or(t.upper[i])]
            for i in eachindex(t.effect)]
    _table(io, "Odds Ratio Estimates",
           [["", "Point", "$lvs% Wald", ""], ["Effect", "Estimate", "Confidence", "Limits"]],
           rows)
end

Base.show(io::IO, t::ResultTable) = show(io, MIME"text/plain"(), t)
Base.show(io::IO, s::StrataSummary) = show(io, MIME"text/plain"(), s)
