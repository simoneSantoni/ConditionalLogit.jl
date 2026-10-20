* Times Stata clogit on the scenarios in data/ (run scenarios.jl first).
*
*   cd benchmark/compare
*   stata-mp -b do stata_bench.do        (or: do stata_bench.do from the Stata prompt)
*
* Data are read in double precision (import delimited reads float by default).
* Default maximize options. Each scenario is timed with 1 processor and, on Stata/MP,
* with all licensed processors. The reported time is the median of 3 fits (data already
* in memory). Output: results/stata.csv and results/stata_coef.csv, in the same layout
* as the Julia and R results, so report.jl can merge them.

version 15
set more off
local scenarios pairs match1to4 choice multi panelfe bigstrata choicelarge
local reps 3
local procs 1
if c(MP) & c(processors_max) > 1 local procs 1 `c(processors_max)'

tempname res coefs
postfile `res' str16 scenario str24 tool threads rows double seconds iterations double loglik ///
    using "results/stata_raw.dta", replace
postfile `coefs' str16 scenario str8 term double estimate using "results/stata_coef_raw.dta", replace

foreach sc of local scenarios {
    import delimited using "data/`sc'.csv", clear asdouble
    unab xs : x*
    foreach np of local procs {
        if c(MP) set processors `np'
        matrix T = J(`reps', 1, .)
        forvalues r = 1/`reps' {
            timer clear 1
            timer on 1
            quietly clogit y `xs', group(s)
            timer off 1
            quietly timer list 1
            matrix T[`r', 1] = r(t1)
        }
        mata: st_numscalar("med", sort(st_matrix("T"), 1)[ceil(rows(st_matrix("T")) / 2)])
        post `res' ("`sc'") ("Stata clogit") (`np') (_N) (scalar(med)) (e(ic)) (e(ll))
        display "`sc'" _col(14) _N " rows  " %8.4f scalar(med) " s  (" e(ic) " iterations, `np' processors)"
    }
    foreach x of local xs {
        post `coefs' ("`sc'") ("`x'") (_b[`x'])
    }
}
postclose `res'
postclose `coefs'
use "results/stata_raw.dta", clear
format seconds loglik %21.0g
export delimited using "results/stata.csv", replace
use "results/stata_coef_raw.dta", clear
format estimate %21.0g
export delimited using "results/stata_coef.csv", replace
erase "results/stata_raw.dta"
erase "results/stata_coef_raw.dta"
display "Stata " c(stata_version) ", MP = " c(MP) ", processors = " c(processors_max)
