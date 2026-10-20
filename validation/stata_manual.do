* Runs the Stata [R] clogit manual examples in Stata and writes full-precision results
* to data/stata_full.csv, with default and with tight convergence settings, so that
* stata_manual.jl can compare ConditionalLogit.jl with Stata beyond the printed digits.
*
*   cd validation && stata-mp -b do stata_manual.do

version 15
set more off
tempname res
postfile `res' str12 example str8 setting str16 term double(estimate se) ///
    double(ll ll_0 N chi2) using "data/stata_full_raw.dta", replace

local tight nrtolerance(1e-12) tolerance(1e-14) ltolerance(1e-16) iterate(200)

foreach setting in default tight {
    local opts = cond("`setting'" == "tight", "`tight'", "")
    use "data/clogitid.dta", clear
    quietly clogit y x1 x2, group(id) `opts'
    foreach t in x1 x2 {
        post `res' ("clogitid") ("`setting'") ("`t'") (_b[`t']) (_se[`t']) (e(ll)) (e(ll_0)) (e(N)) (e(chi2))
    }
    use "data/lowbirth2.dta", clear
    quietly clogit low lwt smoke ptd ht ui i.race, group(pairid) `opts'
    foreach t in lwt smoke ptd ht ui 2.race 3.race {
        post `res' ("lowbirth2") ("`setting'") ("`t'") (_b[`t']) (_se[`t']) (e(ll)) (e(ll_0)) (e(N)) (e(chi2))
    }
    use "data/union.dta", clear
    quietly clogit union age grade not_smsa south black, group(idcode) `opts'
    foreach t in age grade not_smsa south {
        post `res' ("union") ("`setting'") ("`t'") (_b[`t']) (_se[`t']) (e(ll)) (e(ll_0)) (e(N)) (e(chi2))
    }
}
postclose `res'
use "data/stata_full_raw.dta", clear
format estimate se ll ll_0 chi2 %21.0g
export delimited using "data/stata_full.csv", replace
erase "data/stata_full_raw.dta"
