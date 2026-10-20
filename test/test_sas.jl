# SAS/STAT PROC LOGISTIC Example "Conditional Logistic Regression for Matched Pairs
# Data" (sample program LOGIEX11): 63 matched pairs from the Los Angeles endometrial
# cancer study.  SAS code:
#     proc logistic data=Data1;
#        strata ID;
#        model outcome(event='1')=Gall;
#     run;
# Every number SAS prints is compared at SAS's printed precision, with default options
# (GCONV=1E-8, estimate reported at the iterate where the criterion is met).

la, _ = load_fixture("la")
m = clogit(@formula(y ~ gall), la; strata = :s)

f4(x) = @sprintf("%.4f", x)
f3(x) = @sprintf("%.3f", x)

@test converged(m)
@test convergencestatus(m) === :converged
@test m.nobs_read == 126 && m.nobs_used == 126
@test stratasummary(m).nstrata == 63
@test stratasummary(m).nuninformative == 0
pat = only(stratasummary(m).patterns)
@test (pat.nevents, pat.nnonevents, pat.nstrata, pat.freq) == (1, 1, 63, 126)

fs = fitstatistics(m)
@test f3(fs.m2logl.without) == "87.337"
@test f3(fs.m2logl.with) == "83.654"
@test f3(fs.aic.without) == "87.337"
@test f3(fs.aic.with) == "85.654"
@test f3(fs.sc.without) == "87.337"
@test f3(fs.sc.with) == "88.490"          # n = 126 observations, not 63 strata
@test aic(m) ≈ fs.aic.with
@test bic(m) ≈ fs.sc.with

gt = globaltests(m)
@test f4(gt.lr.chisq) == "3.6830" && f4(gt.lr.pvalue) == "0.0550"
@test f4(gt.score.chisq) == "3.5556" && f4(gt.score.pvalue) == "0.0593"
@test f4(gt.wald.chisq) == "3.2970" && f4(gt.wald.pvalue) == "0.0694"
@test gt.lr.df == gt.score.df == gt.wald.df == 1

@test f4(coef(m)[1]) == "0.9555"
@test f4(stderror(m)[1]) == "0.5262"
@test f4((coef(m)[1] / stderror(m)[1])^2) == "3.2970"

orr = oddsratios(m)
@test orr.effect == ["gall"]
@test f3(orr.estimate[1]) == "2.600"
@test f3(orr.lower[1]) == "0.927"
@test f3(orr.upper[1]) == "7.293"

# the printed report carries the same numbers, in the right table rows
out = sprint(show, MIME"text/plain"(), m)
for s in ("87.337", "83.654", "85.654", "88.490", "3.6830", "3.5556", "3.2970", "0.0550",
          "0.0593", "0.0694", "0.9555", "0.5262", "2.600", "0.927", "7.293",
          "GCONV=1E-8", "Analysis of Conditional Maximum Likelihood Estimates",
          "Testing Global Null Hypothesis: BETA=0", "Odds Ratio Estimates")
    @test occursin(s, out)
end
@test [t.title for t in report_tables(out)] ==
      ["Model Information", "Strata Summary", "Model Fit Statistics",
       "Testing Global Null Hypothesis: BETA=0",
       "Analysis of Conditional Maximum Likelihood Estimates", "Odds Ratio Estimates"]
@test test_report_layout(out) == 6
info = table_rows(out, "Model Information")
@test ["Response Variable", "y"] in info
@test ["Number of Strata", "63"] in info
@test ["Number of Observations Used", "126"] in info
@test table_rows(out, "Strata Summary") == [["1", "1", "1", "63", "126"]]
@test table_rows(out, "Model Fit Statistics") ==
      [["AIC", "87.337", "85.654"], ["SC", "87.337", "88.490"],
       ["-2 Log L", "87.337", "83.654"]]
@test table_rows(out, "Testing Global Null Hypothesis: BETA=0") ==
      [["Likelihood Ratio", "3.6830", "1", "0.0550"], ["Score", "3.5556", "1", "0.0593"],
       ["Wald", "3.2970", "1", "0.0694"]]
@test table_rows(out, "Analysis of Conditional Maximum Likelihood Estimates") ==
      [["gall", "1", "0.9555", "0.5262", "3.2970", "0.0694"]]
@test table_rows(out, "Odds Ratio Estimates") == [["gall", "2.600", "0.927", "7.293"]]
@test occursin("\nConvergence criterion (GCONV=1E-8) satisfied.\n", out)

# SAS default response ordering (no EVENT='1'): models Outcome = 0. Same fit
# statistics and tests, negated estimate, reciprocal odds ratio.
m0 = clogit(@formula(y ~ gall), la; strata = :s, event = 0)
@test f4(coef(m0)[1]) == "-0.9555"
@test f4(stderror(m0)[1]) == "0.5262"
@test f3(-2loglikelihood(m0)) == "83.654"
@test f4(globaltests(m0).score.chisq) == "3.5556"
@test oddsratios(m0).estimate[1] ≈ 1 / oddsratios(m).estimate[1]
@test f3(oddsratios(m0).estimate[1]) == "0.385"
