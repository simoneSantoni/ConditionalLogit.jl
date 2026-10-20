"""
    ConditionalLogit

Conditional (fixed-effects / matched-set) logistic regression by exact conditional
maximum likelihood, written to reproduce SAS `PROC LOGISTIC` with a `STRATA`
statement (and `survival::clogit(method = "exact")` in R, Stata `clogit`).

Main entry points: [`clogit`](@ref) / `fit(ConditionalLogitModel, ...)`, the
StatsAPI accessors (`coef`, `stderror`, `vcov`, `loglikelihood`, `aic`, `bic`, ...),
and the SAS-table accessors [`fitstatistics`](@ref), [`globaltests`](@ref),
[`oddsratios`](@ref), [`stratasummary`](@ref).
"""
module ConditionalLogit

using LinearAlgebra
using Printf
using Base.Threads: Atomic, atomic_add!
using StatsAPI
using StatsAPI: RegressionModel
using StatsBase: StatsBase, CoefTable, AbstractWeights, FrequencyWeights, fweights
using StatsModels
using StatsModels: FormulaTerm, AbstractTerm, Term, ContinuousTerm, CategoricalTerm,
                   InteractionTerm, MatrixTerm, schema, apply_schema, modelcols, termvars
using StatsFuns: chisqccdf, norminvcdf
using Tables
using Missings: Missings, disallowmissing

import StatsAPI: fit, coef, coefnames, stderror, vcov, confint, coeftable, loglikelihood,
                 nullloglikelihood, nobs, dof, deviance, nulldeviance, informationmatrix,
                 score, islinear, responsename, modelmatrix, response, fitted, predict,
                 weights, dof_residual

export ConditionalLogitModel, clogit, @formula,
       # StatsAPI verbs (re-exported for convenience)
       fit, coef, coefnames, stderror, vcov, confint, coeftable, loglikelihood,
       nullloglikelihood, nobs, dof, aic, aicc, bic, deviance, nulldeviance,
       informationmatrix, score, islinear, responsename, modelmatrix, response,
       fitted, predict, weights, dof_residual,
       # SAS-style results
       converged, convergencestatus, aliased, iterations, iterationhistory,
       fitstatistics, globaltests, oddsratios, oddsratio, stratasummary,
       ConvergenceError

using StatsAPI: aic, aicc, bic

include("types.jl")
include("design.jl")
include("likelihood.jl")
include("optimize.jl")
include("fit.jl")
include("frontend.jl")
include("inference.jl")
include("statsapi.jl")
include("show.jl")
include("precompile.jl")

end # module
