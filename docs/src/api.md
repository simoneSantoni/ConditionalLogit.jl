# API Reference

## Module

```@docs
ConditionalLogit
```

## Fitting

```@docs
clogit
fit(::Type{<:ConditionalLogitModel}, ::StatsModels.FormulaTerm, ::Any)
fit(::Type{<:ConditionalLogitModel}, ::AbstractMatrix{<:Union{Missing,Real}}, ::AbstractVector, ::AbstractVector)
ConditionalLogit.FitOptions
```

## Model Type

```@docs
ConditionalLogitModel
```

## SAS-style Results

```@docs
fitstatistics
globaltests
oddsratios
oddsratio
stratasummary
ConditionalLogit.FitStatistics
ConditionalLogit.GlobalTests
ConditionalLogit.OddsRatioTable
ConditionalLogit.StrataSummary
ConditionalLogit.StrataPattern
ConditionalLogit.ResultTable
```

## Convergence and Diagnostics

```@docs
converged
convergencestatus
iterations
iterationhistory
ConditionalLogit.IterationRecord
aliased
ConvergenceError
```

## StatsAPI Methods

The usual StatsAPI accessors (`coef`, `coefnames`, `stderror`, `aic`, `aicc`, `bic`,
`deviance`, `nulldeviance`, `responsename`, `weights`, `islinear`) also work on a
[`ConditionalLogitModel`](@ref). The methods below have behaviour specific to the
conditional logit model.

```@docs
vcov(::ConditionalLogitModel)
informationmatrix(::ConditionalLogitModel)
score(::ConditionalLogitModel)
loglikelihood(::ConditionalLogitModel)
nullloglikelihood(::ConditionalLogitModel)
nobs(::ConditionalLogitModel)
dof(::ConditionalLogitModel)
dof_residual(::ConditionalLogitModel)
confint(::ConditionalLogitModel)
coeftable(::ConditionalLogitModel)
modelmatrix(::ConditionalLogitModel)
response(::ConditionalLogitModel)
fitted(::ConditionalLogitModel)
predict(::ConditionalLogitModel)
StatsModels.isnested(::ConditionalLogitModel, ::ConditionalLogitModel)
```

## Index

```@index
Pages = ["api.md"]
```
