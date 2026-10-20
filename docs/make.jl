using ConditionalLogit
using Documenter
using StatsModels

makedocs(;
    modules=[ConditionalLogit],
    checkdocs=:exports,
    authors="Simone Santoni",
    repo=Documenter.Remotes.GitHub("simoneSantoni", "ConditionalLogit.jl"),
    sitename="ConditionalLogit.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://simoneSantoni.github.io/ConditionalLogit.jl/stable/",
        assets=["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Comparison with Other Software" => [
            "Overview" => "comparison.md",
            "SAS PROC LOGISTIC" => "sas.md",
            "Stata clogit" => "stata.md",
            "R survival::clogit" => "r.md",
            "Python statsmodels" => "statsmodels.md",
        ],
        "Algorithm" => "algorithm.md",
        "Validation" => "validation.md",
        "Performance" => "performance.md",
        "Development" => "development.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/simoneSantoni/ConditionalLogit.jl",
    devbranch="main",
)
