# Development

## Tests

```bash
julia --project -e 'using Pkg; Pkg.test()'                      # full test suite (no R needed)
julia --project -e 'using Pkg; Pkg.test(julia_args=`-t 4`)'     # the same, multithreaded
```

The R reference fixtures in `test/reference/` are produced only by
`test/reference/generate_reference.R`, which needs R with the `survival` and `jsonlite`
packages:

```bash
Rscript test/reference/generate_reference.R
```

## Benchmarks

```bash
julia --project=benchmark -t 8 benchmark/bench.jl [choice mixed unequal bigstrata offset stringids]
julia --project=benchmark benchmark/run_threads.jl mixed          # thread sweep, checks bitwise-identical results
```

The benchmark environment uses `[sources]`, which needs Julia 1.11 or later. On Julia
1.10, run `julia --project=benchmark -e 'using Pkg; Pkg.develop(path = ".")'` once first.

## Documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The site is built into `docs/build/`. On every push to `main`, the `Documentation`
GitHub Actions workflow builds it and deploys it to the `gh-pages` branch.

## Source layout

| File | Contents |
|------|----------|
| `src/frontend.jl` | formula and matrix front ends, listwise deletion, event coding, option validation |
| `src/design.jl` | the stratified design: informative strata as contiguous, within-stratum centred column ranges; collinearity sweep |
| `src/likelihood.jl` | per-stratum kernels (softmax for one event, log-space recursion otherwise) and the threaded evaluator |
| `src/optimize.jl` | Newton–Raphson with SAS convergence criteria; separation diagnostics |
| `src/fit.jl` | glue from design to model |
| `src/inference.jl` | global tests, fit statistics, odds ratios |
| `src/statsapi.jl` | StatsAPI and StatsModels methods |
| `src/show.jl` | the SAS-style text report |
