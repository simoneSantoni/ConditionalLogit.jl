# Performance

## ConditionalLogit.jl compared with R, Stata and Python

Seven synthetic scenarios were fitted, from 40,000 to 1,000,000 rows. With default
settings and one thread, ConditionalLogit.jl was:

- **7–22× faster** than R `survival::clogit(method = "exact")`;
- **13–41× faster** than Stata `clogit` on one core;
- **144–324× faster** than Python `statsmodels` `ConditionalLogit`.

The "best" comparison uses each program's fastest multi-core configuration: Stata/MP on
16 cores, and ConditionalLogit.jl on 8 or 24 threads. On that basis ConditionalLogit.jl
was 18–53× faster than Stata/MP. All three programs give the same coefficients, up to
their convergence tolerances (see [Agreement](@ref)).

| Scenario | Rows | R `clogit` | Stata `clogit`, 1 core | Stata/MP `clogit`, 16 cores | statsmodels | Julia, 1 thread | Julia, 8 threads | Julia, 24 threads |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `pairs` | 40,000 | 0.14 s | 0.48 s | 0.52 s | 3.72 s | 11.6 ms | 9.7 ms | 18.1 ms |
| `match1to4` | 100,000 | 0.69 s | 1.20 s | 1.02 s | 10.01 s | 30.9 ms | 24.4 ms | 47.7 ms |
| `choice` | 500,000 | 3.41 s | 5.01 s | 3.73 s | 35.68 s | 0.22 s | 0.15 s | 0.22 s |
| `multi` | 200,000 | 2.61 s | 6.96 s | 2.91 s | 35.02 s | 0.18 s | 81.4 ms | 68.6 ms |
| `panelfe` | 200,000 | 1.69 s | 4.81 s | 2.60 s | 44.59 s | 0.15 s | 62.1 ms | 63.5 ms |
| `bigstrata` | 50,000 | 1.79 s | 3.29 s | 1.00 s | 70.32 s | 0.25 s | 74.8 ms | 56.8 ms |
| `choicelarge` | 1,000,000 | 11.80 s | 18.94 s | 11.51 s | 108.92 s | 0.76 s | 0.50 s | 0.52 s |

### Speedup

Each cell is the other program's time divided by ConditionalLogit.jl's time, so a value
above 1× means ConditionalLogit.jl finished first:

- **"vs R", "vs Stata, 1 core", "vs Stata/MP, 16 cores", "vs statsmodels":**
  ConditionalLogit.jl on one thread. R and statsmodels are single-threaded.
- **"Julia best" columns:** ConditionalLogit.jl at its fastest of 1, 8 or 24 threads.
- **Settings:** every program used its default settings on the same data.

| Scenario | Rows | vs R | vs Stata, 1 core | vs Stata/MP, 16 cores | vs statsmodels | vs R (Julia best) | vs Stata/MP (Julia best) | vs statsmodels (Julia best) |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 1:1 matched pairs (`pairs`) | 40,000 | 12.2× | 41.4× | 44.6× | 322× | 14.5× | 53.0× | 383× |
| 1:4 matching (`match1to4`) | 100,000 | 22.5× | 38.9× | 33.0× | 324× | 28.5× | 41.8× | 411× |
| choice sets of 10 (`choice`) | 500,000 | 15.6× | 23.0× | 17.1× | 164× | 22.7× | 24.8× | 237× |
| 1–9 events per stratum (`multi`) | 200,000 | 14.8× | 39.4× | 16.5× | 198× | 38.1× | 42.5× | 511× |
| fixed-effects panel (`panelfe`) | 200,000 | 11.0× | 31.2× | 16.9× | 290× | 27.3× | 41.8× | 719× |
| 100-row strata, 50 events (`bigstrata`) | 50,000 | 7.3× | 13.3× | 4.0× | 285× | 31.5× | 17.6× | 1237× |
| choice sets of 10, 20 covariates (`choicelarge`) | 1,000,000 | 15.6× | 25.1× | 15.2× | 144× | 23.6× | 23.0× | 218× |

The narrowest margin is Stata/MP on 16 cores on `bigstrata`, at 4.0× against
single-threaded ConditionalLogit.jl. Ratios for the smallest data sets are noisier,
because ConditionalLogit.jl's times there are only 10–30 ms.

### Scenarios

Each scenario is drawn from a conditional logit model with N(0, 1) covariates. Rows are
shuffled, so every program has to group them by stratum.

| Scenario | Strata | Stratum size | Events per stratum | Covariates | Typical use |
|:--|--:|--:|--:|--:|:--|
| `pairs` | 20,000 | 2 | 1 | 5 | 1:1 matched case–control |
| `match1to4` | 20,000 | 5 | 1 | 10 | 1:4 matched case–control |
| `choice` | 50,000 | 10 | 1 | 10 | McFadden choice model |
| `multi` | 20,000 | 10 | 1–9 | 10 | k₁:k₂ matching |
| `panelfe` | 10,000 | 20 | Bernoulli; includes all-0 and all-1 strata | 5 | fixed-effects logit panel |
| `bigstrata` | 500 | 100 | 50 | 5 | large risk sets |
| `choicelarge` | 100,000 | 10 | 1 | 20 | large choice data |

### Same stopping point

The three programs stop at different points:

- ConditionalLogit.jl stops by default at the SAS `GCONV = 1e-8` iterate.
- R stops when the relative change in ℓ is below 1e-9.
- Stata starts from its own initial values and stops when the scaled gradient is below
  1e-5 (`e(ic)` counts its iterations).
- statsmodels uses BFGS and stops when the gradient norm of −ℓ/n is below 1e-5. It does
  not report an iteration count.

The table below repeats the one-thread comparison with ConditionalLogit.jl fully
converged (`gconv = 1e-16`). That is more accurate than either R or Stata, so this is
the conservative comparison: 6–19× faster than R, 11–49× faster than Stata on one core,
and 125–385× faster than statsmodels.

| Scenario | R iterations | Stata iterations | Julia iterations (default / `gconv = 1e-16`) | Julia, `gconv = 1e-16` | R / Julia | Stata / Julia | statsmodels / Julia |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `pairs` | 4 | 3 | 3 / 4 | 9.7 ms | 15× | 49× | 385× |
| `match1to4` | 4 | 2 | 3 / 4 | 35.6 ms | 19× | 34× | 281× |
| `choice` | 4 | 2 | 3 / 4 | 0.24 s | 14× | 21× | 147× |
| `multi` | 5 | 3 | 4 / 5 | 0.22 s | 12× | 32× | 161× |
| `panelfe` | 4 | 3 | 3 / 3 | 0.16 s | 11× | 31× | 287× |
| `bigstrata` | 5 | 2 | 4 / 5 | 0.30 s | 6× | 11× | 233× |
| `choicelarge` | 4 | 2 | 3 / 4 | 0.87 s | 14× | 22× | 125× |

### Agreement

The table gives the largest absolute difference between each program's coefficients and
the fully converged ConditionalLogit.jl fit.

| Scenario | R | Stata | statsmodels |
|:--|--:|--:|--:|
| `pairs` | 2e-15 | 4e-12 | 1e-05 |
| `match1to4` | 3e-15 | 2e-07 | 2e-05 |
| `choice` | 3e-15 | 5e-10 | 9e-05 |
| `multi` | 5e-15 | 7e-09 | 4e-05 |
| `panelfe` | 7e-09 | 7e-09 | 4e-05 |
| `bigstrata` | 1e-14 | 1e-09 | 1e-05 |
| `choicelarge` | 2e-14 | 4e-09 | 1e-04 |

The differences from Stata come from Stata's default stopping rule, not from the
likelihood. Refitting `match1to4` in Stata with `nrtolerance(1e-12) tolerance(1e-14)`
shrinks the difference from 2e-7 to 4e-10. The same holds for R on `panelfe`, where R
stops a little early. statsmodels' default BFGS stops much earlier, at about 1e-5 (see
[Python statsmodels](@ref)); with `method="newton"` it converges to the same estimates.

### Method

- **What is timed:** a complete fit from data already in memory, through each program's
  usual interface. That covers grouping, the design matrix, all iterations and the
  covariance matrix. Reading the CSV file is not timed.
- **Repetitions:** R and Stata give the median of 3 fits. statsmodels also gives the
  median of 3 fits, except where a single fit takes longer than 30 s, which is timed
  once. Its times varied by 10–30% between two runs. ConditionalLogit.jl gives the
  median of 5 fits after one untimed fit that compiles the code. Times under 0.1 s are
  noisy at the millisecond level.
- **Settings:** each program's defaults. R and statsmodels are single-threaded. Stata/MP was run with
  `set processors 1` and with all 16 licensed processors.
- **Precision:** Stata reads the CSV files with `import delimited, asdouble`. Without
  `asdouble` it stores the data as single-precision floats, and its estimates then
  differ in about the 8th digit.
- **Host:** Intel Core i9-14900HX (8 performance and 16 efficiency cores, 32 hardware
  threads), Linux, Julia 1.13.0, R 4.6.1 with `survival` 3.8-6, Stata/MP 15 (16-core
  licence), Python 3.13.14 with `statsmodels` 0.15.0.

### Interpretation

- **One thread:** most of the gain comes from the kernels. Strata with one event use a
  centred softmax, and the general case uses the log-space recursion described in
  [Algorithm](@ref). The cost is O(n_k · min(m_k, n_k − m_k) · p²) per stratum, and the
  per-stratum work allocates nothing.
- **Threads:** threading helps most when the per-stratum work is heavy (`multi`,
  `panelfe`, `bigstrata`: 2.4–4.4× from 1 to 24 threads). With one event per stratum,
  the fixed, partly serial cost of grouping and building the design dominates, so the
  gain is small (`choicelarge`: 1.5×). On this host, going past the 8 performance cores
  can slow the smallest problems down.
- **statsmodels** runs its recursion in pure Python, one group at a time, which
  accounts for most of its gap.
- **Stata/MP** gains up to 3.3× from 16 cores on the heavy scenarios. On `pairs` it is
  slightly slower with 16 cores than with 1.
- **Absolute times depend on the machine.** The ratios show the difference between the
  implementations on this benchmark; they do not predict the speedup on every data set.

## Reproducing the benchmarks

```bash
julia --project=benchmark/compare -e 'using Pkg; Pkg.instantiate()'
julia --project=benchmark/compare benchmark/compare/scenarios.jl         # writes data/*.csv (≈ 570 MB)
Rscript benchmark/compare/r_bench.R                                       # needs survival, data.table
uv venv benchmark/compare/.venv && uv pip install --python benchmark/compare/.venv/bin/python statsmodels pandas
benchmark/compare/.venv/bin/python benchmark/compare/py_bench.py
for t in 1 8 24; do julia --project=benchmark/compare -t $t benchmark/compare/julia_bench.jl; done
(cd benchmark/compare && stata-mp -b do stata_bench.do)                   # Stata 15 or later
julia --project=benchmark/compare benchmark/compare/report.jl            # Markdown tables
```

The raw results behind the tables above are in `benchmark/compare/results/`.

## Package-internal benchmarks

`benchmark/bench.jl` times the individual stages (design build, one evaluation of ℓ,
score and information, and full fits through the matrix and formula interfaces), and
`benchmark/run_threads.jl` sweeps thread counts and checks that the results are bitwise
identical. See [Development](@ref).
