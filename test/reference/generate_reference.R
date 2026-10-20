# Generates the reference fixtures used by test/runtests.jl.
#
#   Rscript test/reference/generate_reference.R      (run from the package root)
#
# Oracle: survival::clogit(..., method = "exact"), which maximises the same exact
# conditional likelihood as SAS PROC LOGISTIC with a STRATA statement.
# Requires R with the 'survival' and 'jsonlite' packages.  Output is deterministic
# (fixed seeds).  Never hand-edit the generated CSV/JSON files: rerun this script.

suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
outdir <- if (length(script)) dirname(normalizePath(script)) else "test/reference"

tight <- coxph.control(eps = 1e-14, toler.chol = 1e-15, iter.max = 200)

num <- function(x) {
  x <- as.numeric(x)
  x[!is.finite(x)] <- NA
  x
}

# Summarise one exact clogit fit.  `at_beta`, if given, is a coefficient vector at
# which the log-likelihood, score and covariance are additionally evaluated without
# iterating (iter.max = 0); this lets the Julia tests check the likelihood kernel at a
# fixed point, independently of the stopping rule.
summarise <- function(name, formula, data, covariates, at_beta = NULL, note = "") {
  fit <- suppressWarnings(clogit(formula, data = data, method = "exact", control = tight))
  b <- coef(fit)
  V <- vcov(fit)
  al <- is.na(b)
  se <- sqrt(diag(V))
  se[al] <- NA
  Vout <- V
  Vout[al, ] <- NA
  Vout[, al] <- NA
  res <- list(
    name = name,
    note = note,
    formula = paste(deparse(formula), collapse = " "),
    covariates = I(covariates),
    coefnames = I(names(b)),
    aliased = I(as.logical(al)),
    coef = I(num(b)),
    se = I(num(se)),
    vcov = lapply(seq_len(nrow(Vout)), function(i) I(num(Vout[i, ]))),
    loglik0 = fit$loglik[1],
    loglik = fit$loglik[2],
    lr = 2 * (fit$loglik[2] - fit$loglik[1]),
    score = unname(fit$score),
    wald = unname(fit$wald.test),
    df = sum(!al),
    n = fit$n,
    nevent = fit$nevent,
    iter = fit$iter
  )
  if (!is.null(at_beta)) {
    g <- suppressWarnings(clogit(formula, data = data, method = "exact", init = at_beta,
                                 control = coxph.control(iter.max = 0)))
    res$at_beta <- list(beta = I(at_beta), loglik = g$loglik[1],
                        vcov = lapply(seq_len(nrow(g$var)), function(i) I(num(g$var[i, ]))),
                        score = unname(g$score))
  }
  res
}

write_fixture <- function(dataname, df, fits) {
  write.csv(df, file.path(outdir, paste0(dataname, ".csv")), row.names = FALSE, na = "NA")
  meta <- list(R = R.version.string, survival = as.character(packageVersion("survival")),
               source = "survival::clogit(method = 'exact')",
               control = "coxph.control(eps = 1e-14, toler.chol = 1e-15, iter.max = 200)")
  out <- list(meta = meta, data = paste0(dataname, ".csv"), fits = fits)
  write_json(out, file.path(outdir, paste0(dataname, ".json")), digits = NA,
             auto_unbox = TRUE, na = "null", pretty = TRUE)
}

# (I() keeps length-1 vectors as JSON arrays.)

# Draw m events by the Gumbel top-m trick (successive sampling without replacement with
# probability proportional to exp(eta)).  This is exactly the conditional logit model
# for m = 1 and a reasonable data generator otherwise; the fixtures only need data.
draw_events <- function(eta, m) {
  y <- integer(length(eta))
  if (m == 0) return(y)
  if (m == length(eta)) return(rep(1L, length(eta)))
  key <- eta - log(-log(runif(length(eta))))
  y[order(key, decreasing = TRUE)[seq_len(m)]] <- 1L
  y
}

## ---------------------------------------------------------------- 1. SAS LOGIEX11
la <- read.csv(file.path(outdir, "la_input.csv"))
la <- la[, c("s", "y", "gall", "hyper")]
write_fixture("la", la, list(
  summarise("la_gall", y ~ gall + strata(s), la, c("gall"), at_beta = c(0.5),
            note = "SAS/STAT LOGISTIC Example 11 (LOGIEX11): 63 matched pairs"),
  summarise("la_gall_hyper", y ~ gall + hyper + strata(s), la, c("gall", "hyper"),
            at_beta = c(0.5, -0.25))
))

## ---------------------------------------------------------------- 2. infert (m_k = 1)
inf <- infert
inf$education <- as.character(inf$education)
inf <- inf[, c("stratum", "case", "spontaneous", "induced", "education", "age", "parity")]
write_fixture("infert", inf, list(
  summarise("infert_si", case ~ spontaneous + induced + strata(stratum), inf,
            c("spontaneous", "induced"), at_beta = c(1, 1)),
  summarise("infert_si_edu", case ~ spontaneous + induced + education + strata(stratum), inf,
            c("spontaneous", "induced", "education"),
            note = "education is constant within strata: R returns NA (aliased)")
))

## ---------------------------------------------------------------- 3. synthetic m_k > 1
set.seed(20240901)
K <- 60
sizes <- sample(1:14, K, replace = TRUE)
rows <- list()
for (k in seq_len(K)) {
  n <- sizes[k]
  x1 <- rnorm(n)
  x2 <- rbinom(n, 1, 0.4)
  x3 <- rnorm(n, mean = 50 + 10 * k, sd = 2)      # large stratum-level mean
  g <- sample(c("a", "b", "c"), n, replace = TRUE)
  eta <- 0.8 * x1 - 0.6 * x2 + 0.15 * x3 + ifelse(g == "b", 0.5, ifelse(g == "c", -0.4, 0))
  m <- if (k %% 13 == 0) 0L else if (k %% 17 == 0) n else sample(0:n, 1)
  if (n >= 2 && k %% 13 != 0 && k %% 17 != 0) m <- max(1L, min(n - 1L, m))
  rows[[k]] <- data.frame(id = sprintf("S%03d", k), y = draw_events(eta, m),
                          x1 = x1, x2 = x2, x3 = x3, g = g,
                          off = round(rnorm(n, sd = 0.5), 3), stringsAsFactors = FALSE)
}
syn <- do.call(rbind, rows)
syn <- syn[sample(nrow(syn)), ]
rownames(syn) <- NULL
syn$noty <- 1L - syn$y
write_fixture("synth_mk", syn, list(
  summarise("synth_x", y ~ x1 + x2 + x3 + strata(id), syn, c("x1", "x2", "x3"),
            at_beta = c(0.3, -0.2, 0.1)),
  summarise("synth_xg", y ~ x1 + x2 + x3 + g + strata(id), syn, c("x1", "x2", "x3", "g")),
  summarise("synth_x_event0", noty ~ x1 + x2 + x3 + strata(id), syn, c("x1", "x2", "x3"),
            note = "response 1 - y: equals SAS default ordering (models y = 0)"),
  summarise("synth_x_offset", y ~ x1 + x2 + x3 + offset(off) + strata(id), syn,
            c("x1", "x2", "x3"), at_beta = c(0.3, -0.2, 0.1))
))

## ---------------------------------------------------------------- 4. missing values
set.seed(7)
mis <- syn[, c("id", "y", "x1", "x2", "x3")]
mis$x1[sample(nrow(mis), 6)] <- NA
mis$y[sample(nrow(mis), 3)] <- NA
mis$id[sample(nrow(mis), 3)] <- NA
write_fixture("synth_missing", mis, list(
  summarise("synth_missing_x", y ~ x1 + x2 + x3 + strata(id), mis, c("x1", "x2", "x3"),
            note = "listwise deletion (R na.omit; SAS default without STRATA / MISSING)")
))

## ---------------------------------------------------------------- 5. large strata
set.seed(99)
ms <- c(60, 100, 140, 30, 170)
big <- do.call(rbind, lapply(seq_along(ms), function(k) {
  n <- 200
  x1 <- rnorm(n, sd = 2)
  x2 <- round(runif(n, -100, 100))
  eta <- 0.4 * x1 - 0.02 * x2
  data.frame(id = k, y = draw_events(eta, ms[k]), x1 = x1, x2 = x2)
}))
write_fixture("synth_big", big, list(
  summarise("synth_big_x", y ~ x1 + x2 + strata(id), big, c("x1", "x2"),
            at_beta = c(0.5, -0.05))
))

## ---------------------------------------------------------------- 6. FREQ (row replication)
set.seed(3)
fr <- do.call(rbind, lapply(1:25, function(k) {
  n <- sample(2:6, 1)
  x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.5)
  data.frame(id = k, y = draw_events(x1, sample(1:(n - 1), 1)), x1 = x1, x2 = x2,
             f = sample(1:3, n, replace = TRUE))
}))
expanded <- fr[rep(seq_len(nrow(fr)), fr$f), ]
fit_exp <- summarise("synth_freq_x", y ~ x1 + x2 + strata(id), expanded, c("x1", "x2"),
                     note = "fit on the row-expanded data (each row repeated f times)")
write_fixture("synth_freq", fr, list(fit_exp))

cat("fixtures written to", outdir, "\n")
