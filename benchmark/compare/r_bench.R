# Times R survival::clogit(method = "exact") on the scenarios in data/.
#
#   Rscript benchmark/compare/r_bench.R
#
# Default settings (coxph.control: eps = 1e-9, iter.max = 20). The reported time is the
# median of REPS fits (data already in memory). Output: results/r.csv and
# results/r_coef.csv.

suppressMessages({ library(survival); library(data.table) })
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
datadir <- file.path(here, "data"); resdir <- file.path(here, "results")
dir.create(resdir, showWarnings = FALSE)
REPS <- 3
scenarios <- c("pairs", "match1to4", "choice", "multi", "panelfe", "bigstrata", "choicelarge")

rows <- list(); coefs <- list()
for (sc in scenarios) {
  df <- as.data.frame(fread(file.path(datadir, paste0(sc, ".csv"))))
  xs <- grep("^x", names(df), value = TRUE)
  f <- as.formula(paste("y ~", paste(xs, collapse = " + "), "+ strata(s)"))
  times <- numeric(REPS)
  for (r in seq_len(REPS)) {
    times[r] <- system.time(fit <- clogit(f, data = df, method = "exact"))[["elapsed"]]
  }
  t <- median(times)
  cat(sprintf("%-12s %9d rows  %8.4f s  (%d iterations)\n", sc, nrow(df), t, fit$iter))
  rows[[sc]] <- data.frame(scenario = sc, tool = "R survival::clogit", threads = 1,
                           rows = nrow(df), seconds = t, iterations = fit$iter,
                           loglik = fit$loglik[2])
  coefs[[sc]] <- data.frame(scenario = sc, term = names(coef(fit)), estimate = unname(coef(fit)))
}
fwrite(do.call(rbind, rows), file.path(resdir, "r.csv"))
fwrite(do.call(rbind, coefs), file.path(resdir, "r_coef.csv"))
cat(R.version.string, " survival", as.character(packageVersion("survival")), "\n")
