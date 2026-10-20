# Downloads the Stata Press example datasets used in the Stata [R] clogit manual entry
# and writes them to validation/data/ (not committed) as .dta (for stata_manual.do) and
# CSV (for stata_manual.jl). Needs R with `haven`.
#
#   Rscript validation/stata_manual.R
#   julia --project=docs validation/stata_manual.jl

suppressMessages(library(haven))
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
out <- file.path(here, "data"); dir.create(out, showWarnings = FALSE)
for (nm in c("clogitid", "lowbirth2", "union")) {
  url <- paste0("https://www.stata-press.com/data/r19/", nm, ".dta")
  download.file(url, file.path(out, paste0(nm, ".dta")), mode = "wb", quiet = TRUE)
  df <- zap_labels(read_dta(file.path(out, paste0(nm, ".dta"))))
  write.csv(as.data.frame(df), file.path(out, paste0(nm, ".csv")), row.names = FALSE)
  cat(nm, nrow(df), "rows\n")
}
