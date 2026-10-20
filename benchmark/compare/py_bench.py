# Times Python statsmodels ConditionalLogit on the scenarios in data/.
#
#   uv venv benchmark/compare/.venv && uv pip install --python benchmark/compare/.venv/bin/python statsmodels pandas
#   benchmark/compare/.venv/bin/python benchmark/compare/py_bench.py
#
# Default settings (BFGS, maxiter = 100; standard errors from a numerical Hessian).
# Each fit goes through the formula interface on a pandas DataFrame already in memory,
# which includes grouping. The reported time is the median of REPS fits (a single fit
# when one takes longer than SLOW seconds). Output: results/statsmodels.csv and
# results/statsmodels_coef.csv.

import os, sys, time, statistics, warnings
import pandas as pd
import statsmodels
from statsmodels.discrete.conditional_models import ConditionalLogit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
SCENARIOS = ["pairs", "match1to4", "choice", "multi", "panelfe", "bigstrata", "choicelarge"]
REPS, SLOW = 3, 30.0
os.makedirs(RES, exist_ok=True)
warnings.simplefilter("ignore")    # dropped-group notices on panelfe


def fit(df, formula):
    return ConditionalLogit.from_formula(formula, groups="s", data=df).fit(disp=0)


rows, coefs = [], []
for sc in SCENARIOS:
    df = pd.read_csv(os.path.join(DATA, sc + ".csv"))
    xs = [c for c in df.columns if c.startswith("x")]
    formula = "y ~ 0 + " + " + ".join(xs)
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        r = fit(df, formula)
        times.append(time.perf_counter() - t0)
        if times[-1] > SLOW:
            break
    t = statistics.median(times)
    print(f"{sc:<12} {len(df):9d} rows  {t:8.4f} s  ({len(times)} fits)", flush=True)
    rows.append(dict(scenario=sc, tool="Python statsmodels", threads=1, rows=len(df),
                     seconds=t, iterations=-1, loglik=r.llf))
    coefs += [dict(scenario=sc, term=k, estimate=v) for k, v in r.params.items()]

pd.DataFrame(rows).to_csv(os.path.join(RES, "statsmodels.csv"), index=False, float_format="%.17g")
pd.DataFrame(coefs).to_csv(os.path.join(RES, "statsmodels_coef.csv"), index=False, float_format="%.17g")
print("Python", sys.version.split()[0], "statsmodels", statsmodels.__version__)
