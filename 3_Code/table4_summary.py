# -*- coding: utf-8 -*-
"""Table 4: out-of-sample summary across risk-aversion levels (lambda = 1-10).
For each fund, strategy and candidate: the average change in the annualised
Sharpe and Sortino ratios and in the Omega ratio from adding the candidate
(candidate minus benchmark, averaged over lambda = 1..10), the number of lambda
values at which the candidate raises the Sharpe ratio, and the lowest lambda
from which it raises the Sharpe ratio at that and every higher lambda.
Computed from the unrounded month-by-month OOS returns in oos_paths.pkl.
Outputs: Table4_summary.csv, oos_summary.json"""

import json, pickle
import numpy as np
import pandas as pd

import replicate_all_CORRECTED as R

out = pickle.load(open("oos_paths.pkl", "rb"))
M = dict(R.METRICS)
NAMES = {"Bayes-Stein": "Bayes–Stein", "Bayes-Stein with Gens": "Bayes–Stein with Gens",
         "Black-Litterman": "Black–Litterman"}
rows, js = [], {}
for fund, fk in (("AustralianSuper", "AS"), ("ART", "ART")):
    for strat in R.STRATEGIES:
        for cand, ck in (("Gold", "+Gold"), ("Bitcoin", "+BTC")):
            res, ws, rf = out[(fund, cand)]
            rec = {}
            for met, short in (("Sharpe Ratio", "Sharpe"), ("Sortino Ratio", "Sortino"), ("Omega Ratio", "Omega")):
                d = np.array([M[met](res[(strat, l, "plus")], rf) - M[met](res[(strat, l, "bench")], rf)
                              for l in R.LAMBDAS])
                lam0 = next((L for L in R.LAMBDAS if (d[R.LAMBDAS.index(L):] > 0).all()), None)
                rec[short] = dict(avg=float(d.mean()), wins=int((d > 0).sum()), lam0=lam0, d=[float(x) for x in d])
            js["%s|%s|%s" % (fk, NAMES.get(strat, strat), ck)] = rec
            rows.append(dict(Fund=fund, Strategy=strat, Candidate=cand,
                             dSharpe=round(rec["Sharpe"]["avg"], 3), dSortino=round(rec["Sortino"]["avg"], 3),
                             dOmega=round(rec["Omega"]["avg"], 3), Sharpe_wins=rec["Sharpe"]["wins"],
                             Gain_from_lambda=rec["Sharpe"]["lam0"]))
df = pd.DataFrame(rows)
df.to_csv("Table4_summary.csv", index=False)
json.dump(js, open("oos_summary.json", "w"), ensure_ascii=False, indent=1)
print(df.to_string(index=False))
