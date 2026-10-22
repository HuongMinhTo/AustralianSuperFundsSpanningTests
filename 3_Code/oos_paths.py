# -*- coding: utf-8 -*-
"""Out-of-sample backtests for both funds and both candidates (60-month expanding
window, lambda = 1-10, five strategies); stores the month-by-month portfolio
weights. Output: oos_paths.pkl (input to tableB3_weights.py)"""
import numpy as np, pandas as pd, pickle
import replicate_all_CORRECTED as R
funds=R.build_funds()
out={}
for fund,M in funds.items():
    matched=M[M["Gold"].notna()&M["Bitcoin"].notna()].dropna(subset=R.BENCH+[R.RF_COL,"Gold","Bitcoin"]).reset_index(drop=True)
    print(fund, len(matched))
    for cand in ["Gold","Bitcoin"]:
        res,ws,rf=R.run_backtest(matched[R.BENCH+[R.RF_COL,cand]].copy(),cand)
        out[(fund,cand)]=(res,ws,rf)
pickle.dump(out,open('oos_paths.pkl','wb'))
