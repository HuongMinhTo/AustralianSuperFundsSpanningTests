# -*- coding: utf-8 -*-
"""Appendix Table B3: month-by-month out-of-sample weights on gold and Bitcoin
(mean, median, % of months at the lower bound) at each lambda. Also checks that
oos_paths.pkl reproduces OOS_results_ALL.csv. Output: TableB3_OOS_weights_full.csv"""
import numpy as np, pandas as pd, pickle
import replicate_all_CORRECTED as R
out=pickle.load(open('oos_paths.pkl','rb'))
old=pd.read_csv('OOS_results_ALL.csv'); oldw=pd.read_csv('OOS_mean_weights_ALL.csv')
# reproducibility
maxd=0; maxw=0
for fund in ['AustralianSuper','ART']:
    resg,wg,rf=out[(fund,'Gold')]; resb,wb,_=out[(fund,'Bitcoin')]
    for strat in R.STRATEGIES:
        for mname,fn in R.METRICS:
            row=old[(old.Fund==fund)&(old.Method==strat)&(old.Metric==mname)].iloc[0]
            for lam in R.LAMBDAS:
                for col,v in ((f"l={lam} Bench",fn(resg[(strat,lam,'bench')],rf)),(f"l={lam} +Gold",fn(resg[(strat,lam,'plus')],rf)),(f"l={lam} +BTC",fn(resb[(strat,lam,'plus')],rf))):
                    maxd=max(maxd,abs(round(v,3)-row[col]))
        for cand,ws in (('Gold',wg),('Bitcoin',wb)):
            r=oldw[(oldw.Fund==fund)&(oldw.Candidate==cand)&(oldw.Strategy==strat)].iloc[0]
            for lam in R.LAMBDAS: maxw=max(maxw,abs(round(ws[(strat,lam,'plus')][:,3].mean(),4)-r[f"l={lam}"]))
print('max diff ratios',maxd,'max diff mean weights',maxw)
rows=[]
for fund in ['AustralianSuper','ART']:
    for cand in ['Gold','Bitcoin']:
        _,ws,_=out[(fund,cand)]
        for strat in R.STRATEGIES:
            floor=0.125 if 'Gens' in strat else 0.0
            for lam in R.LAMBDAS:
                w=ws[(strat,lam,'plus')][:,3]
                W=ws[(strat,lam,'plus')]
                rows.append(dict(Fund=fund,Candidate=cand,Strategy=strat,lam=lam,T=len(w),
                    mean=w.mean()*100,median=np.median(w)*100,sd=w.std(ddof=1)*100,min=w.min()*100,max=w.max()*100,
                    pct_zero=(w<floor+1e-4).mean()*100,  # at lower bound (0 or Gens floor)
                    pct_all_in=(W.max(axis=1)>0.999-3*floor).mean()*100 if True else 0,
                    pct_cand_max=(w>=W.max(axis=1)-1e-9).mean()*100,
                    first=w[:12].mean()*100,last=w[-12:].mean()*100))
df=pd.DataFrame(rows); df.to_csv('TableB3_OOS_weights_full.csv',index=False)
pd.set_option('display.width',250)
for lam in [1,5,10]:
    print('\nlambda',lam); print(df[df.lam==lam][['Fund','Candidate','Strategy','mean','median','pct_zero','min','max','pct_cand_max']].round(1).to_string(index=False))
