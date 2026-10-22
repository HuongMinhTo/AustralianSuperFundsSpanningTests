# -*- coding: utf-8 -*-
"""Table 2: descriptive statistics (monthly and annualised) of the DIY-Mix options,
gold, Bitcoin and BAUBIL, with bias-adjusted skewness and excess kurtosis.
Output: Table2_recomputed.csv"""
import numpy as np, pandas as pd
from scipy import stats
import replicate_all_CORRECTED as R
f=R.build_funds()
def d(s,name):
    s=s.dropna()*100
    return dict(Asset=name,First=str(s.index.min()),Last=str(s.index.max()),N=len(s),Mean=s.mean(),SD=s.std(ddof=1),Min=s.min(),Max=s.max(),
        MeanAnn=s.mean()*12,SDAnn=s.std(ddof=1)*np.sqrt(12),
        Skew_biased=stats.skew(s),Skew_adj=stats.skew(s,bias=False),
        ExK_biased=stats.kurtosis(s),ExK_adj=stats.kurtosis(s,bias=False),
        JB=stats.jarque_bera(s).statistic,JBp=stats.jarque_bera(s).pvalue)
rows=[]
for fund in ['AustralianSuper','ART']:
    M=f[fund]
    for c in R.BENCH: rows.append(dict(Panel=fund,**d(M[c],c)))
M=f['AustralianSuper']
for c in ['Gold','Bitcoin','Cash']: rows.append(dict(Panel='Candidates/rf',**d(M[c],c)))
df=pd.DataFrame(rows); df.to_csv('Table2_recomputed.csv',index=False)
pd.set_option('display.width',250)
print(df.round(2).to_string(index=False))
# correlations matched window
for fund in ['AustralianSuper','ART']:
    M=f[fund]; m=M[R.BENCH+['Gold','Bitcoin']].dropna()
    print(fund, len(m), m.index.min(), m.index.max()); print(m.corr().round(2).to_string())
