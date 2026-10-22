import numpy as np, pandas as pd
import replicate_all_CORRECTED as R
R.HAC_LAGS=[3,4,5,6,8,12]
f=R.build_funds()
rows=[];sens=[]
for fund,M in f.items():
    matched=M[M["Gold"].notna()&M["Bitcoin"].notna()]
    for cand in ['Gold','Bitcoin']:
        rx=R._kz_row(fund,'Matched',cand,matched,True)
        # all tests, including the constrained (DNW) test, use excess returns over BAUBIL
        rows.append(rx)
        X,y,_,_=R._prep(matched,cand,True)
        for L in R.HAC_LAGS:
            h=R.hac_spanning(X,y,L); sens.append(dict(Fund=fund,Asset=cand,lag=L,joint_p=round(h['joint_p'],4),tang_p=round(h['tangency_p_two'],4)))
T=pd.DataFrame(rows); S=pd.DataFrame(sens)
T.to_csv('KZ_spanning_EXCESS.csv',index=False); S.to_csv('HAC_sensitivity_EXCESS.csv',index=False)
cols=['Fund','Asset','T','alpha_ann_%','delta','HK_joint_F','HK_joint_p','Wald_p','stepdown_F1_alpha','tangency_p_classical','stepdown_F2_delta_given_alpha','GMV_p_classical','HAC6_joint_p','GMM_elliptical_p','kurtosis_kappa','DNW_nssNew_p','DNW_nssAll_p','LO_Sharpe_bench','LO_Sharpe_plus','cand_weight_%']
pd.set_option('display.width',300); print(T[cols].T.to_string())
print(S.pivot_table(index=['Fund','Asset'],columns='lag',values='joint_p'))
