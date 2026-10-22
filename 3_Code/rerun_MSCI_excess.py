"""
Section 5.3 re-run: International Shares proxied by MSCI World ex-Australia
(EAGREXAG Index, AUD, gross) in place of the Vanguard unit-price series.
==========================================================================

Only the data layer changes. Every test is imported unchanged from
replicate_all_CORRECTED.py / spanning_K7.py, so the mathematics is identical
to the already-validated implementation.

Outputs
-------
  MSCIx_coverage.csv         series availability, what binds the joint start
  MSCIx_descriptives.csv     Table 8
  MSCIx_downside.csv         Table 9 (own-sample, as in the paper)
  MSCIx_downside_matched.csv
  MSCIx_correlations.csv     matched-sample correlation matrix
  MSCIx_spanning.csv         Table 10, K = 7 and K = 6 panels
  MSCIx_dnw.csv              Table 10a, De Roon-Nijman-Werker, K = 6
  MSCIx_hac_sensitivity.csv  Table 11a
  MSCIx_ljungbox.csv         residual diagnostics behind Section 5.4
  MSCIx_RESULTS.txt          formatted log
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

import replicate_all_CORRECTED as R
import spanning_K7 as S

# ---- the only substantive change: swap the International Shares proxy -----
NEW_INTL = "MSCI_WorldIndex_ExAusAUD.xlsx"
S.BENCH_FILES = [(c, (NEW_INTL if c == "International Shares" else f), role)
                 for c, f, role in S.BENCH_FILES]

BENCH_K7 = list(S.BENCH)
BENCH_K6 = [c for c in BENCH_K7 if c != "International Property"]
RF_COL = S.RF_COL
CANDS = ["Gold", "Bitcoin"]
HAC_MAIN = 6
HAC_LAGS = [3, 4, 5, 6, 8, 12]
CAP = 0.10
DNW_NSIM = 40000
DNW_SEED = 11


def _st(p):
    return ("***" if p < .001 else "**" if p < .01 else "*" if p < .05
            else "." if p < .10 else "")


# ------------------------------------------------------------ tables ------

def descriptives(M):
    rows = []
    for c in BENCH_K7 + CANDS + [RF_COL]:
        r = M[c].dropna()
        jb, jbp = stats.jarque_bera(r.values)
        rows.append({
            "Asset": c, "First": str(r.index.min()), "Last": str(r.index.max()),
            "N": int(len(r)),
            "Mean mth %": round(r.mean() * 100, 2),
            "SD mth %": round(r.std(ddof=1) * 100, 2),
            "Mean ann %": round(r.mean() * 1200, 2),
            "SD ann %": round(r.std(ddof=1) * np.sqrt(12) * 100, 2),
            "Min %": round(r.min() * 100, 2), "Max %": round(r.max() * 100, 2),
            "Skew": round(float(stats.skew(r.values, bias=False)), 2),
            "Excess kurt": round(float(stats.kurtosis(r.values, bias=False)), 2),
            "JB": round(float(jb), 2), "JB p": round(float(jbp), 4)})
    return pd.DataFrame(rows)


def downside(M, ref="Australian Shares"):
    """Co-movement with Australian Shares over each series' own sample."""
    rows = []
    for c in CANDS + [b for b in BENCH_K7 if b != ref]:
        d = M[[ref, c]].dropna()
        row = {"Asset": c}
        for lab, cut in [("Whole", None), ("<=-2%", -0.02), ("<=-5%", -0.05)]:
            sub = d if cut is None else d[d[ref] <= cut]
            x, y = sub[ref].values, sub[c].values
            row[lab + " N"] = int(len(sub))
            row[lab + " r"] = round(float(np.corrcoef(x, y)[0, 1]), 2)
            row[lab + " b"] = round(float(np.polyfit(x, y, 1)[0]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def spanning_row(label, sdf, cand, bench, with_dnw=False):
    X = sdf[bench].values.astype(float)
    y = sdf[cand].values.astype(float)
    rf = sdf[RF_COL].values.astype(float)

    Xe, ye = X - rf[:, None], y - rf
    hk = R.hk_test(Xe, ye)
    kz = R.kz_test(Xe, ye)
    ell = R.gmm_wald_elliptical(Xe, ye)
    hac = R.hac_spanning(Xe, ye, HAC_MAIN)
    a1, t1, p_one = R.dnw_one_sided(Xe, ye)

    EXb = sdf[bench].values - rf[:, None]
    EXp = sdf[list(bench) + [cand]].values - rf[:, None]
    sr_b, _ = R.capped_tangency_sharpe(EXb, None)
    sr_u, w_u = R.capped_tangency_sharpe(EXp, None)
    sr_c, w_c = R.capped_tangency_sharpe(EXp, CAP)

    row = {
        "Sample": label, "Asset": cand, "K": len(bench), "T": int(hk["T"]),
        "First": str(sdf.index.min()), "Last": str(sdf.index.max()),
        "alpha_mth_%": round(float(kz["alpha_hat"][0]) * 100, 3),
        "alpha_ann_%": round(float(kz["alpha_hat"][0]) * 1200, 2),
        "delta": round(float(kz["delta_hat"][0]), 4),
        "HK_F": round(float(hk["F"]), 4), "HK_df1": hk["df1"], "HK_df2": hk["df2"],
        "HK_p": round(float(hk["p_value"]), 4),
        "KZ_W_p": round(float(kz["p_W"]), 4),
        "stepdown_F1_alpha": round(float(kz["F_Z"]), 4),
        "stepdown_F1_p": round(float(kz["p_Z"]), 4),
        "stepdown_F2_delta": round(float(kz["F_K"]), 4),
        "stepdown_F2_p": round(float(kz["p_K"]), 4),
        "GMM_ell_Wa_e": round(ell["Wa_e"], 4), "GMM_ell_p": round(ell["p"], 4),
        "HAC6_joint_p": round(hac["joint_p"], 4),
        "HAC6_tangency_p": round(hac["tangency_p_two"], 4),
        "HAC6_GMV_p": round(hac["gmv_p"], 4),
        "HAC6_onesided_p": round(hac["tangency_p_one"], 4),
        "HAC6_alpha_t": round(hac["alpha_t"], 3),
        "onesided_t_alpha": round(t1, 3), "onesided_p": round(p_one, 4),
        "LO_Sharpe_bench": round(sr_b, 4), "LO_Sharpe_plus": round(sr_u, 4),
        "LO_weight_%": round(float(w_u[-1]) * 100, 2),
        "LO_Sharpe_capped": round(sr_c, 4),
        "LO_weight_capped_%": round(float(w_c[-1]) * 100, 2),
        "dSharpe_uncapped": round(sr_u - sr_b, 4),
        "dSharpe_capped": round(sr_c - sr_b, 4),
    }
    if with_dnw:
        # constrained test on excess returns over BAUBIL, evaluated at the
        # risk-free rate (BAUBIL is available to members as a risk-free asset)
        dnw_new = R.dnw_spanning_rf(Xe, ye, constrain_benchmark=False)
        dnw_all = R.dnw_spanning_rf(Xe, ye, constrain_benchmark=True)
        row.update({
            "DNW_nssNew_stat": round(dnw_new["stat"], 4),
            "DNW_nssNew_p": round(dnw_new["p"], 4),
            "DNW_nssNew_restrictions": dnw_new["n_restrictions"],
            "DNW_nssAll_stat": round(dnw_all["stat"], 4),
            "DNW_nssAll_p": round(dnw_all["p"], 4),
            "DNW_nssAll_bench_held": "; ".join(bench[i] for i in dnw_all["bench_held"]),
            "DNW_nssAll_restrictions": dnw_all["n_restrictions"],
            "DNW_v_lo": round(dnw_all["v_lo"], 6),
            "DNW_v_hi": round(dnw_all["v_hi"], 6)})
    return row


def hac_grid(sdf, cand, bench, label):
    rf = sdf[RF_COL].values.astype(float)
    X = sdf[bench].values.astype(float) - rf[:, None]
    y = sdf[cand].values.astype(float) - rf
    out = {"Regression": label, "T": len(sdf),
           "Classical": round(float(R.hk_test(X, y)["p_value"]), 4)}
    for L in HAC_LAGS:
        h = R.hac_spanning(X, y, L)
        out["L=%d joint" % L] = round(h["joint_p"], 4)
        out["L=%d tang" % L] = round(h["tangency_p_two"], 4)
    return out


def ljung(sdf, cand, bench, label):
    X = sdf[bench].values.astype(float)
    y = sdf[cand].values.astype(float)
    Z = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    e = y - Z @ b
    lb = acorr_ljungbox(e, lags=[12], return_df=True)
    ac = [float(np.corrcoef(e[k:], e[:-k])[0, 1]) for k in range(1, 13)]
    band = 2.0 / np.sqrt(len(e))
    outside = ", ".join("lag%d:%+.2f" % (k + 1, a)
                        for k, a in enumerate(ac) if abs(a) > band)
    out = {"Regression": label, "T": len(e),
           "LB(12)": round(float(lb["lb_stat"].iloc[0]), 2),
           "LB(12) p": round(float(lb["lb_pvalue"].iloc[0]), 4),
           "band +/-": round(band, 3),
           "outside band": outside or "none"}
    for k, a in enumerate(ac, 1):
        out["ac%d" % k] = round(a, 3)
    return out


# ------------------------------------------------------------- driver -----

def main():
    M = S.build_panel()

    cov = S.coverage(M)
    cov.to_csv("MSCIx_coverage.csv", index=False)

    desc = descriptives(M)
    desc.to_csv("MSCIx_descriptives.csv", index=False)

    dn_own = downside(M)
    dn_own.to_csv("MSCIx_downside.csv", index=False)

    m7 = M.dropna(subset=BENCH_K7 + [RF_COL] + CANDS)
    downside(m7).to_csv("MSCIx_downside_matched.csv", index=False)
    corr = m7[BENCH_K7 + CANDS].corr().round(3)
    corr.to_csv("MSCIx_correlations.csv")

    specs = []
    for tag, bench in [("K7", BENCH_K7), ("K6", BENCH_K6)]:
        matched = M.dropna(subset=list(bench) + [RF_COL] + CANDS)
        specs += [
            (tag, bench, tag + " Gold, longest",
             S.sample_for(M, "Gold", bench), "Gold"),
            (tag, bench, tag + " Gold, matched", matched, "Gold"),
            (tag, bench, tag + " Bitcoin",
             S.sample_for(M, "Bitcoin", bench), "Bitcoin"),
        ]

    rows, hacs, lbs = [], [], []
    for tag, bench, lab, sdf, cand in specs:
        r = spanning_row(lab, sdf, cand, bench, with_dnw=(tag == "K6"))
        r["Panel"] = tag
        rows.append(r)
        hacs.append(hac_grid(sdf, cand, bench, lab))
        lbs.append(ljung(sdf, cand, bench, lab))

    span = pd.DataFrame(rows)
    front = ["Panel", "Sample", "Asset", "K", "T", "First", "Last"]
    span = span[front + [c for c in span.columns if c not in front]]
    span.to_csv("MSCIx_spanning.csv", index=False)

    dnw = span[span.Panel == "K6"][
        ["Sample", "Asset", "T", "DNW_nssNew_stat", "DNW_nssNew_p",
         "DNW_nssAll_stat", "DNW_nssAll_p", "DNW_nssAll_bench_held"]]
    dnw.to_csv("MSCIx_dnw.csv", index=False)

    hacdf = pd.DataFrame(hacs)
    hacdf.to_csv("MSCIx_hac_sensitivity.csv", index=False)
    lbdf = pd.DataFrame(lbs)
    lbdf.to_csv("MSCIx_ljungbox.csv", index=False)

    # -------------------------------- report --------------------------------
    pd.set_option("display.width", 260, "display.max_columns", 300)
    W = 104
    L = []

    def P(s=""):
        print(s)
        L.append(s)

    P("=" * W)
    P("SECTION 5.3 RE-RUN - International Shares = MSCI World ex-Australia (EAGREXAG, AUD)")
    P("=" * W)
    for c, f, role in S.BENCH_FILES:
        P("   %-26s %-34s %s" % (c, f, role))
    P("   %-26s %-34s %s" % ("Cash (risk-free)", S.CASH_FILE,
                             "Bloomberg AusBond Bank Bill"))
    P("")
    P("SERIES COVERAGE")
    P("-" * W)
    P(cov.to_string(index=False))
    P("")
    P("TABLE 8 - DESCRIPTIVE STATISTICS (own sample)")
    P("-" * W)
    P(desc.to_string(index=False))
    P("")
    P("TABLE 9 - CO-MOVEMENT WITH AUSTRALIAN SHARES (own sample)")
    P("-" * W)
    P(dn_own.to_string(index=False))
    P("")
    P("CORRELATION MATRIX (K=7 matched sample)")
    P("-" * W)
    P(corr.to_string())
    P("")
    P("TABLE 10 - SPANNING")
    P("-" * W)
    for _, r in span.iterrows():
        P("")
        P("-- %-8s | %-20s | %s..%s | K=%d T=%d"
          % (r["Asset"], r["Sample"], r["First"], r["Last"], int(r["K"]), int(r["T"])))
        P("   alpha = %.3f%%/mth (%.2f%% pa)   delta = %+.4f"
          % (r["alpha_mth_%"], r["alpha_ann_%"], r["delta"]))
        hp, ha = r["HK_p"], r["HAC6_joint_p"]
        P("   HK exact F = %.3f df(%d,%d)  p = %.4f%s   |   HAC(6) joint p = %.4f%s"
          % (r["HK_F"], r["HK_df1"], r["HK_df2"], hp, _st(hp), ha, _st(ha)))
        P("   step-down F1 alpha = %.3f p = %.4f%s   |   F2 delta|alpha = %.3f p = %.4f%s"
          % (r["stepdown_F1_alpha"], r["stepdown_F1_p"], _st(r["stepdown_F1_p"]),
             r["stepdown_F2_delta"], r["stepdown_F2_p"], _st(r["stepdown_F2_p"])))
        P("   long-only tangency Sharpe %.3f -> %.3f (w = %.1f%%)  |  capped 10%%: %.3f"
          % (r["LO_Sharpe_bench"], r["LO_Sharpe_plus"], r["LO_weight_%"],
             r["LO_Sharpe_capped"]))
        if r["Panel"] == "K6":
            P("   DNW no-short-sale: candidate only xi = %.3f p = %.4f%s   |   "
              "all assets xi = %.3f p = %.4f%s   held: %s"
              % (r["DNW_nssNew_stat"], r["DNW_nssNew_p"], _st(r["DNW_nssNew_p"]),
                 r["DNW_nssAll_stat"], r["DNW_nssAll_p"], _st(r["DNW_nssAll_p"]),
                 r["DNW_nssAll_bench_held"]))
    P("")
    P("TABLE 11a - HAC BANDWIDTH SENSITIVITY")
    P("-" * W)
    P(hacdf.to_string(index=False))
    P("")
    P("RESIDUAL DIAGNOSTICS (Section 5.4)")
    P("-" * W)
    P(lbdf[["Regression", "T", "LB(12)", "LB(12) p", "band +/-",
            "outside band"]].to_string(index=False))
    P("")
    P("Significance: *** p<.001  ** p<.01  * p<.05  . p<.10")

    with open("MSCIx_RESULTS.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
