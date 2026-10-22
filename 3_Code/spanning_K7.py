"""
K = 7 mean-variance spanning tests: Gold and Bitcoin vs a broad Australian
asset-class benchmark
=========================================================================

Follows the methodology of "Gold_Bitcoin_AusSuperBalancedFund_13082026.docx"
(the K = 7 robustness analysis), applied to the benchmark set requested by
the user and to the LONGEST time series each source file supports.

Benchmark (K = 7), one candidate added at a time (N = 1):

    ASA52      Australian shares            (S&P/ASX 200 accumulation)
    VANINSH    International shares         (Vanguard Intl Shares Index, AUD)
    ASA6PROP   Australian listed property   (S&P/ASX 200 A-REIT accumulation)
    RAHRSAH    International listed property (hedged AUD)
    FDCICSAH   Listed infrastructure        (hedged AUD)
    BACMO      Australian fixed interest    (Bloomberg AusBond Composite)
    LEGATRAH   Global fixed interest        (Bloomberg Global Agg, hedged AUD)

Test assets (separately) : Gold (GOLDAX_split_adjusted), Bitcoin (BTCAUD)
Risk-free / cash        : BAUBIL (Bloomberg AusBond Bank Bill Index), held
                          OUTSIDE the K = 7 risky benchmark, exactly as in
                          the source paper.

NOTE ON TWO TICKERS. The request named "VANANISH" and "RAHRSAP"; the files
that exist are VANINSH.xlsx (Vanguard International Shares Index) and
RAHRSAH.xlsx, which are also the series the source paper describes for
those two roles. They are used here. The source paper used MSCIACWI for
international shares, not VANINSH, so results are not expected to
reproduce that paper exactly - see the validation block in the output.

Estimation is OLS. Inference, following the paper:
  PRIMARY      Huberman-Kandel (1987) exact finite-sample F test, in the
               Kan-Zhou (2012) eq. (28) form that is correct for N = 1;
               and HAC(6) Newey-West robust joint inference.
  SUPPLEMENTARY (reported alongside, not a change of method)
               Kan-Zhou step-down F1 (alpha) / F2 (delta | alpha), the GMM
               elliptical Wald, and the one-sided no-short-sale test.

All spanning / HAC / step-down functions are imported from
replicate_all_CORRECTED.py so the mathematics is identical to the already
validated implementation; only the data layer and the drivers are new.

Outputs
-------
  K7_descriptives.csv        Table 3 equivalent, per-series own sample
  K7_coverage.csv            what binds the joint sample start date
  K7_downside_corr.csv       co-movement with Australian shares
  K7_spanning.csv            the spanning results
  K7_correlations.csv        full correlation matrix, matched sample
  K7_results_log.txt         formatted console output
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", message="Could not infer format")

import replicate_all_CORRECTED as R   # validated spanning / HAC machinery

DATA_DIR = os.environ.get("DATA_DIR", ".")

# ---------------------------------------------------------------- config ---

BENCH_FILES = [                       # (column name, file, role)
    ("Australian Shares",      "ASA52.xlsx",    "Australian shares"),
    ("International Shares",   "VANINSH.xlsx",  "International shares"),
    ("Australian Property",    "ASA6PROP.xlsx", "Australian listed property"),
    ("International Property", "RAHRSAH.xlsx",  "International listed property"),
    ("Infrastructure",         "FDCICSAH.xlsx", "Listed infrastructure"),
    ("Australian Fixed Interest", "BACMO.xlsx", "Australian fixed interest"),
    ("Global Fixed Interest",  "LEGATRAH.xlsx", "Global fixed interest"),
]
BENCH = [c for c, _, _ in BENCH_FILES]

CASH_FILE = "BAUBIL.xlsx"
RF_COL    = "Cash"
GOLD_FILE = "GOLDAX_split_adjusted.csv"
BTC_FILE  = "BTCAUD.csv"
CANDIDATES = ["Gold", "Bitcoin"]

SAMPLE_END     = "2026-07"     # last complete month in every source file
HAC_LAG        = 6             # primary HAC lag, as in the paper
HAC_SENS_LAGS  = [4, 5, 12]
DOWNSIDE_CUTS  = [-0.02, -0.05]

# Robustness: the source paper proxied international shares with MSCIACWI.
# If that file is reachable it is used for a like-for-like comparison against
# the VANINSH series requested here. Set to None to skip.
ACWI_FILE = os.path.join(DATA_DIR, "MSCIACWI.xlsx")   # optional; skipped if absent
PAPER_END = "2026-02"          # the source paper's last month
DIST_YIELDS = [0.02, 0.025, 0.03]   # distribution add-back sensitivity

# ------------------------------------------------------------ data layer ---

def _p(name):
    return os.path.join(DATA_DIR, name)


def _level_xlsx_monthly(path, name):
    """Read a Bloomberg-style monthly level sheet and return monthly returns.

    Handles both layouts present in the source files:
      (a) row 0 = ["Security", "<TICKER> Index"], data from row 1;
      (b) a metadata block (Security / Start Date / End Date / Period /
          Currency), a blank row, a ["Date", "PX_LAST"] header, then data.
    Both are parsed by locating the first row whose first cell is a date.
    """
    raw = pd.ExcelFile(path).parse(0, header=None)
    d = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    v = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    ok = d.notna() & v.notna()
    if not ok.any():
        raise ValueError(f"{path}: no (date, level) rows found")
    s = (pd.Series(v[ok].values, index=d[ok].values)
           .sort_index().resample("ME").last())
    r = s.pct_change(fill_method=None).rename(name)
    if r.dropna().empty:
        raise ValueError(f"{path}: no returns produced")
    return r


def _cash_monthly(path):
    r = _level_xlsx_monthly(path, RF_COL)
    chk = r.dropna()
    if not ((chk > -0.005) & (chk < 0.02)).all():
        raise ValueError(f"{path}: implausible monthly cash returns")
    return r


def build_panel():
    """Monthly return panel: K = 7 benchmark, Cash (BAUBIL), Gold, Bitcoin."""
    cols = [_level_xlsx_monthly(_p(f), c) for c, f, _ in BENCH_FILES]
    cols.append(_cash_monthly(_p(CASH_FILE)))
    cols.append(R._price_monthly(_p(GOLD_FILE), "Gold", dayfirst=True))
    cols.append(R._price_monthly(_p(BTC_FILE),  "Bitcoin", dayfirst=False))
    M = pd.concat(cols, axis=1, sort=True)
    M.index = M.index.to_period("M")
    if SAMPLE_END:
        M = M[M.index <= pd.Period(SAMPLE_END, "M")]
    return M


def coverage(M):
    """Per-series availability - shows which series binds the joint start."""
    rows = []
    for c in BENCH + [RF_COL] + CANDIDATES:
        s = M[c].dropna()
        rows.append({"Series": c, "First": str(s.index.min()),
                     "Last": str(s.index.max()), "N": int(len(s))})
    return pd.DataFrame(rows)


def sample_for(M, cand, bench=None):
    """Largest sample with every benchmark asset, Cash and the candidate."""
    bench = BENCH if bench is None else bench
    return M.dropna(subset=list(bench) + [RF_COL, cand])

# ------------------------------------------------------- descriptive block --

def descriptives(M):
    rows = []
    for c in BENCH + CANDIDATES + [RF_COL]:
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
            "JB": round(float(jb), 2),
            "JB p": round(float(jbp), 4),
        })
    return pd.DataFrame(rows)


def downside_corr(M, ref="Australian Shares"):
    """Co-movement with Australian shares, whole sample and equity downturns."""
    rows = []
    for c in CANDIDATES + [b for b in BENCH if b != ref]:
        d = M[[ref, c]].dropna()
        row = {"Asset": c}
        for lab, cut in [("Whole", None)] + [(f"<= {int(k*100)}%", k)
                                             for k in DOWNSIDE_CUTS]:
            sub = d if cut is None else d[d[ref] <= cut]
            if len(sub) < 3:
                row[f"{lab} N"] = len(sub)
                row[f"{lab} r"] = np.nan; row[f"{lab} beta"] = np.nan
                continue
            x = sub[ref].values; y = sub[c].values
            row[f"{lab} N"] = int(len(sub))
            row[f"{lab} r"] = round(float(np.corrcoef(x, y)[0, 1]), 2)
            row[f"{lab} beta"] = round(float(np.polyfit(x, y, 1)[0]), 2)
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------- spanning block --

def spanning_row(label, sdf, cand, bench=None):
    """HK exact F (primary) + HAC(6) (primary) + KZ step-down / GMM / no-short
    (supplementary), for one candidate on one sample."""
    bench = BENCH if bench is None else list(bench)
    X = sdf[bench].values.astype(float)
    y = sdf[cand].values.astype(float)
    hk = R.hk_test(X, y)
    kz = R.kz_test(X, y)
    hac = R.hac_spanning(X, y, HAC_LAG)
    out = {
        "Sample": label, "Asset": cand, "T": int(hk["T"]), "K": len(bench),
        "First": str(sdf.index.min()), "Last": str(sdf.index.max()),
        "alpha_mth_%": round(float(kz["alpha_hat"][0]) * 100, 3),
        "alpha_ann_%": round(float(kz["alpha_hat"][0]) * 1200, 2),
        "delta": round(float(kz["delta_hat"][0]), 4),
        # ---- primary inference (paper) ----
        "HK_F": round(float(hk["F"]), 3),
        "HK_df1": hk["df1"], "HK_df2": hk["df2"],
        "HK_p": round(float(hk["p_value"]), 4),
        f"HAC{HAC_LAG}_joint_p": round(hac["joint_p"], 4),
        f"HAC{HAC_LAG}_alpha_t": round(hac["alpha_t"], 3),
        # ---- supplementary ----
        "stepdown_F1_alpha": round(float(kz["F_Z"]), 3),
        "stepdown_F1_df": str(kz["df_Z"]),
        "stepdown_F1_p": round(float(kz["p_Z"]), 4),
        "stepdown_F2_delta": round(float(kz["F_K"]), 3),
        "stepdown_F2_df": str(kz["df_K"]),
        "stepdown_F2_p": round(float(kz["p_K"]), 4),
        f"HAC{HAC_LAG}_tangency_p": round(hac["tangency_p_two"], 4),
        f"HAC{HAC_LAG}_GMV_p": round(hac["gmv_p"], 4),
        f"HAC{HAC_LAG}_noShort_1sided_p": round(hac["tangency_p_one"], 4),
    }
    for L in HAC_SENS_LAGS:
        out[f"HAC{L}_joint_p"] = round(R.hac_spanning(X, y, L)["joint_p"], 4)
    return out


def _st(p):
    return ("***" if p < .001 else "**" if p < .01 else "*" if p < .05
            else "." if p < .10 else "")

# ------------------------------------------------------------ robustness ---

def _three_tests(M, end=None):
    """Run the three headline spanning tests on a (possibly truncated) panel."""
    MM = M if end is None else M[M.index <= pd.Period(end, "M")]
    g = sample_for(MM, "Gold")
    b = sample_for(MM, "Bitcoin")
    m = MM.dropna(subset=BENCH + [RF_COL] + CANDIDATES)
    return {lab: spanning_row(lab, sdf, cand) for lab, sdf, cand in
            [("Gold longest", g, "Gold"), ("Gold matched", m, "Gold"),
             ("Bitcoin", b, "Bitcoin")]}


def robustness(M):
    """Three separate sensitivity checks, each isolating ONE change:

    A. International-shares proxy. VANINSH (requested) vs MSCIACWI (the
       series the source paper used), on the IDENTICAL window so the
       comparison is like-for-like.
    B. Sample end. The same VANINSH panel truncated at the paper's last
       month versus extended to the latest month available. This isolates
       what the extra data does, which is the point of the request.
    C. Missing distributions. VANINSH is a fund unit price and shows an
       annual June ex-distribution drop, so it understates the total return
       of international shares. A flat annual yield is added back.
    """
    rows = []

    def add(tag, res):
        for lab, r in res.items():
            rows.append({"Check": tag, "Test": lab, "T": r["T"],
                         "First": r["First"], "Last": r["Last"],
                         "alpha_mth_%": r["alpha_mth_%"], "HK_F": r["HK_F"],
                         "HK_p": r["HK_p"],
                         f"HAC{HAC_LAG}_p": r[f"HAC{HAC_LAG}_joint_p"]})

    add(f"A. VANINSH, end {PAPER_END}", _three_tests(M, PAPER_END))
    acwi_ok = False
    if ACWI_FILE and os.path.exists(ACWI_FILE):
        try:
            a = _level_xlsx_monthly(ACWI_FILE, "International Shares")
            a.index = a.index.to_period("M")
            Ma = M.copy(); Ma["International Shares"] = a.reindex(M.index)
            add(f"A. MSCIACWI, end {PAPER_END}", _three_tests(Ma, PAPER_END))
            acwi_ok = True
        except Exception as e:
            print(f"[warn] MSCIACWI comparison skipped: {e}")

    add(f"B. VANINSH, end {SAMPLE_END} (full)", _three_tests(M))

    for y in DIST_YIELDS:
        Md = M.copy()
        Md["International Shares"] = Md["International Shares"] + y / 12.0
        add(f"C. VANINSH +{y*100:.1f}%pa dist, end {SAMPLE_END}",
            _three_tests(Md))
    return pd.DataFrame(rows), acwi_ok


def distribution_check(M):
    """Mean return by calendar month. A fund unit price that pays an annual
    June distribution shows a June trough that an accumulation index does
    not; this is the evidence that VANINSH is price-only."""
    d = M[["International Shares", "Australian Shares"]].dropna().copy()
    d["m"] = d.index.month
    t = (d.groupby("m")[["International Shares", "Australian Shares"]].mean() * 100).round(2)
    t.columns = ["VANINSH mean %", "ASA52 (accum) mean %"]
    gaps = {c: round(float(t.loc[6, c] - t.drop(6)[c].mean()), 2) for c in t.columns}
    return t, gaps


# ---------------------------------------------------------------- driver ---

def main():
    M = build_panel()

    cov = coverage(M)
    cov.to_csv("K7_coverage.csv", index=False)

    desc = descriptives(M)
    desc.to_csv("K7_descriptives.csv", index=False)

    gold_s = sample_for(M, "Gold")
    btc_s  = sample_for(M, "Bitcoin")
    matched = M.dropna(subset=BENCH + [RF_COL] + CANDIDATES)

    matched[BENCH + CANDIDATES].corr().round(3).to_csv("K7_correlations.csv")
    dcorr = downside_corr(matched)
    dcorr.to_csv("K7_downside_corr.csv", index=False)

    rows = [
        spanning_row("Longest Gold sample", gold_s, "Gold"),
        spanning_row("Matched Bitcoin-period", matched, "Gold"),
        spanning_row("Longest Bitcoin sample", btc_s, "Bitcoin"),
    ]
    span = pd.DataFrame(rows)
    span.to_csv("K7_spanning.csv", index=False)

    # ------------------------------ report ------------------------------
    pd.set_option("display.width", 250, "display.max_columns", 200)
    W = 100
    print("=" * W)
    print("K = 7 MEAN-VARIANCE SPANNING: GOLD AND BITCOIN vs BROAD ASSET-CLASS BENCHMARK")
    print("Methodology follows Gold_Bitcoin_AusSuperBalancedFund_13082026.docx")
    print("=" * W)
    print("Benchmark (K = 7), Cash held outside the risky set:")
    for c, f, role in BENCH_FILES:
        print(f"   {c:24s} {f:16s} {role}")
    print(f"   {'Cash (risk-free)':24s} {CASH_FILE:16s} Bloomberg AusBond Bank Bill")
    print(f"   Test assets (N=1, separately): {GOLD_FILE}, {BTC_FILE}")

    print("\n" + "=" * W)
    print("SERIES COVERAGE - what binds the joint sample")
    print("=" * W)
    print(cov.to_string(index=False))
    binder = cov[cov.Series.isin(BENCH)].sort_values("First").iloc[-1]
    print(f"\nJoint K=7 start is set by {binder['Series']} (first return {binder['First']}).")
    print(f"Largest Gold sample    : {gold_s.index.min()} .. {gold_s.index.max()}  T={len(gold_s)}")
    print(f"Largest Bitcoin sample : {btc_s.index.min()} .. {btc_s.index.max()}  T={len(btc_s)}")

    print("\n" + "=" * W)
    print("DESCRIPTIVE STATISTICS (each series over its own largest sample)")
    print("=" * W)
    print(desc.to_string(index=False))

    print("\n" + "=" * W)
    print("CO-MOVEMENT WITH AUSTRALIAN SHARES (matched sample)")
    print("=" * W)
    print(dcorr.to_string(index=False))

    print("\n" + "=" * W)
    print("CORRELATION MATRIX (matched sample)")
    print("=" * W)
    print(matched[BENCH + CANDIDATES].corr().round(3).to_string())

    print("\n" + "=" * W)
    print("SPANNING TESTS (K = 7, N = 1)")
    print("=" * W)
    for _, r in span.iterrows():
        print(f"\n-- {r['Asset']:8s} | {r['Sample']:24s} | {r['First']}..{r['Last']} | T={int(r['T'])}")
        print(f"   intercept alpha = {r['alpha_mth_%']:.3f}% per month "
              f"({r['alpha_ann_%']:.2f}% p.a.)   delta = {r['delta']:+.4f}")
        hp = r["HK_p"]; ha = r[f"HAC{HAC_LAG}_joint_p"]
        print(f"   PRIMARY   HK exact F = {r['HK_F']:.3f}  df({r['HK_df1']}, {r['HK_df2']})  "
              f"p = {hp:.4f}{_st(hp)}")
        print(f"             HAC({HAC_LAG}) joint p = {ha:.4f}{_st(ha)}   "
              f"(alpha t = {r[f'HAC{HAC_LAG}_alpha_t']:.3f})")
        cells = "  ".join(f"L{L}:{r[f'HAC{L}_joint_p']:.4f}" for L in
                          sorted(HAC_SENS_LAGS + [HAC_LAG]))
        print(f"             HAC lag sensitivity (joint p)  {cells}")
        f1p = r["stepdown_F1_p"]; f2p = r["stepdown_F2_p"]
        print(f"   SUPPL.    step-down F1 alpha (tangency) = {r['stepdown_F1_alpha']:.3f} "
              f"p = {f1p:.4f}{_st(f1p)}   |   F2 delta|alpha (GMV) = "
              f"{r['stepdown_F2_delta']:.3f} p = {f2p:.4f}{_st(f2p)}")
        o1 = r[f"HAC{HAC_LAG}_noShort_1sided_p"]
        print(f"             HAC({HAC_LAG}) tangency p = {r[f'HAC{HAC_LAG}_tangency_p']:.4f} | "
              f"GMV p = {r[f'HAC{HAC_LAG}_GMV_p']:.4f} | one-sided no-short p = {o1:.4f}{_st(o1)}")

    # -------------------------- robustness --------------------------
    cal, gaps = distribution_check(M)
    rob, acwi_ok = robustness(M)
    rob.to_csv("K7_robustness.csv", index=False)
    cal.to_csv("K7_distribution_check.csv")

    print("\n" + "=" * W)
    print("DATA CHECK - is VANINSH a total-return series?")
    print("=" * W)
    print(cal.to_string())
    print(f"\nJune minus other-month average:  VANINSH {gaps['VANINSH mean %']:+.2f} pp   "
          f"ASA52 (accumulation) {gaps['ASA52 (accum) mean %']:+.2f} pp")
    print("A fund unit price that distributes income each June shows a June trough that an")
    print("accumulation index does not. VANINSH's June gap is about twice ASA52's, so VANINSH")
    print("is read as PRICE-ONLY: it understates international-share total returns, which")
    print("biases the spanning test IN FAVOUR of the candidate. Check C quantifies this.")

    print("\n" + "=" * W)
    print("ROBUSTNESS - each block changes exactly one thing")
    print("=" * W)
    print(rob.to_string(index=False))
    if acwi_ok:
        print("\nA: VANINSH vs MSCIACWI on the identical window isolates the proxy choice.")
    print("B: same series, later end date, isolates what the extra months do.")
    print("C: distribution add-back isolates the price-only bias.")

    print("\nSignificance: *** p<.001  ** p<.01  * p<.05  . p<.10")
    print("HK exact F uses the Kan-Zhou eq. (28) N=1 form, F(2, T-K-1).")
    print("Saved: K7_coverage.csv, K7_descriptives.csv, K7_downside_corr.csv,")
    print("       K7_correlations.csv, K7_spanning.csv, K7_robustness.csv,")
    print("       K7_distribution_check.csv")


if __name__ == "__main__":
    main()
