"""
Gold & Bitcoin vs Australian super benchmarks — full replication script
=======================================================================
Version 4. Spanning tests rewritten to follow Kan and Zhou (2012, "Tests of
Mean-Variance Spanning", Annals of Economics and Finance 13, 139-187; "KZ")
and Huberman and Kandel (1987, Journal of Finance 42, 873-888; "HK")
equation by equation. Cash is now the Bloomberg AusBond Bank Bill Index
(BAUBIL) for BOTH funds.

Produces, for AustralianSuper and the Australian Retirement Trust (ART):
  (A) spanning tests: HK exact F, and the KZ Wald / LR / LM trio;
  (B) the KZ step-down test (their eqs 46-47): alpha first, then delta|alpha;
  (C) GMM / HAC-robust spanning inference for non-normal, heteroskedastic
      returns (KZ section 5), including the elliptical kurtosis-adjusted
      Wald of KZ eq (53);
  (D) no-short-sale (long-only) diagnostics;
  (E) the Platanakis-Urquhart (2020) out-of-sample battery, uncapped and
      with the candidate capped, over the full integer risk-aversion grid
      lambda = 1,2,...,10.

RISK-FREE RATE. The Bloomberg AusBond Bank Bill Index (BAUBIL.xlsx, monthly
index level -> monthly return) is the risk-free rate for BOTH funds and in
EVERY excess return the script forms: the spanning regressions, the long-only
tangency Sharpes, the OOS optimiser inputs, and the reported Sharpe/Sortino
ratios. AustralianSuper's own Cash investment option is loaded only for the
like-for-like comparison in rf_diagnostics() and is never used as rf.

WHAT CHANGED IN V4 AND WHY (all verified numerically against the papers)
-----------------------------------------------------------------------
1. G-matrix. KZ eq (17)/(31): G = T A(X'X)^-1 A' = [[1+a1, b1],[b1, c1]]
   with a1 = mu1'V11^-1 mu1, b1 = mu1'V11^-1 1_K, c1 = 1_K'V11^-1 1_K and
   V11 the ML (divide by T) covariance. Built explicitly here. The ridge
   term previously used introduced ~4e-6 relative error and is dropped.
2. delta sign. KZ define delta = 1_N - beta 1_K (their section 2.1). Used
   here. (The quadratic form is invariant to the sign, but the reported
   delta now matches the paper.)
3. HK F-test. HK's published F contains a typo (KZ footnote 9): it prints
   U where U^(1/2) belongs. Correct forms:
       N >= 2 : (U^(-1/2) - 1)(T-K-N)/N     ~ F(2N, 2(T-K-N))   KZ eq (27)
       N  = 1 : (U^(-1)   - 1)(T-K-1)/2     ~ F(2, T-K-1)       KZ eq (28)
   KZ footnote 1 warns that using the N>=2 form when N=1 is a common error.
4. N = 1 p-values. KZ p.147: when N = 1 the Wald, LR and LM statistics are
   monotone transformations of each other, so all three take the exact
   p-value of the F-test in eq (28). Previously each was pushed through a
   generic chi2-to-F rescaling, producing three different p-values for what
   is one test. Fixed.
5. Step-down. KZ eqs (46)-(47):
       F1 = ((T-K-N)/N)  (|Sigma_bar|/|Sigma_hat| - 1) ~ F(N, T-K-N)
       F2 = ((T-K-N+1)/N)(|Sigma_til|/|Sigma_bar| - 1) ~ F(N, T-K-N+1)
   where Sigma_bar imposes alpha = 0 and Sigma_til imposes alpha = 0 and
   delta = 0. F1 and F2 are independent under the null. The previous code
   used a conditional-variance shortcut with the wrong scale (T instead of
   T-K-N) and the wrong second df, and its delta statistic was not KZ's
   sequential test at all. Now computed from restricted residual variances.
   Ordering follows KZ footnote 17: test the economically more important
   component first, which here is alpha (the tangency channel).
6. Excess returns are permitted. KZ footnote 2 states returns may be total
   or excess of the risk-free rate; the restriction beta'1_K = 1 is
   unchanged. The v3 guard that blocked RETURNS='excess' was wrong and is
   removed.

INTERPRETATION WARNING (KZ section 3.1, pp.155-156)
---------------------------------------------------
KZ show the spanning tests have strong power against assets that shrink the
global minimum-variance portfolio, but LITTLE power against assets that only
improve the tangency portfolio, because delta is estimated far more
precisely than alpha. Gold and Bitcoin here move alpha and not delta, i.e.
they sit in exactly the region where the joint test is weakest. A modest
joint p-value is therefore expected and should NOT be read as evidence of no
economic benefit; KZ recommend reading the step-down components separately,
which is why F1 (alpha) is reported alongside the joint tests.

Data
----
  AustralianSuper_Daily_Rates_01_Jul_2008_-_21_Aug_2026.csv  daily % rates
  ART_AustralianShares_Daily_Rates.csv                       unit exit price
  ART_InternationalSharesUnhedged_Daily_Rates.csv            unit exit price
  ART_FixedIncome_Daily_Rates.csv                            unit exit price
  GOLDAX_split_adjusted.csv / BTCAUD.csv                     daily price
  BAUBIL.xlsx   Bloomberg AusBond Bank Bill Index, monthly level -> Cash

Requires: numpy, pandas, scipy, openpyxl.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv
from scipy.optimize import minimize

# ------------------------------- configuration ----------------------------

DATA_DIR = os.environ.get("DATA_DIR", ".")

AU_FILE   = "AustralianSuper_Daily Rates 01 Jul 2008 - 21 Aug 2026.csv"
ART_AUS   = "ART_AustralianShares_Daily_Rates.csv"
ART_INTL  = "ART_InternationalSharesUnhedged_Daily_Rates.csv"
ART_BOND  = "ART_FixedIncome_Daily_Rates.csv"
GOLD_FILE = "GOLDAX_split_adjusted.csv"
BTC_FILE  = "BTCAUD.csv"
CASH_FILE = "BAUBIL.xlsx"

BENCH      = ["Australian Shares", "International Shares", "Bonds"]
RF_COL     = "Cash"       # this column is the BAUBIL series (see _baubil_monthly)
RETURNS    = "raw"        # "raw" or "excess" (both valid; KZ footnote 2)
SAMPLE_END = "2026-07"    # last complete month

HAC_ROBUST_LAG = 6
HAC_SENS_LAGS  = [4, 5, 12]
HAC_LAGS       = sorted(set([HAC_ROBUST_LAG] + HAC_SENS_LAGS))

CANDIDATE_CAP = 0.10
BOOT_B        = 1000
BOOT_BLOCK    = 3
BOOT_SEED     = 12345
DNW_NSIM      = 40000     # chi-bar-squared simulation draws (Kodde-Palm null)

LAMBDAS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]      # full integer grid, 1..10
LAMBDAS_DIAG = LAMBDAS + [25, 50, 100]         # corner diagnostic adds high-lambda
WINDOW  = 60
STRATEGIES = [
    "Markowitz",
    "Markowitz with Gens",
    "Bayes-Stein",
    "Bayes-Stein with Gens",
    "Black-Litterman",
]

# ============ spanning tests, following Kan-Zhou (2012) exactly ============

def _to2d(arr):
    a = np.asarray(arr, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _efficient_set_constants(R):
    """KZ p.148: a = mu'V^-1 mu, b = mu'V^-1 1, c = 1'V^-1 1, with mu and V
    the ML (divide by T) mean and covariance."""
    R = _to2d(R); T = R.shape[0]
    mu = R.mean(axis=0)
    V = (R - mu).T @ (R - mu) / T
    Vi = inv(V)
    one = np.ones(R.shape[1])
    return float(mu @ Vi @ mu), float(mu @ Vi @ one), float(one @ Vi @ one)


def spanning_core(R_bench, R_test):
    """OLS spanning regression plus every quantity KZ's tests need.

    Returns alpha, delta = 1_N - beta 1_K (KZ s.2.1), the ML residual
    covariance Sigma_hat, the restricted covariances Sigma_bar (alpha = 0)
    and Sigma_til (alpha = 0 and delta = 0), U = |Sigma_hat|/|Sigma_til|,
    and G (KZ eq 31) and H (KZ eq 22).
    """
    R1 = _to2d(R_bench); Y = _to2d(R_test)
    T, K = R1.shape; N = Y.shape[1]
    X = np.column_stack([np.ones(T), R1])

    B = np.linalg.lstsq(X, Y, rcond=None)[0]          # [alpha; beta']
    alpha = B[0, :].copy()
    beta = B[1:, :].T                                  # N x K
    E = Y - X @ B
    Sig_hat = E.T @ E / T
    delta = np.ones(N) - beta @ np.ones(K)             # KZ sign convention

    # restricted: alpha = 0  -> regress Y on R1 with no intercept
    Ba = np.linalg.lstsq(R1, Y, rcond=None)[0]
    Ea = Y - R1 @ Ba
    Sig_bar = Ea.T @ Ea / T

    # restricted: alpha = 0 and beta 1_K = 1  -> substitute out one benchmark
    Z = R1[:, :-1] - R1[:, [-1]]
    Yz = Y - R1[:, [-1]]
    Bz = np.linalg.lstsq(Z, Yz, rcond=None)[0]
    Ez = Yz - Z @ Bz
    Sig_til = Ez.T @ Ez / T

    U = float(np.linalg.det(Sig_hat) / np.linalg.det(Sig_til))

    a1, b1, c1 = _efficient_set_constants(R1)
    G = np.array([[1.0 + a1, b1], [b1, c1]])           # KZ eq (17)/(31)
    Theta = np.column_stack([alpha, delta])            # N x 2
    Si = inv(Sig_hat)
    H = Theta.T @ Si @ Theta                           # KZ eq (22)

    return dict(T=T, K=K, N=N, alpha=alpha, beta=beta, delta=delta,
                Sig_hat=Sig_hat, Sig_bar=Sig_bar, Sig_til=Sig_til,
                U=U, G=G, H=H, resid=E, X=X, a1=a1, b1=b1, c1=c1)


def hk_test(R_bench, R_test, label=None):
    """Huberman-Kandel (1987) likelihood-ratio spanning test, with the exact
    F distribution given by KZ eqs (27) and (28).

    KZ footnote 9: HK's printed statistic uses U where U^(1/2) belongs.
    KZ footnote 1: the N>=2 form must not be used when N = 1.
    """
    c = spanning_core(R_bench, R_test)
    T, K, N, U = c["T"], c["K"], c["N"], c["U"]
    if not (U and U > 0):
        F = p = np.nan; df1 = 2 * N; df2 = T - K - N
    elif N == 1:
        df1, df2 = 2, T - K - 1                        # KZ eq (28)
        F = (1.0 / U - 1.0) * ((T - K - 1) / 2.0)
        p = float(1 - stats.f.cdf(F, df1, df2))
    else:
        df1, df2 = 2 * N, 2 * (T - K - N)              # KZ eq (27)
        F = (U ** -0.5 - 1.0) * ((T - K - N) / float(N))
        p = float(1 - stats.f.cdf(F, df1, df2))
    return dict(test="HK", label=label, T=T, K=K, N=N, U=U,
                F=F, p_value=p, df1=df1, df2=df2,
                alpha_hat=c["alpha"], delta_hat=c["delta"])


def kz_test(R_bench, R_test, label=None):
    """Kan-Zhou (2012) Wald / LR / LM spanning tests plus the step-down test.

    W  = T (lambda1 + lambda2)             KZ eq (25)
    LR = -T ln U                           KZ eq (19)
    LM = T sum lambda_i/(1+lambda_i)       KZ eq (26)
    lambda_i are the eigenvalues of H G^-1 (KZ Lemma 1: they measure the
    maximum and minimum gap between the two frontiers in squared Sharpe
    ratio terms).

    For N = 1 (KZ p.147) all three are monotone transformations of the exact
    F in eq (28), so all three carry that single exact p-value. For N >= 2
    the asymptotic chi2(2N) p-values are reported together with the KZ
    small-sample degrees-of-freedom corrections (their footnote 6).

    Step-down (KZ eqs 46-47), ordered alpha then delta|alpha per footnote 17:
        F1 = ((T-K-N)/N)  (|Sigma_bar|/|Sigma_hat| - 1) ~ F(N, T-K-N)
        F2 = ((T-K-N+1)/N)(|Sigma_til|/|Sigma_bar| - 1) ~ F(N, T-K-N+1)
    """
    c = spanning_core(R_bench, R_test)
    T, K, N, U, G, H = c["T"], c["K"], c["N"], c["U"], c["G"], c["H"]

    lam = np.sort(np.linalg.eigvals(H @ inv(G)).real)[::-1]
    lam = np.clip(lam, 0.0, None)
    W = float(T * lam.sum())
    LR = float(-T * np.log(U)) if (U and U > 0) else np.nan
    LM = float(T * (lam / (1.0 + lam)).sum())

    hk = hk_test(R_bench, R_test, label)
    if N == 1:
        # all three are monotone transforms of the exact F (KZ p.147)
        p_W = p_LR = p_LM = hk["p_value"]
        W_F = LR_F = LM_F = hk["F"]
        note = "N=1: W, LR, LM share the exact F p-value (KZ eq 28)"
    else:
        df = 2 * N
        p_W  = float(1 - stats.chi2.cdf(W  * (T - K - N + 1) / T, df))
        p_LR = float(1 - stats.chi2.cdf(LR * (T - K - (N + 1) / 2.0) / T, df))
        p_LM = float(1 - stats.chi2.cdf(LM * (T - K + 1) / T, df))
        W_F = LR_F = LM_F = np.nan
        note = "N>=2: chi2(2N) with KZ footnote-6 small-sample corrections"

    # ---- step-down, KZ eqs (46)-(47) ----
    det = np.linalg.det
    F1 = ((T - K - N) / N) * (det(c["Sig_bar"]) / det(c["Sig_hat"]) - 1.0)
    p1 = float(1 - stats.f.cdf(F1, N, T - K - N))
    F2 = ((T - K - N + 1) / N) * (det(c["Sig_til"]) / det(c["Sig_bar"]) - 1.0)
    p2 = float(1 - stats.f.cdf(F2, N, T - K - N + 1))

    return dict(
        test="KZ", label=label, T=T, K=K, N=N, U=U, note=note,
        W_chi2=W, LR_chi2=LR, LM_chi2=LM,
        p_W=p_W, p_LR=p_LR, p_LM=p_LM,
        W_F=W_F, LR_F=LR_F, LM_F=LM_F,
        lambda1=float(lam[0]), lambda2=float(lam[1]) if len(lam) > 1 else 0.0,
        # step-down: F_Z = tangency (alpha), F_K = GMV (delta | alpha)
        F_Z=float(F1), p_Z=p1, df_Z=(N, T - K - N),
        F_K=float(F2), p_K=p2, df_K=(N, T - K - N + 1),
        alpha_hat=c["alpha"], delta_hat=c["delta"], beta_hat=c["beta"],
        G=G, H=H, a1=c["a1"], b1=c["b1"], c1=c["c1"], sh2_K=c["a1"],
    )


def gmm_wald_elliptical(R_bench, R_test, label=None):
    """KZ eq (53): GMM Wald test valid under a multivariate elliptical
    return distribution, W_a^e = T tr(H G_a^-1) with

        G_a = [[1+(1+k)a1, (1+k)b1], [(1+k)b1, (1+k)c1]]      KZ eq (54)
        k   = E[((R-mu)'V^-1(R-mu))^2] / ((N+K)(N+K+2)) - 1   KZ eq (52)

    KZ Table 5 shows the ordinary Wald over-rejects badly when returns have
    fat tails (the bias limit is about k/2, their eq 55), which matters here
    because gold and especially Bitcoin have pronounced excess kurtosis.
    """
    c = spanning_core(R_bench, R_test)
    R1 = _to2d(R_bench); Y = _to2d(R_test)
    Rall = np.column_stack([R1, Y])
    T, M = Rall.shape
    mu = Rall.mean(axis=0)
    V = (Rall - mu).T @ (Rall - mu) / T
    dm = Rall - mu
    d2 = np.einsum("ij,jk,ik->i", dm, inv(V), dm)      # Mahalanobis squared
    kappa = float((d2 ** 2).mean() / (M * (M + 2)) - 1.0)
    a1, b1, c1 = c["a1"], c["b1"], c["c1"]
    G_a = np.array([[1 + (1 + kappa) * a1, (1 + kappa) * b1],
                    [(1 + kappa) * b1,     (1 + kappa) * c1]])
    Wa_e = float(T * np.trace(c["H"] @ inv(G_a)))
    df = 2 * c["N"]
    return dict(label=label, kappa=kappa, Wa_e=Wa_e,
                p=float(1 - stats.chi2.cdf(Wa_e, df)), df=df)

# ==================== OOS functions (verbatim: OOS_tests.py) ====================

def optimise(mu, Sigma, lam, floor):
    """Long-only quadratic-utility optimiser with optional Gens floor."""
    N = len(mu)
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0}]
    bounds = [(floor, 1.0)] * N
    obj = lambda x: -(x @ mu - lam / 2.0 * x @ Sigma @ x)
    # multi-start: equal weights and a corner tilted to the highest mean
    starts = [np.full(N, 1.0 / N),
              np.eye(N)[np.argmax(mu)] * (1 - N * floor) + floor]
    best, best_val = None, np.inf
    for x0 in starts:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    w = np.clip(best, floor, None)
    return w / w.sum()


def bayes_stein(mu, Sigma, T):
    """Jorion (1986) shrinkage estimates of the mean vector and covariance."""
    N = len(mu)
    Si = np.linalg.inv(Sigma)
    ones = np.ones(N)
    mu_g = (Si @ ones) @ mu / (ones @ Si @ ones)   # grand (GMV) mean
    d = mu - mu_g * ones
    q = d @ Si @ d
    phi = (N + 2) / q
    g = (N + 2) / ((N + 2) + T * q)                # shrinkage intensity
    mu_bs = (1 - g) * mu + g * mu_g * ones
    Sigma_bs = ((T + phi + 1) / (T + phi)) * Sigma \
        + (phi / (T * (T + phi + 1))) * np.outer(ones, ones) / (ones @ Si @ ones)
    return mu_bs, Sigma_bs


def black_litterman(mu, Sigma, lam, c=0.1625):
    """Black-Litterman posterior with 1/N reference, P = I, Omega = Sigma."""
    N = len(mu)
    x_ref = np.full(N, 1.0 / N)
    Pi = lam * Sigma @ x_ref                       # implied excess returns
    Omega = Sigma                                  # (1/delta) P Sigma P', 1/delta = 1
    cSi = np.linalg.inv(c * Sigma)
    Oi = np.linalg.inv(Omega)
    A = np.linalg.inv(cSi + Oi)
    mu_bl = A @ (cSi @ Pi + Oi @ mu)               # views Q = sample means
    Sigma_bl = Sigma + A
    return mu_bl, Sigma_bl


def strategy_weights(strategy, excess_window, lam):
    """Portfolio weights for one strategy given the estimation window."""
    mu = excess_window.mean(axis=0)
    Sigma = np.cov(excess_window, rowvar=False, ddof=1)
    N, T = len(mu), excess_window.shape[0]
    floor = 1.0 / (2 * N) if "Gens" in strategy else 0.0
    if strategy.startswith("Bayes-Stein"):
        mu, Sigma = bayes_stein(mu, Sigma, T)
    elif strategy == "Black-Litterman":
        mu, Sigma = black_litterman(mu, Sigma, lam)
    return optimise(mu, Sigma, lam, floor)

# ------------------------------ backtest loop -----------------------------

def run_backtest(data, candidate):
    """Expanding-window OOS returns for benchmark and candidate portfolios."""
    bench = ["Australian Shares", "International Shares", "Bonds"]
    plus = bench + [candidate]
    R_b, R_p = data[bench].values, data[plus].values
    rf = data[RF_COL].values                       # BAUBIL monthly risk-free
    EX_b, EX_p = R_b - rf[:, None], R_p - rf[:, None]
    T = len(data)

    results, weights = {}, {}
    for strat in STRATEGIES:
        for lam in LAMBDAS:
            for tag, R, EX in (("bench", R_b, EX_b), ("plus", R_p, EX_p)):
                rets, ws = [], []
                for t in range(WINDOW, T):
                    w = strategy_weights(strat, EX[:t], lam)  # info up to t-1
                    rets.append(w @ R[t])                     # realised at t
                    ws.append(w)
                results[(strat, lam, tag)] = np.array(rets)
                weights[(strat, lam, tag)] = np.array(ws)
    return results, weights, rf[WINDOW:]

# ------------------------------ performance -------------------------------

def sharpe(r, rf):
    ex = r - rf
    return ex.mean() * 12 / (ex.std(ddof=1) * np.sqrt(12))

def sortino(r, rf):
    ex = r - rf
    dd = np.sqrt((np.minimum(ex, 0.0) ** 2).mean()) * np.sqrt(12)
    return ex.mean() * 12 / dd

def omega(r, rf=None):
    """Omega ratio at threshold 0 on RAW returns (note: Sharpe and Sortino use
    EXCESS returns over Cash, so Omega is on a different basis; this is the
    source script's convention, kept for continuity). rf is accepted for a
    common call signature but deliberately unused. Returns +inf / nan at the
    degenerate no-loss / no-data boundaries instead of raising."""
    gains = r[r > 0].sum(); losses = -r[r < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses

METRICS = [("Sharpe Ratio", sharpe), ("Sortino Ratio", sortino),
           ("Omega Ratio", omega)]

# --------------------------------- main -----------------------------------
# ==================== no-short-sale additions (new) ====================

def dnw_one_sided(R_bench, R_test):
    """One-sided t-test on the spanning intercept, H1: alpha > 0.

    ATTRIBUTION CORRECTED. This is a plain one-sided t on alpha, motivated by
    the long-only constraint. It is NOT the De Roon-Nijman-Werker (2001) test,
    which converts short-sale constraints into inequality restrictions with a
    mixed chi-bar-squared null. For a single test asset the two coincide in
    spirit (only a positive alpha is exploitable long-only) but not in
    distribution theory. Classical (homoskedastic) OLS standard error."""
    R_test = _to2d(R_test)
    T, K = R_bench.shape
    Xa = np.column_stack([np.ones(T), R_bench])
    bf = np.linalg.lstsq(Xa, R_test, rcond=None)[0]
    resid = (R_test - Xa @ bf).reshape(-1)
    s2 = resid @ resid / (T - K - 1)
    se_a = np.sqrt(s2 * inv(Xa.T @ Xa)[0, 0])
    a = float(bf[0, 0]); t = a / se_a
    return a, t, float(1 - stats.t.cdf(t, T - K - 1))


def long_only_tangency_sharpe(EX):
    """Annualised max Sharpe over long-only, fully-invested weights."""
    mu = EX.mean(axis=0); S = np.cov(EX, rowvar=False, ddof=1); N = len(mu)
    neg_sr = lambda x: -(x @ mu) / np.sqrt(x @ S @ x) if x @ S @ x > 0 else 1e9
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0}]
    bnds = [(0.0, 1.0)] * N
    best, bv = np.full(N, 1.0 / N), np.inf
    for x0 in [np.full(N, 1.0 / N)] + list(np.eye(N)):
        r = minimize(neg_sr, x0, method="SLSQP", bounds=bnds, constraints=cons,
                     options={"maxiter": 1000, "ftol": 1e-12})
        if r.success and r.fun < bv:
            best, bv = r.x, r.fun
    x = np.clip(best, 0.0, None); x = x / x.sum()
    return float((x @ mu) / np.sqrt(x @ S @ x) * np.sqrt(12)), x


# ============ HAC-robust inference for the OLS spanning test ===========
# OLS coefficients, Newey-West (Bartlett) covariance. Validated to machine
# precision against statsmodels OLS(cov_type='HAC', use_correction=False).

def _hac_cov(X, resid, L):
    """Newey-West (Bartlett) sandwich covariance of the OLS coefficients."""
    u = X * resid[:, None]
    XtX_inv = inv(X.T @ X)
    B = u.T @ u
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        G = u[j:].T @ u[:-j]
        B += w * (G + G.T)
    return XtX_inv @ B @ XtX_inv

def hac_spanning(R_bench, R_test, L):
    """HAC-robust test of the spanning restriction on OLS estimates.
    H0(joint): alpha=0 AND delta=beta'i-1=0. Returns joint Wald chi2(2),
    tangency (alpha) and GMV (delta) channel tests, and the one-sided
    tangency test, all using the Newey-West(L) covariance. Channel p-values
    use t(T-K-1) (matching the manuscript's Newey-West row); the joint uses
    chi2(2)."""
    y = np.asarray(R_test)
    if y.ndim > 1 and y.shape[1] > 1:
        raise NotImplementedError(
            "hac_spanning supports a single test asset (N=1) only; "
            "for N>1 the restriction matrix must be expanded per asset.")
    y = y.reshape(-1)
    T, K = R_bench.shape
    X = np.column_stack([np.ones(T), R_bench])
    theta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ theta
    V = _hac_cov(X, resid, L)
    Rr = np.zeros((2, K + 1)); Rr[0, 0] = 1.0; Rr[1, 1:] = 1.0
    qv = np.array([0.0, 1.0])
    d = Rr @ theta - qv
    W = float(d @ inv(Rr @ V @ Rr.T) @ d)
    dfr = T - K - 1
    a = theta[0]; se_a = np.sqrt(V[0, 0]); t_a = a / se_a
    delta = theta[1:].sum() - 1.0
    se_d = np.sqrt(float((Rr[1:2] @ V @ Rr[1:2].T)[0, 0])); z_d = delta / se_d
    return {
        "lag": L,
        "joint_Wald_chi2": W, "joint_p": float(1 - stats.chi2.cdf(W, 2)),
        "alpha_t": t_a,
        "tangency_p_two": float(2 * (1 - stats.t.cdf(abs(t_a), dfr))),
        "tangency_p_one": float(1 - stats.t.cdf(t_a, dfr)),   # H1: alpha>0
        "gmv_t": z_d, "gmv_p": float(2 * (1 - stats.t.cdf(abs(z_d), dfr))),
    }



# ===== De Roon-Nijman-Werker (2001) spanning with short-sale constraints =====
# Implements their eqs (14)-(20) with the Kodde-Palm (1986) chi-bar-squared
# null. NOTE: this REPLACES the earlier `dnw_one_sided`, which was only a
# one-sided t on the spanning intercept and was mis-attributed to DNW.



def _long_only_tangency(mu, S, zero_beta):
    """Long-only portfolio maximising (mu'w - zero_beta)/sqrt(w'Sw)."""
    K = len(mu)
    f = lambda w: -((w @ mu - zero_beta) / np.sqrt(w @ S @ w)) if w @ S @ w > 0 else 1e9
    best, bv = np.ones(K) / K, np.inf
    for x0 in [np.ones(K) / K] + list(np.eye(K)):
        r = minimize(f, x0, method="SLSQP", bounds=[(0.0, 1.0)] * K,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                     options={"maxiter": 500, "ftol": 1e-12})
        if r.success and r.fun < bv:
            best, bv = r.x, r.fun
    w = np.clip(best, 0.0, None)
    return w / w.sum()


def _v_range(Rb_gross):
    """v in [1/E(R_gmv), 1] (DNW pp.729-730)."""
    mu = Rb_gross.mean(axis=0)
    S = np.cov(Rb_gross, rowvar=False, ddof=1)
    one = np.ones(Rb_gross.shape[1])
    wg = inv(S) @ one
    wg = wg / wg.sum()
    return 1.0 / float(wg @ mu), 1.0


def _segments(Rb_gross, v_lo, v_hi, ngrid=41, tol=1e-4):
    """Partition [v_lo, v_hi] by the support of the long-only efficient
    portfolio (the assets whose short-sale constraints do not bind)."""
    mu = Rb_gross.mean(axis=0)
    S = np.cov(Rb_gross, rowvar=False, ddof=1)
    grid = np.linspace(v_lo, v_hi, ngrid)
    segs, cur, start = [], None, v_lo
    for v in grid:
        w = _long_only_tangency(mu, S, 1.0 / v)
        sup = tuple(np.where(w > tol)[0])
        if cur is None:
            cur = sup
        elif sup != cur:
            segs.append((cur, start, v))
            cur, start = sup, v
    segs.append((cur, start, v_hi))
    return segs


from itertools import combinations

def _faces(p):
    """All index subsets F (coords allowed non-zero); the rest are pinned to 0."""
    out = []
    for k in range(p + 1):
        out.extend(combinations(range(p), k))
    return out


def _xi_batch(Z, A):
    """Kodde-Palm inequality Wald, vectorised.

    xi = min over theta <= 0 of (z-theta)' A (z-theta), A = Var[g]^-1.
    The minimiser lies on some face F (theta_F free, theta_{F^c} = 0); on that
    face the stationary point is theta_F = (A_FF)^-1 A_{F,:} z. Evaluating every
    face and keeping the smallest objective among those with theta_F <= 0 gives
    the exact projection onto the negative orthant.
    """
    Z = np.atleast_2d(Z)
    n, p = Z.shape
    best = np.einsum("ij,jk,ik->i", Z, A, Z)          # face F = {} (theta = 0)
    for F in _faces(p):
        if len(F) == 0:
            continue
        F = list(F)
        AFF_inv = inv(A[np.ix_(F, F)])
        M = AFF_inv @ A[F, :]                          # theta_F = M z
        Th = Z @ M.T
        ok = np.all(Th <= 1e-12, axis=1)
        if not ok.any():
            continue
        R = Z[ok].copy()
        R[:, F] -= Th[ok]
        obj = np.einsum("ij,jk,ik->i", R, A, R)
        idx = np.where(ok)[0]
        best[idx] = np.minimum(best[idx], obj)
    return best


def _xi(z, Vi, p):
    return float(_xi_batch(np.asarray(z, float)[None, :], Vi)[0])


def dnw_spanning(Rb_net, y_net, constrain_benchmark=True, nsim=20000, seed=11):
    """DNW spanning test. constrain_benchmark=False reproduces their
    'no short sales of the new asset' row; True is 'no short sales'."""
    Rb = 1.0 + np.asarray(Rb_net, float)
    y = 1.0 + np.asarray(y_net, float)
    T, K = Rb.shape
    v_lo, v_hi = _v_range(Rb)
    segs = (_segments(Rb, v_lo, v_hi) if constrain_benchmark
            else [(tuple(range(K)), v_lo, v_hi)])

    # stacked GMM over the segment regressions
    blocks, gvec, rows = [], [], []
    moments = []
    for sup, a_lo, a_hi in segs:
        idx = list(sup)
        Xj = np.column_stack([np.ones(T), Rb[:, idx]])
        thj = np.linalg.lstsq(Xj, y, rcond=None)[0]
        ej = y - Xj @ thj
        blocks.append((Xj, thj, ej, idx))
        moments.append(Xj * ej[:, None])
        for v in (a_lo, a_hi):
            gvec.append(v * thj[0] + thj[1:].sum() - 1.0)
            rows.append((idx, v, len(blocks) - 1))
    g = np.array(gvec)

    gt = np.concatenate(moments, axis=1)              # T x sum(kj)
    S = gt.T @ gt / T
    Dblocks = [Xj.T @ Xj / T for Xj, _, _, _ in blocks]
    Dinv = np.zeros((S.shape[0], S.shape[0]))
    o = 0
    for Dj in Dblocks:
        k = Dj.shape[0]
        Dinv[o:o + k, o:o + k] = inv(Dj)
        o += k
    Vth = Dinv @ S @ Dinv.T / T                        # Var of stacked theta

    offs, o = [], 0
    for Dj in Dblocks:
        offs.append(o); o += Dj.shape[0]
    P = np.zeros((len(g), S.shape[0]))
    for r_i, (idx, v, bi) in enumerate(rows):
        o0 = offs[bi]
        P[r_i, o0] = v
        P[r_i, o0 + 1:o0 + 1 + len(idx)] = 1.0
    Vg = P @ Vth @ P.T
    Vg = Vg + 1e-14 * np.eye(len(g))
    Vgi = inv(Vg)
    stat = _xi(g, Vgi, len(g))

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Vg)
    Z = rng.standard_normal((nsim, len(g))) @ L.T
    null = _xi_batch(Z, Vgi)
    p = float((null >= stat).mean())

    return dict(v_lo=v_lo, v_hi=v_hi, n_segments=len(segs),
                segments=[(s, round(a, 6), round(b, 6)) for s, a, b in segs],
                g=g, stat=stat, p=p, n_restrictions=len(g))

def dnw_spanning_rf(Rb_excess, y_excess, constrain_benchmark=True):
    """De Roon-Nijman-Werker (2001) short-sale-constrained spanning test when a
    risk-free asset (BAUBIL) is available.

    With a riskless asset, the only relevant zero-beta rate is the risk-free
    rate itself, so the inequality restriction is evaluated at that single
    point: on excess returns the restriction is alpha <= 0, where alpha is the
    intercept of the candidate's excess return on the excess returns of the
    benchmark assets (all benchmark assets if only the candidate is long-only;
    the assets held in the long-only tangency portfolio if all assets are
    long-only). With one inequality restriction the Kodde-Palm statistic is
    xi = max(0, alpha/se)^2 and its null distribution is the chi-bar-squared
    mixture 0.5*chi2(0) + 0.5*chi2(1), so p = 0.5 * P(chi2(1) > xi).
    The standard error is heteroskedasticity-robust (White), as in dnw_spanning.
    (dnw_spanning, which evaluates a range of zero-beta rates on gross returns,
    is not appropriate for excess returns: the admissible range collapses or
    inverts when the minimum-variance portfolio earns less than the risk-free
    rate.)
    """
    Xe = np.asarray(Rb_excess, float); ye = np.asarray(y_excess, float)
    T, K = Xe.shape
    if constrain_benchmark:
        mu = Xe.mean(axis=0); S = np.cov(Xe, rowvar=False, ddof=1)
        w = _long_only_tangency(1.0 + mu, S, 1.0)   # tangency w.r.t. the risk-free rate
        idx = list(np.where(w > 1e-4)[0])
    else:
        idx = list(range(K))
    X = np.column_stack([np.ones(T), Xe[:, idx]])
    th = np.linalg.lstsq(X, ye, rcond=None)[0]
    e = ye - X @ th
    D = X.T @ X / T
    Sm = (X * e[:, None]).T @ (X * e[:, None]) / T
    V = inv(D) @ Sm @ inv(D) / T
    a, se = th[0], np.sqrt(V[0, 0])
    xi = max(a / se, 0.0) ** 2
    p = 0.5 * float(1 - stats.chi2.cdf(xi, 1)) if xi > 0 else 1.0
    return dict(stat=float(xi), p=p, alpha=float(a), se=float(se), n_restrictions=1,
                n_segments=1, bench_held=[int(i) for i in idx], v_lo=1.0, v_hi=1.0)

# ============================ data pipeline ===========================

def _p(name):
    return os.path.join(DATA_DIR, name)

def _au_monthly(path):
    df = pd.read_csv(path)
    df["Rate Date"] = pd.to_datetime(df["Rate Date"], dayfirst=True)
    df = df.sort_values("Rate Date").set_index("Rate Date")
    cols = ["Australian Shares", "International Shares",
            "Diversified Fixed Interest", "Cash"]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    m = (1.0 + df[cols] / 100.0).cumprod().resample("ME").last().pct_change(fill_method=None)
    return m.rename(columns={"Diversified Fixed Interest": "Bonds"})

def _price_monthly(path, name, dayfirst):
    df = pd.read_csv(path, dtype={"Date": str})
    df["Date"] = pd.to_datetime(df["Date"].str.slice(0, 10) if not dayfirst
                                else df["Date"], dayfirst=dayfirst)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return (df.sort_values("Date").set_index("Date")["Close"]
              .resample("ME").last().pct_change(fill_method=None).rename(name))

def _art_unit_monthly(path, name):
    df = pd.read_csv(path, skiprows=1)
    df.columns = ["Date", "Price"]
    df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%d-%b-%y")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return (df.sort_values("Date").set_index("Date")["Price"]
              .resample("ME").last().pct_change(fill_method=None).rename(name))

def _truncate(M):
    M = M.copy(); M.index = M.index.to_period("M")
    return M[M.index <= pd.Period(SAMPLE_END, "M")] if SAMPLE_END else M

def _baubil_monthly(path):
    """Bloomberg AusBond Bank Bill Index monthly level -> monthly return.
    Used as the Cash / risk-free series for BOTH funds (it is the benchmark
    both funds' Cash options target), replacing the earlier practice of
    using AustralianSuper's own Cash option as an ART proxy."""
    df = pd.read_excel(path, sheet_name=0, header=0)
    df = df.iloc[:, :2]
    df.columns = ["Date", "Level"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Level"] = pd.to_numeric(df["Level"], errors="coerce")
    df = df.dropna().sort_values("Date")
    if df.empty:
        raise ValueError(f"{path}: no usable (date, level) rows found")
    s = df.set_index("Date")["Level"].resample("ME").last()
    r = s.pct_change(fill_method=None).rename("Cash")
    # Bank-bill returns are small and non-negative in this sample; a sign or
    # scale error in the source sheet would show up here rather than silently
    # contaminating every excess return in the study.
    chk = r.dropna()
    if not ((chk > -0.005) & (chk < 0.02)).all():
        bad = chk[(chk <= -0.005) | (chk >= 0.02)]
        raise ValueError(f"{path}: implausible monthly cash returns:\n{bad}")
    return r


def rf_diagnostics(funds):
    """Coverage/level check on the BAUBIL risk-free series, and a comparison
    with AustralianSuper's own Cash OPTION (the proxy used before BAUBIL).
    Both track the Bloomberg AusBond Bank Bill Index, so a high correlation
    with a small negative mean gap (the option is net of fees and tax) is the
    expected result, and confirms the switch is a like-for-like replacement."""
    baubil = _baubil_monthly(_p(CASH_FILE))
    au_opt = _au_monthly(_p(AU_FILE))["Cash"]
    b = baubil.copy(); b.index = b.index.to_period("M")
    o = au_opt.copy(); o.index = o.index.to_period("M")
    j = pd.concat([b.rename("BAUBIL"), o.rename("AuSuper Cash option")],
                  axis=1).dropna()
    j = j[j.index <= pd.Period(SAMPLE_END, "M")] if SAMPLE_END else j
    rows = [{"series": "BAUBIL (rf used everywhere)",
             "first": str(b.dropna().index.min()), "last": str(b.dropna().index.max()),
             "T": int(b.dropna().shape[0]),
             "mean_%pa": round(float(b.dropna().mean()) * 1200, 3)},
            {"series": "AuSuper Cash option (overlap)",
             "first": str(j.index.min()), "last": str(j.index.max()), "T": int(len(j)),
             "mean_%pa": round(float(j["AuSuper Cash option"].mean()) * 1200, 3)},
            {"series": "BAUBIL (same overlap)",
             "first": str(j.index.min()), "last": str(j.index.max()), "T": int(len(j)),
             "mean_%pa": round(float(j["BAUBIL"].mean()) * 1200, 3)}]
    corr = float(j["BAUBIL"].corr(j["AuSuper Cash option"]))
    gap = float((j["AuSuper Cash option"] - j["BAUBIL"]).mean()) * 1200
    # every fund frame must carry the same rf column, fully populated wherever
    # the benchmark assets are observed
    for fund, M in funds.items():
        sub = M.dropna(subset=BENCH)
        assert RF_COL in M.columns, f"{fund}: {RF_COL} missing"
        assert sub[RF_COL].notna().all(), f"{fund}: rf missing in {int(sub[RF_COL].isna().sum())} benchmark months"
        assert np.allclose(sub[RF_COL].values,
                           b.reindex(sub.index).values), f"{fund}: rf is not the BAUBIL series"
    return pd.DataFrame(rows), corr, gap


def _oos_span(M):
    """Number of OOS months and the matched-sample span used by run_backtest."""
    matched = M[M["Gold"].notna() & M["Bitcoin"].notna()].dropna(
        subset=BENCH + [RF_COL, "Gold", "Bitcoin"])
    return len(matched) - WINDOW, matched.index[0], matched.index[-1]


def _print_grid(g, suffixes, chunk=5):
    """Print a performance grid in lambda blocks; 10 lambdas x 3 portfolios is
    30 numeric columns, which is unreadable on one line."""
    base = ["Method", "Metric"]
    for i in range(0, len(LAMBDAS), chunk):
        blk = LAMBDAS[i:i + chunk]
        cols = base + [f"l={l} {sfx}" for l in blk for sfx in suffixes]
        print(f"\n  -- lambda {blk[0]}-{blk[-1]} " + "-" * 40)
        print(g[cols].to_string(index=False))


def build_funds():
    gold = _price_monthly(_p(GOLD_FILE), "Gold", dayfirst=True)
    btc  = _price_monthly(_p(BTC_FILE),  "Bitcoin", dayfirst=False)
    cash = _baubil_monthly(_p(CASH_FILE))
    au = _au_monthly(_p(AU_FILE)).drop(columns=["Cash"])
    ausuper = _truncate(pd.concat([au, cash, gold, btc], axis=1, sort=True))
    art = _truncate(pd.concat([
        _art_unit_monthly(_p(ART_AUS),  "Australian Shares"),
        _art_unit_monthly(_p(ART_INTL), "International Shares"),
        _art_unit_monthly(_p(ART_BOND), "Bonds"),
        cash, gold, btc], axis=1, sort=True))
    return {"AustralianSuper": ausuper, "ART": art}


# =============================== drivers ==============================

def _prep(sample_df, cand, excess):
    d  = sample_df[BENCH + [RF_COL, cand]].dropna()
    X  = d[BENCH].values.astype(float)
    rf = d[RF_COL].values.astype(float)
    y  = d[cand].values.astype(float)
    return (X - rf[:, None], y - rf, d, rf) if excess else (X, y, d, rf)

def _kz_row(fund, sample, cand, sample_df, excess):
    X, y, d, rf = _prep(sample_df, cand, excess)
    hk = hk_test(X, y); kz = kz_test(X, y)
    ell = gmm_wald_elliptical(X, y)
    a, t, p1 = dnw_one_sided(X, y)
    if excess:   # risk-free asset available: test at the bank-bill rate only
        dnw_new = dnw_spanning_rf(X, y, constrain_benchmark=False)
        dnw_all = dnw_spanning_rf(X, y, constrain_benchmark=True)
    else:        # gross returns: range of zero-beta rates (DNW 2001)
        dnw_new = dnw_spanning(X, y, constrain_benchmark=False, nsim=DNW_NSIM)
        dnw_all = dnw_spanning(X, y, constrain_benchmark=True,  nsim=DNW_NSIM)
    srb, _  = long_only_tangency_sharpe(d[BENCH].values - rf[:, None])
    srp, wp = long_only_tangency_sharpe(d[BENCH + [cand]].values - rf[:, None])
    hac6 = hac_spanning(X, y, HAC_ROBUST_LAG)
    return {
        "Fund": fund, "Sample": sample, "Asset": cand,
        "Returns": "excess" if excess else "raw", "T": kz["T"],
        "alpha_ann_%": round(float(kz["alpha_hat"][0]) * 1200, 2),
        "delta": round(float(kz["delta_hat"][0]), 4),
        # classical (OLS, iid-normal) inference
        "HK_joint_F": round(hk["F"], 4), "HK_joint_p": round(hk["p_value"], 4),
        "Wald_p": round(kz["p_W"], 4), "LR_p": round(kz["p_LR"], 4), "LM_p": round(kz["p_LM"], 4),
        "lambda1": round(kz["lambda1"], 6), "lambda2": round(kz["lambda2"], 6),
        "stepdown_F1_alpha": round(kz["F_Z"], 4), "stepdown_F2_delta_given_alpha": round(kz["F_K"], 4),
        "df_F1": str(kz["df_Z"]), "df_F2": str(kz["df_K"]),
        "GMM_elliptical_Wa_e": round(ell["Wa_e"], 4), "GMM_elliptical_p": round(ell["p"], 4),
        "kurtosis_kappa": round(ell["kappa"], 4),
        "DNW_nssNew_stat": round(dnw_new["stat"], 4), "DNW_nssNew_p": round(dnw_new["p"], 4),
        "DNW_nssAll_stat": round(dnw_all["stat"], 4), "DNW_nssAll_p": round(dnw_all["p"], 4),
        "DNW_segments": dnw_all["n_segments"], "DNW_restrictions": dnw_all["n_restrictions"],
        "DNW_v_lo": round(dnw_all["v_lo"], 6),
        "tangency_p_classical": round(kz["p_Z"], 4),      # KZ eq (46), alpha
        "GMV_p_classical": round(kz["p_K"], 4),           # KZ eq (47), delta | alpha
        "noShort_1sided_p_classical": round(p1, 4),
        # HAC(6) robustness
        f"HAC{HAC_ROBUST_LAG}_joint_p": round(hac6["joint_p"], 4),
        f"HAC{HAC_ROBUST_LAG}_tangency_p": round(hac6["tangency_p_two"], 4),
        f"HAC{HAC_ROBUST_LAG}_noShort_1sided_p": round(hac6["tangency_p_one"], 4),
        f"HAC{HAC_ROBUST_LAG}_GMV_p": round(hac6["gmv_p"], 4),
        f"HAC{HAC_ROBUST_LAG}_alpha_t": round(hac6["alpha_t"], 3),
        # long-only economic significance
        "LO_Sharpe_bench": round(srb, 4), "LO_Sharpe_plus": round(srp, 4),
        "cand_weight_%": round(float(wp[-1]) * 100, 2),
    }

def run_kz(funds, excess):
    rows, sens = [], []
    for fund, M in funds.items():
        matched  = M[M["Gold"].notna() & M["Bitcoin"].notna()]
        goldfull = M[M["Gold"].notna()]
        for sample, sdf, cand in [("Matched", matched, "Gold"),
                                   ("Matched", matched, "Bitcoin"),
                                   ("Gold full", goldfull, "Gold")]:
            rows.append(_kz_row(fund, sample, cand, sdf, excess))
            X, y, _, _ = _prep(sdf, cand, excess)
            for L in HAC_LAGS:
                h = hac_spanning(X, y, L)
                sens.append({"Fund": fund, "Sample": sample, "Asset": cand, "lag": L,
                    "role": "robustness" if L == HAC_ROBUST_LAG else "sensitivity",
                    "joint_p": round(h["joint_p"], 4),
                    "tangency_two_p": round(h["tangency_p_two"], 4),
                    "noShort_1sided_p": round(h["tangency_p_one"], 4),
                    "GMV_p": round(h["gmv_p"], 4),
                    "alpha_t": round(h["alpha_t"], 3)})
    return pd.DataFrame(rows), pd.DataFrame(sens)


def corner_diagnostics(funds):
    """Fix #3 diagnostic. At MONTHLY frequency the mean term (~0.008) swamps
    (lambda/2) x'Sigma x (~0.0005*lambda), so low-lambda quadratic-utility
    portfolios are corner solutions (effectively 100% in the highest-mean
    asset), not interior mean-variance portfolios. This reports, for the
    Markowitz BENCHMARK portfolio, the mean weights and the share of OOS
    months at a corner (max weight > 0.99), so the low-lambda columns of the
    performance grids can be read for what they are."""
    rows = []
    for fund, M in funds.items():
        matched = M[M["Gold"].notna() & M["Bitcoin"].notna()].dropna(
            subset=BENCH + [RF_COL]).reset_index(drop=True)
        EX = matched[BENCH].values - matched[RF_COL].values[:, None]
        T = len(matched)
        for lam in LAMBDAS_DIAG:
            ws = np.array([strategy_weights("Markowitz", EX[:t], lam)
                           for t in range(WINDOW, T)])
            rows.append({"Fund": fund, "lambda": lam,
                         **{f"mean_w_{c}": round(float(ws[:, i].mean()), 4)
                            for i, c in enumerate(BENCH)},
                         "pct_months_at_corner": round(float((ws.max(1) > 0.99).mean() * 100), 1),
                         "interior": (ws.max(1) > 0.99).mean() < 0.5})
    return pd.DataFrame(rows)


def run_oos(funds):
    grids, weights, long_rows = {}, [], []
    for fund, M in funds.items():
        matched = M[M["Gold"].notna() & M["Bitcoin"].notna()].dropna(
            subset=BENCH + [RF_COL, "Gold", "Bitcoin"]).reset_index(drop=True)
        og, wg, rf = run_backtest(matched[BENCH + [RF_COL, "Gold"]].copy(), "Gold")
        ob, wb, _  = run_backtest(matched[BENCH + [RF_COL, "Bitcoin"]].copy(), "Bitcoin")
        rows = []
        for strat in STRATEGIES:
            for mname, fn in METRICS:
                row = {"Fund": fund, "Method": strat, "Metric": mname}
                for lam in LAMBDAS:
                    row[f"l={lam} Bench"] = fn(og[(strat, lam, "bench")], rf)
                    row[f"l={lam} +Gold"] = fn(og[(strat, lam, "plus")], rf)
                    row[f"l={lam} +BTC"]  = fn(ob[(strat, lam, "plus")], rf)
                rows.append(row)
        grids[fund] = pd.DataFrame(rows).round(3)
        long_rows.extend(rows)
        for cand, ws in (("Gold", wg), ("Bitcoin", wb)):
            for strat in STRATEGIES:
                weights.append({"Fund": fund, "Candidate": cand, "Strategy": strat,
                    **{f"l={lam}": round(ws[(strat, lam, "plus")][:, 3].mean(), 4)
                       for lam in LAMBDAS}})
    return grids, pd.DataFrame(long_rows).round(3), pd.DataFrame(weights)


def _st(p):
    return ("***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "." if p < .10 else "")


# =================== weight-cap robustness (new) ======================
# (i) OOS with the candidate capped; (ii) spanning-side capped tangency
#     Sharpe improvement with a moving-block bootstrap p-value.

def optimise_capped(mu, Sigma, lam, floor, cap_last):
    """Long-only quadratic-utility optimiser with the last asset (candidate)
    capped at cap_last. If a Gens floor exceeds the cap, the candidate is
    pinned at the cap (lb = ub = cap)."""
    N = len(mu)
    bounds = [(floor, 1.0)] * N
    if cap_last is not None:
        bounds[-1] = (min(floor, cap_last), cap_last)
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0}]
    obj = lambda x: -(x @ mu - lam / 2.0 * x @ Sigma @ x)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    corner = np.clip(lo.copy(), None, hi); j = int(np.argmax(mu))
    corner[j] = min(hi[j], 1.0 - (lo.sum() - lo[j]))
    starts = [np.full(N, 1.0 / N), corner]          # match optimise()'s multi-start
    best, bv = None, np.inf
    for x0 in starts:
        r = minimize(obj, x0, method="SLSQP", bounds=bounds,
                     constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
        if r.success and r.fun < bv:
            best, bv = r.x, r.fun
    if best is None:
        best = np.full(N, 1.0 / N)
    w = np.clip(best, 0.0, None)
    if cap_last is not None:                       # exact projection onto the cap
        w[-1] = min(max(w[-1], min(floor, cap_last)), cap_last)
        rem = 1.0 - w[-1]; wb = np.clip(w[:-1], floor, None); w[:-1] = wb / wb.sum() * rem
    else:
        w = np.clip(w, floor, None); w = w / w.sum()
    return w

def strategy_weights_capped(strategy, EX, lam, cap_last):
    mu = EX.mean(axis=0); Sigma = np.cov(EX, rowvar=False, ddof=1)
    N, T = len(mu), EX.shape[0]
    floor = 1.0 / (2 * N) if "Gens" in strategy else 0.0
    if strategy.startswith("Bayes-Stein"):
        mu, Sigma = bayes_stein(mu, Sigma, T)
    elif strategy == "Black-Litterman":
        mu, Sigma = black_litterman(mu, Sigma, lam)
    return optimise_capped(mu, Sigma, lam, floor, cap_last)

def run_backtest_capped(data, candidate, cap):
    """As run_backtest, but the candidate weight is capped at `cap` in the
    augmented ('plus') portfolios. The benchmark portfolios are unaffected."""
    bench = ["Australian Shares", "International Shares", "Bonds"]
    plus = bench + [candidate]
    R_b, R_p = data[bench].values, data[plus].values
    rf = data[RF_COL].values                       # BAUBIL monthly risk-free
    EX_b, EX_p = R_b - rf[:, None], R_p - rf[:, None]
    T = len(data)
    results, weights = {}, {}
    for strat in STRATEGIES:
        for lam in LAMBDAS:
            for tag, R, EX, cp in (("bench", R_b, EX_b, None), ("plus", R_p, EX_p, cap)):
                rets, ws = [], []
                for t in range(WINDOW, T):
                    w = strategy_weights_capped(strat, EX[:t], lam, cp)
                    rets.append(w @ R[t]); ws.append(w)
                results[(strat, lam, tag)] = np.array(rets)
                weights[(strat, lam, tag)] = np.array(ws)
    return results, weights, rf[WINDOW:]


def capped_tangency_sharpe(EX, cap=None):
    """Annualised max Sharpe over long-only, fully-invested weights with the
    last asset capped at `cap`. Solved as a convex QP:
        min y'Sigma y  s.t.  mu'y = 1, y >= 0, y_last <= cap*sum(y);
        w = y/sum(y),  Sharpe = sqrt(12)/sqrt(y'Sigma y)."""
    mu = EX.mean(axis=0); S = np.cov(EX, rowvar=False, ddof=1); N = len(mu)
    if mu.max() <= 0:
        return np.nan, np.full(N, np.nan)
    cons = [{"type": "eq", "fun": lambda y: mu @ y - 1.0}]
    if cap is not None and cap < 1.0:
        cons.append({"type": "ineq",
                     "fun": lambda y: cap * (y.sum() - y[-1]) - (1 - cap) * y[-1]})
    r = minimize(lambda y: y @ S @ y, np.ones(N) / N, method="SLSQP",
                 bounds=[(0, None)] * N, constraints=cons,
                 options={"maxiter": 500, "ftol": 1e-12})
    y = np.clip(r.x, 0, None); w = y / y.sum()
    return float((mu @ w) / np.sqrt(w @ S @ w) * np.sqrt(12)), w

def _mbb_idx(T, block, rng):
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=nb)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:T]

def bootstrap_capped_spanning(EXb, EXp, cap, B, block, seed):
    """Moving-block bootstrap for the tangency-Sharpe improvement.

    TWO CORRECTIONS relative to the earlier version:

    (1) The BENCHMARK leg is never capped. The cap applies to the candidate
        only; the benchmark has no candidate to cap. Previously the benchmark
        was called with the same cap, which capped its LAST column (Bonds),
        producing a crippled comparison and, in the gold-full samples where
        the benchmark holds 49-81% Bonds, a "capped" gain larger than the
        uncapped one (algebraically impossible).

    (2) The p-value is RECENTRED on the null. Delta* >= 0 in every resample by
        construction, because the augmented feasible set contains the
        benchmark's (candidate weight 0 is always feasible). Counting
        Delta* <= 0 therefore only counts floating-point ties and can never
        reject. The recentred p-value is
              p = #{ Delta* - Delta_hat >= Delta_hat } / B,
        i.e. the bootstrap distribution is shifted to have mean zero under H0.
    """
    rng = np.random.default_rng(seed)
    sr_b   = capped_tangency_sharpe(EXb, None)[0]          # benchmark: never capped
    d_unc  = capped_tangency_sharpe(EXp, None)[0] - sr_b
    d_cap  = capped_tangency_sharpe(EXp, cap)[0]  - sr_b
    T = EXb.shape[0]
    du_star, dc_star = [], []
    for _ in range(B):
        idx = _mbb_idx(T, block, rng)
        Eb, Ep = EXb[idx], EXp[idx]
        srb_s = capped_tangency_sharpe(Eb, None)[0]
        du_star.append(capped_tangency_sharpe(Ep, None)[0] - srb_s)
        dc_star.append(capped_tangency_sharpe(Ep, cap)[0]  - srb_s)
    du_star = np.asarray(du_star); dc_star = np.asarray(dc_star)
    p_unc = (1 + int(np.sum(du_star - d_unc >= d_unc))) / (B + 1)
    p_cap = (1 + int(np.sum(dc_star - d_cap >= d_cap))) / (B + 1)
    return {"d_unc": d_unc, "p_unc": p_unc, "d_cap": d_cap, "p_cap": p_cap,
            "mean_du_star": float(du_star.mean()), "mean_dc_star": float(dc_star.mean()),
            "naive_p_unc": (1 + int(np.sum(du_star <= 0))) / (B + 1)}


def run_oos_capped(funds, cap):
    grids, long_rows, weights = {}, [], []
    for fund, M in funds.items():
        matched = M[M["Gold"].notna() & M["Bitcoin"].notna()].dropna(
            subset=BENCH + [RF_COL, "Gold", "Bitcoin"]).reset_index(drop=True)
        og, wg, rf = run_backtest_capped(matched[BENCH + [RF_COL, "Gold"]].copy(), "Gold", cap)
        ob, wb, _  = run_backtest_capped(matched[BENCH + [RF_COL, "Bitcoin"]].copy(), "Bitcoin", cap)
        rows = []
        for strat in STRATEGIES:
            for mname, fn in METRICS:
                row = {"Fund": fund, "Method": strat, "Metric": mname}
                for lam in LAMBDAS:
                    row[f"l={lam} Bench"]        = fn(og[(strat, lam, "bench")], rf)
                    row[f"l={lam} +Gold(cap)"]   = fn(og[(strat, lam, "plus")], rf)
                    row[f"l={lam} +BTC(cap)"]    = fn(ob[(strat, lam, "plus")], rf)
                rows.append(row)
        grids[fund] = pd.DataFrame(rows).round(3)
        long_rows.extend(rows)
        for cand, ws in (("Gold", wg), ("Bitcoin", wb)):
            for strat in STRATEGIES:
                weights.append({"Fund": fund, "Candidate": cand, "Strategy": strat,
                    **{f"l={lam}": round(ws[(strat, lam, "plus")][:, 3].mean(), 4)
                       for lam in LAMBDAS}})
    return grids, pd.DataFrame(long_rows).round(3), pd.DataFrame(weights)

def run_spanning_capped(funds, cap):
    rows = []
    for fund, M in funds.items():
        matched  = M[M["Gold"].notna() & M["Bitcoin"].notna()]
        goldfull = M[M["Gold"].notna()]
        for sample, sdf, cand in [("Matched", matched, "Gold"),
                                   ("Matched", matched, "Bitcoin"),
                                   ("Gold full", goldfull, "Gold")]:
            d  = sdf[BENCH + [RF_COL, cand]].dropna()
            rf = d[RF_COL].values
            EXb = d[BENCH].values - rf[:, None]
            EXp = d[BENCH + [cand]].values - rf[:, None]
            srb  = capped_tangency_sharpe(EXb, None)[0]
            sru, wu = capped_tangency_sharpe(EXp, None)
            src, wc = capped_tangency_sharpe(EXp, cap)
            bt = bootstrap_capped_spanning(EXb, EXp, cap, BOOT_B, BOOT_BLOCK, BOOT_SEED)
            rows.append({
                "Fund": fund, "Sample": sample, "Asset": cand, "T": len(d),
                "cap_%": round(cap * 100, 1),
                "bench_tangSharpe": round(srb, 4),
                "uncapped_tangSharpe": round(sru, 4), "uncapped_wt_%": round(wu[-1] * 100, 2),
                "uncapped_dSharpe": round(bt["d_unc"], 4), "uncapped_boot_p": round(bt["p_unc"], 4),
                "capped_tangSharpe": round(src, 4), "capped_wt_%": round(wc[-1] * 100, 2),
                "capped_dSharpe": round(bt["d_cap"], 4), "capped_boot_p": round(bt["p_cap"], 4),
                "mean_boot_dSharpe_unc": round(bt["mean_du_star"], 4),
                "naive_uncentred_p": round(bt["naive_p_unc"], 4),
            })
    return pd.DataFrame(rows)


def main():
    funds = build_funds()
    excess = (RETURNS == "excess")   # both permitted (KZ footnote 2)

    rf_tab, rf_corr, rf_gap = rf_diagnostics(funds)
    rf_tab.to_csv("RiskFree_BAUBIL_diagnostics.csv", index=False)
    print("=" * 96)
    print("RISK-FREE RATE - BAUBIL (Bloomberg AusBond Bank Bill Index), used for BOTH funds")
    print("in every excess return: spanning regressions, tangency Sharpes, OOS optimiser")
    print("inputs, and the reported Sharpe / Sortino ratios.")
    print("=" * 96)
    print(rf_tab.to_string(index=False))
    print(f"\nBAUBIL vs AustralianSuper Cash option over the overlap: corr={rf_corr:.4f}, "
          f"mean gap (option - BAUBIL) = {rf_gap:+.3f}% p.a.")
    print(f"Risk-aversion grid: lambda = {LAMBDAS}")

    kz, sens = run_kz(funds, excess)
    kz.to_csv("KZ_spanning_ALL.csv", index=False)
    sens.to_csv("HAC_sensitivity.csv", index=False)

    print("=" * 96)
    print("SPANNING TESTS - Huberman-Kandel (1987) / Kan-Zhou (2012)  |  K=3 = {Australian Shares, International Shares, Bonds}")
    print(f"Cash = BAUBIL (Bloomberg AusBond Bank Bill), both funds | returns={RETURNS} | HAC robustness lag={HAC_ROBUST_LAG}, "
          f"sensitivity lags={HAC_SENS_LAGS}")
    print("=" * 96)
    for _, r in kz.iterrows():
        print(f"\n-- {r['Fund']:15s} {r['Asset']:8s} | {r['Sample']:9s} | T={int(r['T'])} | "
              f"alpha={r['alpha_ann_%']:6.2f}% p.a.")
        print(f"   HK/KZ      joint HK p={r['HK_joint_p']:.4f}{_st(r['HK_joint_p'])} | "
              f"W/LR/LM p={r['Wald_p']:.4f}{_st(r['Wald_p'])}")
        print(f"   STEP-DOWN  F1 alpha (tangency) ={r['stepdown_F1_alpha']:7.3f} df{r['df_F1']} "
              f"p={r['tangency_p_classical']:.4f}{_st(r['tangency_p_classical'])}   "
              f"F2 delta|alpha (GMV) ={r['stepdown_F2_delta_given_alpha']:6.3f} df{r['df_F2']} "
              f"p={r['GMV_p_classical']:.4f}{_st(r['GMV_p_classical'])}")
        print(f"   GMM        elliptical Wa_e p={r['GMM_elliptical_p']:.4f}{_st(r['GMM_elliptical_p'])} "
              f"(kurtosis kappa={r['kurtosis_kappa']:.2f})")
        print(f"   DNW(2001)  no short sales of candidate p={r['DNW_nssNew_p']:.4f}{_st(r['DNW_nssNew_p'])} | "
              f"no short sales at all p={r['DNW_nssAll_p']:.4f}{_st(r['DNW_nssAll_p'])} "
              f"({r['DNW_segments']} segment(s), {r['DNW_restrictions']} inequality restrictions)")
        j6 = r[f"HAC{HAC_ROBUST_LAG}_joint_p"]; t6 = r[f"HAC{HAC_ROBUST_LAG}_tangency_p"]
        o6 = r[f"HAC{HAC_ROBUST_LAG}_noShort_1sided_p"]; g6 = r[f"HAC{HAC_ROBUST_LAG}_GMV_p"]
        print(f"   HAC({HAC_ROBUST_LAG})     joint p={j6:.4f}{_st(j6)} | tangency p={t6:.4f}{_st(t6)} | "
              f"GMV p={g6:.4f}{_st(g6)} | no-short 1-sided p={o6:.4f}{_st(o6)}")
        srow = sens[(sens.Fund == r["Fund"]) & (sens.Sample == r["Sample"]) & (sens.Asset == r["Asset"])]
        cells = "  ".join(f"L{int(x.lag)}:{x.noShort_1sided_p:.4f}{_st(x.noShort_1sided_p)}"
                          for _, x in srow.sort_values("lag").iterrows())
        print(f"   SENSITIVITY (no-short 1-sided p by lag)   {cells}")

    diag = corner_diagnostics(funds)
    diag.to_csv("Lambda_corner_diagnostics.csv", index=False)
    print("\n" + "=" * 96)
    print("FIX #3 DIAGNOSTIC — Markowitz BENCHMARK portfolio: corner vs interior by lambda")
    print("(monthly returns: mean term swamps the risk term, so low lambda => corner solutions)")
    print("=" * 96)
    print(diag.to_string(index=False))

    grids, oos_long, weights = run_oos(funds)
    oos_long.to_csv("OOS_results_ALL.csv", index=False)
    weights.to_csv("OOS_mean_weights_ALL.csv", index=False)
    with pd.ExcelWriter("OOS_results_ALL.xlsx") as xl:
        for fund, g in grids.items():
            g.drop(columns="Fund").to_excel(xl, sheet_name=fund[:31], index=False)
        weights.to_excel(xl, sheet_name="MeanWeights", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 200)
    for fund, g in grids.items():
        n_oos, first, last = _oos_span(funds[fund])
        print("\n" + "=" * 96)
        print(f"{fund} — OUT-OF-SAMPLE ({WINDOW}-mo expanding, {n_oos} OOS months, "
              f"matched sample {first}..{last}); Bench vs +Gold vs +BTC")
        print("=" * 96)
        _print_grid(g.drop(columns="Fund"), ["Bench", "+Gold", "+BTC"])

    print("\nSignificance: *** p<.001  ** p<.01  * p<.05  . p<.10")
    print("Estimation is OLS throughout; HAC changes only the inference covariance.")
    print("Step-down follows KZ eqs (46)-(47): F1 tests alpha, F2 tests delta given alpha.")
    print("KZ s.3.1: the JOINT tests have little power against assets that move only the")
    print("tangency portfolio (alpha) rather than the GMV portfolio (delta) - which is this")
    print("case - so read F1 alongside the joint p-values, per KZ's own recommendation.")
    print("Saved: KZ_spanning_ALL.csv, HAC_sensitivity.csv, OOS_results_ALL.csv, OOS_results_ALL.xlsx")

    # ---------------- weight-cap robustness ----------------
    if CANDIDATE_CAP is not None:
        cap = CANDIDATE_CAP
        # (i) spanning-side: capped tangency-Sharpe improvement + bootstrap p
        span_cap = run_spanning_capped(funds, cap)
        span_cap.to_csv("Spanning_CAPPED.csv", index=False)
        print("\n" + "=" * 96)
        print(f"SPANNING-SIDE {cap*100:.0f}% CAP  |  capped long-only tangency-Sharpe improvement")
        print("(one-sided moving-block bootstrap p for improvement > 0; "
              f"B={BOOT_B}, block={BOOT_BLOCK})")
        print("=" * 96)
        print("Note: an upper cap changes the SIZE of the frontier improvement, not the")
        print("sign-based rejection. The classical/HAC one-sided no-short test (H1: alpha>0)")
        print("is therefore unchanged by the cap; the cap-sensitive statistic is the")
        print("capped tangency-Sharpe improvement below.")
        for _, r in span_cap.iterrows():
            print(f"\n-- {r['Fund']:15s} {r['Asset']:8s} | {r['Sample']:9s} | T={int(r['T'])}")
            print(f"   bench SR={r['bench_tangSharpe']:.3f} | "
                  f"uncapped +{r['Asset']}: SR={r['uncapped_tangSharpe']:.3f} "
                  f"(wt {r['uncapped_wt_%']:.1f}%) dSR={r['uncapped_dSharpe']:+.3f} "
                  f"p={r['uncapped_boot_p']:.4f}{_st(r['uncapped_boot_p'])}")
            print(f"   {cap*100:.0f}% cap  +{r['Asset']}: SR={r['capped_tangSharpe']:.3f} "
                  f"(wt {r['capped_wt_%']:.1f}%) dSR={r['capped_dSharpe']:+.3f} "
                  f"p={r['capped_boot_p']:.4f}{_st(r['capped_boot_p'])}")

        # (ii) OOS with the candidate capped
        cgrids, coos_long, cweights = run_oos_capped(funds, cap)
        coos_long.to_csv("OOS_results_CAPPED.csv", index=False)
        cweights.to_csv("OOS_mean_weights_CAPPED.csv", index=False)
        with pd.ExcelWriter("OOS_results_CAPPED.xlsx") as xl:
            for fund, g in cgrids.items():
                g.drop(columns="Fund").to_excel(xl, sheet_name=fund[:31], index=False)
            cweights.to_excel(xl, sheet_name="MeanWeights", index=False)
        for fund, g in cgrids.items():
            print("\n" + "=" * 96)
            print(f"{fund} — OUT-OF-SAMPLE, candidate capped at {cap*100:.0f}%; "
                  f"Bench vs +Gold(cap) vs +BTC(cap)")
            print("=" * 96)
            _print_grid(g.drop(columns="Fund"), ["Bench", "+Gold(cap)", "+BTC(cap)"])
        print(f"\nSaved: Spanning_CAPPED.csv, OOS_results_CAPPED.csv, OOS_results_CAPPED.xlsx")


if __name__ == "__main__":
    main()
