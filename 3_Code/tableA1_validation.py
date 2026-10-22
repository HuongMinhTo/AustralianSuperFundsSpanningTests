# -*- coding: utf-8 -*-
"""Appendix Table A1: dataset-computed vs published 1-, 5- and 10-year returns
for the DIY-Mix options, at each fund's latest published month-end
(AustralianSuper: 30 June 2026; ART: 31 July 2026). Published figures were
retrieved from the funds' websites on 26 August 2026.
Output: TableA1_validation.csv
(Computation taken from refresh_table1.py, without the Word-editing step.)"""

import numpy as np
import pandas as pd

import replicate_all_CORRECTED as R

AU_END = pd.Period("2026-06", "M")
ART_END = pd.Period("2026-07", "M")

PUB = {   # (1yr, 5yr p.a., 10yr p.a.), as published
    ("AustralianSuper", "Australian Shares"):    (10.22, 9.66, 10.55),
    ("AustralianSuper", "International Shares"): (14.46, 9.51, 12.54),
    ("AustralianSuper", "Bonds"):                (2.12, 0.95, 2.00),
    ("ART", "Australian Shares"):                (6.07, 8.31, 9.17),
    ("ART", "International Shares"):             (11.25, 11.06, 12.52),
    ("ART", "Bonds"):                            (1.30, -0.28, 1.14),
}


def series():
    au = pd.read_csv(R._p(R.AU_FILE))
    au["Rate Date"] = pd.to_datetime(au["Rate Date"], dayfirst=True)
    au = au.sort_values("Rate Date").set_index("Rate Date")
    cols = ["Australian Shares", "International Shares", "Diversified Fixed Interest"]
    au[cols] = au[cols].apply(pd.to_numeric, errors="coerce")
    m = (1 + au[cols] / 100).cumprod().resample("ME").last().pct_change(fill_method=None)
    m.index = m.index.to_period("M")
    m = m.rename(columns={"Diversified Fixed Interest": "Bonds"})
    out = {("AustralianSuper", c): m[c] for c in ["Australian Shares", "International Shares", "Bonds"]}
    for c, f in [("Australian Shares", R.ART_AUS), ("International Shares", R.ART_INTL), ("Bonds", R.ART_BOND)]:
        s = R._art_unit_monthly(R._p(f), c)
        s.index = s.index.to_period("M")
        out[("ART", c)] = s
    return out


def perf(s, end, months):
    w = s[s.index <= end].dropna().iloc[-months:]
    g = float((1 + w).prod())
    return ((g - 1) if months == 12 else g ** (12.0 / months) - 1) * 100


if __name__ == "__main__":
    S = series()
    rows = []
    for key, pub in PUB.items():
        end = AU_END if key[0] == "AustralianSuper" else ART_END
        for (lab, n), p in zip([("1-year", 12), ("5-year (p.a.)", 60), ("10-year (p.a.)", 120)], pub):
            c = perf(S[key], end, n)
            rows.append(dict(Fund=key[0], Asset=key[1], Period=lab, Computed=round(c, 2), Published=p, Diff=round(c - p, 2)))
    df = pd.DataFrame(rows)
    df.to_csv("TableA1_validation.csv", index=False)
    print(df.to_string(index=False))
