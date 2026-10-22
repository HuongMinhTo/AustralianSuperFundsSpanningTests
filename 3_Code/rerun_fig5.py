# -*- coding: utf-8 -*-
"""Regenerate Figure 5 with the MSCI World ex-Australia international-shares
series, in the manuscript's existing visual style, and swap it into the
updated document in place."""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spanning_K7 as S7

S7.BENCH_FILES = [(c, ("MSCI_WorldIndex_ExAusAUD.xlsx"
                       if c == "International Shares" else f), role)
                  for c, f, role in S7.BENCH_FILES]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "-",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

OUT = os.environ.get("FIG_DIR", os.path.join("..", "figures"))   # 4_Results/figures when run from 4_Results/tables
os.makedirs(OUT, exist_ok=True)
PNG = os.path.join(OUT, "Figure5_downside_scatter.png")


def fig5():
    M = S7.build_panel()
    ref = "Australian Shares"
    panels = ["Gold", "Bitcoin", "International Shares", "Australian Property",
              "International Property", "Infrastructure",
              "Australian Fixed Interest", "Global Fixed Interest"]
    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    for ax, c in zip(axes.ravel(), panels):
        d = M[[ref, c]].dropna() * 100
        x, y = d[ref].values, d[c].values
        mid = (x <= -2) & (x > -5)
        low = x <= -5
        hi = x > -2
        ax.scatter(x[hi], y[hi], s=42, color="#8FB4D9", alpha=0.75,
                   edgecolor="#5588BB", linewidth=0.4, zorder=3,
                   label="Australian Shares > -2%")
        ax.scatter(x[mid], y[mid], s=52, color="#F5E63C", edgecolor="#B8A800",
                   linewidth=0.5, zorder=4,
                   label="-5% < Australian Shares \u2264 -2%")
        ax.scatter(x[low], y[low], s=58, color="#EE2222", edgecolor="#AA0000",
                   linewidth=0.5, zorder=5, label="Australian Shares \u2264 -5%")
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a + b * xs, color="#3778A8", lw=2, zorder=6)
        ax.axhline(0, color="#3778A8", lw=1.2, zorder=2)
        ax.axvline(-5, color="#EE2222", lw=1.6, zorder=2)
        ax.axvline(-2, color="#F5D000", lw=1.6, zorder=2)
        ax.set_title(c, fontsize=14)
        ax.set_xlabel("Australian Shares monthly return (%)", fontsize=11)
        ax.set_ylabel("%s monthly return (%%)" % c, fontsize=11)
        ax.text(0.02, 0.97,
                "Whole sample: N=%d, r=%.2f\nSlope=%.2f"
                % (len(d), np.corrcoef(x, y)[0, 1], b),
                transform=ax.transAxes, va="top", fontsize=11)
        ax.grid(alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=13,
               bbox_to_anchor=(0.5, -0.008), frameon=False)
    fig.suptitle("Asset Returns vs Australian Shares \u2014 Whole Sample\n"
                 "Yellow: Australian Shares between -2% and -5%; "
                 "Red: Australian Shares \u2264 -5%", fontsize=15, y=0.997)
    fig.tight_layout(rect=[0, 0.012, 1, 0.982])
    fig.savefig(PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", PNG)


def swap_into_doc(path):
    """Replace the image that immediately precedes the 'Figure 5.' caption."""
    import docx
    from docx.oxml.ns import qn

    d = docx.Document(path)
    paras = d.paragraphs
    cap = next(i for i, p in enumerate(paras)
               if p.text.strip().startswith("Figure 5."))
    rid = None
    for j in range(cap - 1, max(cap - 6, -1), -1):
        blips = paras[j]._element.findall(".//" + qn("a:blip"))
        if blips:
            rid = blips[-1].get(qn("r:embed"))
            break
    if rid is None:
        raise RuntimeError("no image found before the Figure 5 caption")
    part = d.part.related_parts[rid]
    with open(PNG, "rb") as fh:
        part._blob = fh.read()
    d.save(path)
    print("swapped Figure 5 image (rId=%s) into %s" % (rid, path))


if __name__ == "__main__":
    fig5()   # writes figure5_scatter.png (Figure 5 in the manuscript)
