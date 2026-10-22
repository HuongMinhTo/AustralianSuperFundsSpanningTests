# Do Gold and Bitcoin Expand the Opportunity Set for Australian Superannuation Members?

Author: Minh Huong To (Deakin University). Folder last organised: 15 October 2026.

**Repository contents.** This repository contains the code (`3_Code/`), the result files (`4_Results/`) and this README. The data (`2_Data/`), the manuscript files and the `supplementary/` folder are not included. The data come from fund websites and licensed third-party sources and are not redistributed; to reproduce the results, obtain the series listed under *Data* below and place them in `2_Data/` with the file names shown. `oos_paths.pkl` is not included either; `oos_paths.py` recreates it.

## Folder structure

| Folder | Contents |
|---|---|
| `1_Manuscript/` | Current manuscript v11 (clean and tracked-changes versions) and the supervisor comments-and-responses table. The same three files are in `latest_reports/`. |
| `2_Data/` | Raw input data used by the code (fund daily unit prices, gold, Bitcoin, BAUBIL, and the seven Balanced-option benchmark indices). |
| `3_Code/` | Python scripts that reproduce every table in v11 and Figure 5. |
| `4_Results/tables/` | Output files behind the tables in v11. |
| `4_Results/figures/` | Figures 1–5 exactly as they appear in v11 (separate files for journal submission). |
| `supplementary/` | Everything not needed to reproduce the v11 results: earlier manuscript versions, document-editing scripts, superseded or dropped analyses, logs, reference papers and background data. |

## Data (`2_Data/`)

Not included in this repository. File names expected by the code:

| File | Series | Used for |
|---|---|---|
| `AustralianSuper_Daily Rates 01 Jul 2008 - 21 Aug 2026.csv` | AustralianSuper DIY-Mix and diversified option unit prices | Main benchmark, Table A1 |
| `ART_AustralianShares_Daily_Rates.csv`, `ART_InternationalSharesUnhedged_Daily_Rates.csv`, `ART_FixedIncome_Daily_Rates.csv` | ART single-asset-class options | Main benchmark, Table A1 |
| `GOLDAX_split_adjusted.csv` | Global X Physical Gold (ASX: GOLD), split-adjusted | Gold |
| `BTCAUD.csv` | BTC–AUD (Yahoo Finance / CoinMarketCap) | Bitcoin |
| `BAUBIL.xlsx` | Bloomberg AusBond Bank Bill Index | Risk-free rate |
| `ASA52.xlsx`, `MSCI_WorldIndex_ExAusAUD.xlsx`, `ASA6PROP.xlsx`, `RAHRSAH.xlsx`, `FDCICSAH.xlsx`, `BACMO.xlsx`, `LEGATRAH.xlsx` | Australian shares, international shares (MSCI World ex-Australia), Australian property, international property, infrastructure, Australian and global fixed interest | Broader Balanced-option benchmark (K = 6, 7) |

## How to reproduce the results

Requires Python 3 with numpy, pandas, scipy, statsmodels, openpyxl and matplotlib. Run every script from `4_Results/tables/`:

```bash
cd 4_Results/tables
export DATA_DIR=../../2_Data PYTHONPATH=../../3_Code

python3 ../../3_Code/replicate_all_CORRECTED.py   # out-of-sample results: Table 4, Tables B1-B2 (OOS_results_ALL.*)
python3 ../../3_Code/table3_excess.py             # Table 3 and the DIY-Mix rows of Table 11
python3 ../../3_Code/table2_descriptives.py       # Table 2
python3 ../../3_Code/oos_paths.py                 # month-by-month OOS weights (oos_paths.pkl)
python3 ../../3_Code/tableB3_weights.py           # Appendix Table B3
python3 ../../3_Code/table4_summary.py            # Table 4 (from oos_paths.pkl)
python3 ../../3_Code/rerun_MSCI_excess.py         # Tables 7-11 (broader benchmark, K = 6 and K = 7)
python3 ../../3_Code/rerun_fig5.py                # Figure 5 (written to 4_Results/figures)
python3 ../../3_Code/tableA1_validation.py        # Appendix Table A1 (data validation)
```

`replicate_all_CORRECTED.py` and `spanning_K7.py` are also imported as modules by the other scripts. Run on its own, `replicate_all_CORRECTED.py` also writes total-return spanning results (`KZ_spanning_ALL.csv`, `HAC_sensitivity.csv`); these are not used in the paper, whose spanning results come from `table3_excess.py` and `rerun_MSCI_excess.py`. `spanning_K7.py` is used only as a module.

## Where each result comes from

| Manuscript item | Script | Output file (`4_Results/tables/`) |
|---|---|---|
| Table 2 | `table2_descriptives.py` | `Table2_recomputed.csv`, `Table5_Table2_recomputed.xlsx` |
| Table 3 | `table3_excess.py` | `KZ_spanning_EXCESS.csv` |
| Table 4 | `oos_paths.py`, `table4_summary.py` | `Table4_summary.csv`, `oos_summary.json` |
| Tables B1–B2 | `replicate_all_CORRECTED.py` | `OOS_results_ALL.csv`, `OOS_results_ALL.xlsx` |
| Table B3; λ = 1 corner allocations in Section 5.2 | `oos_paths.py`, `tableB3_weights.py`; `replicate_all_CORRECTED.py` | `TableB3_OOS_weights_full.csv`; `Lambda_corner_diagnostics.csv` |
| Table 7 | `rerun_MSCI_excess.py` | `MSCIx_descriptives.csv` |
| Table 8, Figure 5 | `rerun_MSCI_excess.py`, `rerun_fig5.py` | `MSCIx_downside.csv`, `MSCIx_downside_matched.csv` |
| Table 9 | `rerun_MSCI_excess.py` | `MSCIx_spanning.csv` |
| Table 10 | `rerun_MSCI_excess.py` | `MSCIx_dnw.csv` |
| Table 11 | `table3_excess.py`, `rerun_MSCI_excess.py` | `HAC_sensitivity_EXCESS.csv`, `MSCIx_hac_sensitivity.csv` |
| Table A1 | `tableA1_validation.py` | `TableA1_validation.csv` |
| Section 3.1 risk-free rate discussion | `replicate_all_CORRECTED.py` | `RiskFree_BAUBIL_diagnostics.csv` |

Notes
- All spanning tests, including the short-sale-constrained De Roon–Nijman–Werker test, use monthly returns in excess of BAUBIL. Because BAUBIL is available as a risk-free asset, the constrained test is evaluated at the bank-bill rate (`dnw_spanning_rf` in `replicate_all_CORRECTED.py`): one inequality restriction, p-value = half the χ²(1) tail probability. The earlier range-of-zero-beta-rates version (`dnw_spanning`) is kept for gross returns only.
- Table 4 is computed from the unrounded out-of-sample returns, not from the rounded values in Tables B1–B2.
- Tables 1, 5 and 6 are compiled from fund documents (investment objectives, Balanced-option allocation, index proxies) and need no code.
- Figures 1–4 in `4_Results/figures/` are the published versions taken from the manuscript; the scripts that originally drew them are not in this folder.
