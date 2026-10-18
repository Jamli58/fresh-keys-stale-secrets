# Third-party raw data

The experiments run without these files: the generator ships its derived city table (`src/numpriv/data/cities_rpp2024.csv`), and the sector margin medians are constants in `src/numpriv/config.py`. The raw files are needed only to rebuild those derived inputs or to rerun the FinQA reconstruction check. Download them into `data/raw/`, which is git-ignored.

| File | Source | Used by |
|---|---|---|
| `data/raw/finqa_test.json` | FinQA test split, `dataset/test.json` of https://github.com/czyssrs/FinQA | `scripts/run_m0.py` (D2 reconstruction check, result `R003a.json`) |
| `data/raw/soi/22co01ccr.xlsx` | IRS Statistics of Income, corporation returns, tax year 2022, Table 1 (minor industries) | `scripts/sector_margins.py` (sector margin anchors) |
| `data/raw/marpp/MARPP_MSA_2008_2024.csv` | BEA Regional Price Parities by MSA, https://apps.bea.gov/regional/zip/MARPP.zip | `scripts/build_city_table.py` (city price levels) |

The IT-consulting margin anchor is the January 2026 Computer Services gross margin of Damodaran's industry margin tables (https://pages.stern.nyu.edu/~adamodar/), entered as a constant.
