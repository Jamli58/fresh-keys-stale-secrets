"""Recompute the generator's sector margin anchors from IRS Statistics of Income (public domain).

Source: IRS SOI, Returns of Active Corporations, Table 1 (selected income statement items
by minor industry), tax year 2022, file 22co01ccr.xlsx from
https://www.irs.gov/pub/irs-soi/22co01ccr.xlsx. Gross margin = 1 - cost of goods sold /
business receipts.

IT consulting is not taken from SOI: service corporations deduct most staff pay as
salaries and wages rather than cost of goods sold, so SOI's "Computer systems design and
related services" (0.69) overstates a consulting day's margin. config.py uses Damodaran's
Computer Services gross margin (NYU Stern, January 2026) instead; it is printed here for
reference only. Cross-check: IRS SOI nonfarm sole proprietorships, Table 2, tax year 2023
(23sp02is.xls), which runs higher because owners' own labor is not a cost.

    uv run --with openpyxl python scripts/sector_margins.py
"""

import logging
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("sector_margins")
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "soi" / "22co01ccr.xlsx"
NAME_COL, RECEIPTS_COL, COGS_COL = 0, 5, 6
FIRST_DATA_ROW = 7

# generator sector -> SOI minor industry (exact row label in Table 1)
SOI_ROWS = {
    "bakery_cafe": "Food services and drinking places",
    "machine_shop": "Cutlery, hardware, spring, and wire, machine shops; screw, nut, and bolt",
    "apparel_ecommerce": "Clothing and clothing accessories retailers",
    "construction_sub": "Specialty trade contractors",
    "specialty_retail": "Sporting goods, hobby, book, music, and miscellaneous retailers",
    "it_consulting (not used)": "Computer systems design and related services",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = pd.read_excel(RAW, header=None)
    rows = {str(df.iloc[i, NAME_COL]).strip(): i for i in range(FIRST_DATA_ROW, len(df))}
    for sector, label in SOI_ROWS.items():
        if label not in rows:
            raise SystemExit(f"row not found in {RAW.name}: {label!r}")
        i = rows[label]
        receipts, cogs = float(df.iloc[i, RECEIPTS_COL]), float(df.iloc[i, COGS_COL])
        LOG.info("%-26s gross margin %.3f  (%s)", sector, 1.0 - cogs / receipts, label)


if __name__ == "__main__":
    main()
