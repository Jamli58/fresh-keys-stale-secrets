"""Derive the generator's city table from BEA Regional Price Parities (public domain).

Source: U.S. Bureau of Economic Analysis, Regional Price Parities by Metropolitan
Statistical Area, table MARPP (LineCode 1, "RPPs: All items"), 2024 values,
file MARPP_MSA_2008_2024.csv from https://apps.bea.gov/regional/zip/MARPP.zip
(last updated 2026-02-19). US average = 100.

Selection (fixed before M2): sort all MSAs with a 2024 value by RPP and take
N_CITIES evenly spaced ranks, so the list spans the whole price-level range.
"""

import csv
import logging
from pathlib import Path

LOG = logging.getLogger("cities")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "marpp" / "MARPP_MSA_2008_2024.csv"
OUT = ROOT / "src" / "numpriv" / "data" / "cities_rpp2024.csv"
N_CITIES = 60
MSA_SUFFIX = " (Metropolitan Statistical Area)"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rows = [r for r in csv.DictReader(RAW.open(encoding="latin-1")) if r.get("Description")]
    msas = sorted(
        (float(r["2024"]), r["GeoFIPS"].strip().strip('"'), r["GeoName"].replace(MSA_SUFFIX, "").strip())
        for r in rows
        if r["LineCode"].strip() == "1" and r["GeoName"].endswith(MSA_SUFFIX) and r["2024"].strip() not in ("", "(NA)")
    )
    step = (len(msas) - 1) / (N_CITIES - 1)
    picked = [msas[round(i * step)] for i in range(N_CITIES)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_fips", "city", "rpp_all_items_2024"])
        for rpp, fips, name in picked:
            w.writerow([fips, name, f"{rpp:.3f}"])
    LOG.info("wrote %d cities (RPP %.1f-%.1f) from %d MSAs to %s", len(picked), picked[0][0], picked[-1][0], len(msas), OUT)


if __name__ == "__main__":
    main()
