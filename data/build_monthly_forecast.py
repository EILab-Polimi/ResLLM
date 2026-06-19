#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a near-term monthly inflow-forecast file from CNRFC HEFS volume stats.

The committed water-year hindcast (``data/FOLC1_wy_hindcast.csv``) was produced by
``_archive/_cnrfc_forecast/compile_wy_forecasts.py``, which drops all monthly-horizon
``QCMFH*`` columns and keeps only each issue row's seasonal (``QCSFH*``) and water-year
(``QCYFH*``) residual volumes. This script recovers the monthly horizon: for every
forecast issue date it reads the raw HEFS stats file and extracts a near-term inflow
outlook (the mean monthly volume summed over the next ``N`` upcoming months, plus the next
single month's mean and dry/wet tails).

The near-term outlook drives a winter (flood-season) forecast-based flood-control curve,
mirroring CalSim-3's ``OctMarRunoffEst`` storm-risk logic — distinct from the
remaining-water-year forecast (``QCYFHM``) that drives the slow constraints (water-year
type, carryover, recession flood curve).

Forecast-stat code legend (from ``_archive/_cnrfc_forecast/data/README_Hindcast.docx``):
``QCMFH*`` = monthly volume; suffix ``M``=mean, ``5``=median, ``C``/``W``=min/max,
``1``/``G``/``H``/``9`` = 90/75/25/10% exceedance (``1`` is the dry/low tail, ``9`` the
wet/high tail), all in TAF.

In each raw stats file (read with ``header=1``) the first column holds the row's month
timestamp; ``iloc[0]`` is the issue date (its monthly columns are 0 — the current month
is "now") and ``iloc[1:]`` is the monthly trajectory (one row per upcoming month-start).
Percentile volumes are not additive, so only the mean (``QCMFHM``) is summed across
months; the tails are reported for the next single month only.

Inputs
------
- ``_archive/_cnrfc_forecast/data/FOLC1_vol_stats/YYYYMMDD12_FOLC1F_hefs_csv_stats.csv``
  (gitignored archive; one-time source — re-runnable only where the archive exists)

Outputs
-------
- ``data/FOLC1_monthly_forecast.csv`` (committed) — daily rows 1989-10-01 … 2023-09-30
  with columns: ``date, nt1_mean, nt3_mean, nt1_p10, nt1_p90``.

Usage
-----
python data/build_monthly_forecast.py [--horizon-months 3]
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Repo-root-relative paths (this file lives at data/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
ARCHIVE_DIR = os.path.join(
    _REPO_ROOT, "_archive", "_cnrfc_forecast", "data", "FOLC1_vol_stats"
)
OUTPUT_FILE = os.path.join(_REPO_ROOT, "data", "FOLC1_monthly_forecast.csv")

FILE_SUFFIX = "12_FOLC1F_hefs_csv_stats.csv"  # issue hour is 12z
START_DATE = datetime(1989, 10, 1)
END_DATE = datetime(2023, 9, 30)
MISSING_SENTINEL = -999.0


def _near_term(df: pd.DataFrame, horizon_months: int) -> dict[str, float]:
    """Extract the near-term monthly outlook from one issue file's frame.

    Parameters:
        df: Raw stats frame read with ``header=1``; ``iloc[0]`` is the issue row and
            ``iloc[1:]`` are upcoming month rows.
        horizon_months: Number of upcoming months ``N`` to sum for ``nt3_mean``.

    Returns:
        Dict with ``nt1_mean``, ``nt{N}_mean`` aliased as ``nt3_mean``, ``nt1_p10``,
        ``nt1_p90`` (NaN where unavailable).
    """
    future = df.iloc[1:].copy()
    for col in ("QCMFHM", "QCMFH1", "QCMFH9"):
        if col in future.columns:
            future[col] = future[col].replace(MISSING_SENTINEL, np.nan)

    out = {"nt1_mean": np.nan, "nt3_mean": np.nan, "nt1_p10": np.nan, "nt1_p90": np.nan}
    if future.empty or "QCMFHM" not in future.columns:
        return out

    means = pd.to_numeric(future["QCMFHM"], errors="coerce").to_numpy()
    head = means[:horizon_months]
    head = head[np.isfinite(head)]
    if head.size:
        out["nt3_mean"] = float(head.sum())

    if np.isfinite(means[0]):
        out["nt1_mean"] = float(means[0])
    if "QCMFH1" in future.columns:
        v = pd.to_numeric(future["QCMFH1"].iloc[0], errors="coerce")
        out["nt1_p10"] = float(v) if np.isfinite(v) else np.nan
    if "QCMFH9" in future.columns:
        v = pd.to_numeric(future["QCMFH9"].iloc[0], errors="coerce")
        out["nt1_p90"] = float(v) if np.isfinite(v) else np.nan
    return out


def build(horizon_months: int = 3) -> pd.DataFrame:
    """Compile the daily near-term monthly outlook across the hindcast period."""
    if not os.path.isdir(ARCHIVE_DIR):
        raise FileNotFoundError(
            f"Archive source not found: {ARCHIVE_DIR}\n"
            "This one-time build needs the gitignored CNRFC stats files."
        )

    records: list[dict] = []
    missing = 0
    current = START_DATE
    while current <= END_DATE:
        path = os.path.join(ARCHIVE_DIR, current.strftime("%Y%m%d") + FILE_SUFFIX)
        rec: dict = {"date": current}
        if os.path.exists(path):
            df = pd.read_csv(path, header=1)
            if not df.empty:
                rec.update(_near_term(df, horizon_months))
            else:
                missing += 1
        else:
            missing += 1
        records.append(rec)
        current += timedelta(days=1)

    out = pd.DataFrame.from_records(
        records, columns=["date", "nt1_mean", "nt3_mean", "nt1_p10", "nt1_p90"]
    )
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    print(f"Compiled {len(out)} daily rows; {missing} issue files missing/empty.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon-months",
        type=int,
        default=3,
        help="Number of upcoming months summed for the near-term mean (default 3).",
    )
    args = parser.parse_args()

    out = build(args.horizon_months)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
