#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the historical Sacramento River Index (SRI) the FMS keys on.

The Lower American River Flow Management Standard (ARWA-103) sets the Jan–Feb minimum
instream flow from the Sacramento River Index — "75 percent of exceedance forecast on the
first of the current month" (FMS Table 1), with tier boundaries at 10.2 and 15.7 MAF
(critical / dry-below-normal / above-normal-wet). That index is published in the CalSim-3
DV as ``SACRIVERINDEX_75EXC_DV`` (TAF, monthly forecast), so we read it from the same
DCR2023 DV used for the rest of the demand stack rather than scraping CDEC — the CDEC
"Sacramento Valley Water Year Index" (40-30-30) is a different, lower-scale index whose
values would not match the FMS's 10.2/15.7 MAF thresholds.

This gives the monthly-updated SRI per water year so the FMS min-flow implementation
(``Reservoir`` complex mode) uses the actual index instead of a Folsom-inflow
approximation. It is a forecast (not perfect hindsight), so it leaks no future information.

Run under the ``pyCalSim`` env (has pyarrow):

    /Users/wyatt/miniforge3/envs/pyCalSim/bin/python \
        data/extract_sacramento_river_index.py

Output
------
- ``data/sacramento_river_index.csv`` — columns ``date, wy, calmon, mowy, sri_taf`` (first of
  each month, all water years in the DV). The runtime looks up the Jan (mowy 4) and Feb
  (mowy 5) value for the current water year.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# This file lives at data/, so repo root is one level up.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_PARQUET = os.path.abspath(
    os.path.join(_REPO_ROOT, "..", "pyCalSim-replica", "data", "dss_extracted",
                 "DCR2023_DV_9.3.1_Danube_Hist_v1.7.parquet")
)
OUT = os.path.join(_REPO_ROOT, "data", "sacramento_river_index.csv")
ARC_SRI = "SACRIVERINDEX_75EXC_DV"


def main() -> None:
    if not os.path.exists(DEFAULT_PARQUET):
        raise FileNotFoundError(f"DV parquet not found: {DEFAULT_PARQUET}")
    df = pq.read_table(DEFAULT_PARQUET, columns=["pathname", "row", "value", "units"]).to_pandas()
    df["B"] = df["pathname"].str.split("/").str[2]
    df = df[df["B"] == ARC_SRI].copy()
    dyear = df["pathname"].str.split("/").str[4].str[-4:].astype(int)
    absm = (dyear - 1920) * 12 + df["row"]
    df["calyear"] = 1920 + absm // 12
    df["calmon"] = absm % 12 + 1
    df["mowy"] = (df["calmon"] - 10) % 12 + 1
    df["wy"] = np.where(df["calmon"] >= 10, df["calyear"] + 1, df["calyear"])
    df["date"] = pd.to_datetime(dict(year=df["calyear"], month=df["calmon"], day=1))
    out = (df[["date", "wy", "calmon", "mowy", "value"]]
           .rename(columns={"value": "sri_taf"})
           .sort_values("date").reset_index(drop=True))
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)

    # ---- report: Jan SRI + FMS tier per WY over the ResLLM run period ----
    jan = out[out["mowy"] == 4].set_index("wy")["sri_taf"]
    def tier(s):  # MAF thresholds 10.2 / 15.7 -> TAF 10200 / 15700
        return "critical" if s < 10200 else ("wet/an" if s >= 15700 else "dry/bn")
    print(f"Wrote {OUT}  ({len(out)} rows, WY{out['wy'].min()}–{out['wy'].max()})")
    print(f"SRI range: {out['sri_taf'].min():.0f}–{out['sri_taf'].max():.0f} TAF "
          f"({out['sri_taf'].min()/1000:.1f}–{out['sri_taf'].max()/1000:.1f} MAF)\n")
    print("Jan SRI (75% exc forecast) and FMS tier, WY1996–2016:")
    for wy in range(1996, 2017):
        if wy in jan.index:
            s = jan[wy]
            print(f"  WY{wy}: {s:8.0f} TAF ({s/1000:5.1f} MAF)  -> {tier(s)}")


if __name__ == "__main__":
    main()
