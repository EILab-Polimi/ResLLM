#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract Folsom's release/demand decomposition from the CalSim-3 DV parquet.

Data-grounding step for ResLLM's complex-mode demand stack. CalSim-3 decomposes the
Nimbus/Folsom outflow channel (``C_NTOMA``) into named physical components via its DSS
value-types, which map onto the layered stack the agent reasons about:

    C_NTOMA  =  C_NTOMA_MIF  +  C_NTOMA_ADD1  +  C_NTOMA_ADD2
    (channel)   FLOW-MIN-      FLOW-ADD-         FLOW-SPILL-
                INSTREAM        INSTREAM          POWER
                ~1029 TAF/yr    ~1032 TAF/yr      ~410 TAF/yr
                min-flow floor  supplemental      flood spill
                (forced)        downstream demand (excluded — physics, not demand)

Off-channel diversions (drawn from storage, not in ``C_NTOMA``) are summed from the
``D_FOLSM*`` / ``D_NTOMA_FSC003`` delivery nodes (upstream M&I, ~155 TAF/yr) and the
below-Nimbus ``D_AMR007/D_AMR017`` nodes (lower-American M&I, ~90 TAF/yr). The ``_WR`` /
``_CVP`` / ``_MFP`` contract suffixes partition each top node, so senior (water-rights) vs
junior (project) is split per-node by its ``_WR`` child without double-counting.

The supplemental demand fit bins realized ``C_NTOMA_ADD1`` against the Folsom inflow
estimate ``FOLSOMINFLOWEST_DV`` (TAF; long-run mean ~2631, matching the runtime
``wy_inflow_index`` mean ~2608), producing a ``[wy_index -> TAF/month]`` table the config's
``supplemental_schedule`` is interpolated on.

Run under the ``pyCalSim`` conda env (has ``pyarrow``; the project ``llm`` env does not):

    /Users/wyatt/miniforge3/envs/pyCalSim/bin/python \
        data/extract_calsim_folsom.py

Inputs
------
- ``../pyCalSim-replica/data/dss_extracted/DCR2023_DV_9.3.1_Danube_Hist_v1.7.parquet``
  (gitignored external repo; columns ``pathname/row/value/units``). Override with
  ``--parquet``.

Outputs
-------
- ``data/calsim_folsom_decomposition.csv`` — 12 rows, mowy Oct->Sep; monthly TAF for
  ``min_instream``, ``supplemental``, ``spill_power``, ``channel_total``,
  ``upstream_mi_{total,senior,junior}``, ``lower_mi``.
- ``data/calsim_supplemental_schedule.csv`` — the supplemental demand fit: per
  (mowy x ARI quantile bin) mean ``C_NTOMA_ADD1`` (FLOW-ADD-INSTREAM) in TAF/month, ready
  to transcribe into the config ``supplemental_schedule`` (a config-ready snippet is
  printed). Scales the supplemental demand with hydrology (dry ~433, normal ~983, wet ~1706
  TAF/yr) instead of a single ~1033 TAF/yr climatology.

Usage
-----
/Users/wyatt/miniforge3/envs/pyCalSim/bin/python data/extract_calsim_folsom.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Repo-root-relative paths (this file lives at data/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_PARQUET = os.path.abspath(
    os.path.join(
        _REPO_ROOT,
        "..",
        "pyCalSim-replica",
        "data",
        "dss_extracted",
        "DCR2023_DV_9.3.1_Danube_Hist_v1.7.parquet",
    )
)
DECOMP_OUT = os.path.join(_REPO_ROOT, "data", "calsim_folsom_decomposition.csv")
SUPPLEMENTAL_OUT = os.path.join(_REPO_ROOT, "data", "calsim_supplemental_schedule.csv")

# Unit conversion and calendar (leap days dropped — 365-day water year).
CFS_TO_TAF = 0.00198347  # cfs-month -> TAF when multiplied by days_in_month
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])  # Jan..Dec
WY_START, WY_END = 1922, 2021

# --- Channel decomposition (CalSim value-types) -----------------------------------------
ARC_MIF = "C_NTOMA_MIF"          # FLOW-MIN-INSTREAM  — forced regulatory floor
ARC_SUPPLEMENTAL = "C_NTOMA_ADD1"  # FLOW-ADD-INSTREAM — supplemental downstream-flow demand (Delta support)
ARC_SPILL = "C_NTOMA_ADD2"       # FLOW-SPILL-POWER   — winter flood spill (excluded)
ARC_CHANNEL = "C_NTOMA"          # total Nimbus/Folsom outflow
ARC_INDEX = "FOLSOMINFLOWEST_DV"  # Folsom inflow estimate (TAF) — wetness index

# --- Off-channel diversions -------------------------------------------------------------
# Top-level delivery nodes withdrawn from Folsom/Natoma storage (upstream of Nimbus). The
# _WR/_CVP/_MFP siblings partition each node (they sum to it), so sum ONLY these top nodes
# for the total and split senior/junior per-node by the node's _WR child — no double-count.
UPSTREAM_TOP_NODES = [
    "D_FOLSM_WTPSJP",   # San Juan
    "D_FOLSM_WTPRSV",   # Roseville
    "D_FOLSM_WTPFOL",   # City of Folsom
    "D_FOLSM_WTPEDH",   # El Dorado Hills
    "D_FOLSM_EDC_CVP",  # El Dorado County (CVP)
    "D_FOLSM_PCWA_CVP",  # Placer County WA (CVP)
    "D_FOLSM_26S_NU4",
    "D_FOLSM_26S_PU3",
    "D_FOLSM_EDCWA_WR",
    "D_FOLSM_EID_WR",
    "D_FOLSM_DITCH_WR",
    "D_NTOMA_FSC003",   # Folsom South Canal (CVP)
]
# Below-Nimbus (lower-American) M&I — served from the in-channel flow, not a storage draw.
LOWER_MI_NODES = ["D_AMR007_WTPFBN", "D_AMR017_WTPBJM"]

MONTH_NAMES = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]


def _load(parquet: str, bparts: list[str]) -> pd.DataFrame:
    """Load and decode the requested B-parts from the DV parquet.

    Parameters:
        parquet: Path to the extracted DV parquet.
        bparts: Exact DSS B-part names to keep.

    Returns:
        Frame with columns ``B``, ``wy``, ``mowy``, ``calmon``, ``value``, ``units``, and
        ``taf`` (CFS arcs converted via ``CFS_TO_TAF x days``; TAF arcs passed through),
        restricted to water years ``WY_START..WY_END``.
    """
    df = pq.read_table(parquet, columns=["pathname", "row", "value", "units"]).to_pandas()
    df["B"] = df["pathname"].str.split("/").str[2]
    df = df[df["B"].isin(bparts)].copy()
    dyear = df["pathname"].str.split("/").str[4].str[-4:].astype(int)
    absm = (dyear - 1920) * 12 + df["row"]
    df["calyear"] = 1920 + absm // 12
    df["calmon"] = absm % 12 + 1                       # 1=Jan .. 12=Dec
    df["mowy"] = (df["calmon"] - 10) % 12 + 1          # 1=Oct .. 12=Sep
    df["wy"] = np.where(df["calmon"] >= 10, df["calyear"] + 1, df["calyear"])
    df = df[(df["wy"] >= WY_START) & (df["wy"] <= WY_END)].copy()
    is_cfs = df["units"].astype(str).str.upper() == "CFS"
    df["taf"] = np.where(
        is_cfs, df["value"] * CFS_TO_TAF * DAYS_IN_MONTH[df["calmon"].to_numpy() - 1], df["value"]
    )
    return df


def _monthly_mean_taf(df: pd.DataFrame, bpart: str) -> pd.Series:
    """Mean TAF per water-year-month (mowy 1..12) for one arc."""
    sub = df[df["B"] == bpart]
    return sub.groupby("mowy")["taf"].mean().reindex(range(1, 13), fill_value=0.0)


def build_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Build the monthly demand-stack decomposition (TAF, mowy Oct->Sep)."""
    out = pd.DataFrame({"mowy": range(1, 13), "month": MONTH_NAMES})
    out["min_instream"] = _monthly_mean_taf(df, ARC_MIF).to_numpy()
    out["supplemental"] = _monthly_mean_taf(df, ARC_SUPPLEMENTAL).to_numpy()
    out["spill_power"] = _monthly_mean_taf(df, ARC_SPILL).to_numpy()
    out["channel_total"] = _monthly_mean_taf(df, ARC_CHANNEL).to_numpy()

    # Upstream M&I: total = sum of top nodes; senior = each node's _WR child (or the node
    # itself if it ends in _WR); junior = total - senior.
    present = set(df["B"].unique())
    senior = pd.Series(0.0, index=range(1, 13))
    total = pd.Series(0.0, index=range(1, 13))
    for node in UPSTREAM_TOP_NODES:
        if node not in present:
            continue
        node_m = _monthly_mean_taf(df, node)
        total = total.add(node_m, fill_value=0.0)
        if node.endswith("_WR"):
            senior = senior.add(node_m, fill_value=0.0)
        elif (node + "_WR") in present:
            senior = senior.add(_monthly_mean_taf(df, node + "_WR"), fill_value=0.0)
    out["upstream_mi_total"] = total.to_numpy()
    out["upstream_mi_senior"] = senior.to_numpy()
    out["upstream_mi_junior"] = (total - senior).to_numpy()

    lower = pd.Series(0.0, index=range(1, 13))
    for node in LOWER_MI_NODES:
        if node in present:
            lower = lower.add(_monthly_mean_taf(df, node), fill_value=0.0)
    out["lower_mi"] = lower.to_numpy()

    return out.round(2)


def build_supplemental_fit(df: pd.DataFrame, n_bins: int = 7) -> pd.DataFrame:
    """Fit the supplemental demand schedule: mean ``C_NTOMA_ADD1`` (TAF) by mowy x ARI bin.

    The annual ``FOLSOMINFLOWEST_DV`` index is split into ``n_bins`` equal-count quantile
    bins; each bin's breakpoint is its mean index (centroid) and each month's value is the
    bin-mean realized supplemental TAF. The top breakpoint sits at the very-wet centroid so
    the wet tail is captured. At runtime the schedule is indexed on the annual water-year
    hydrology (``wy_index``), not the spill-decremented ARI, so these annual-index
    breakpoints match the lookup. Values are NOT made monotonic: the supplemental can dip in
    a wetter bin some months, and ``np.interp`` needs only the increasing breakpoints the bin
    centroids provide.

    Returns a table with columns ``bin``, ``ari_taf``, ``n_wy`` and ``Oct..Sep`` TAF.
    """
    idx = df[df["B"] == ARC_INDEX].groupby("wy")["value"].sum()  # TAF/yr
    bin_of_wy = pd.qcut(idx, n_bins, labels=False, duplicates="drop")

    sup = df[df["B"] == ARC_SUPPLEMENTAL].copy()
    sup["bin"] = sup["wy"].map(bin_of_wy)
    n_actual = int(sup["bin"].nunique())
    permonth = {b: sup[sup["bin"] == b].groupby("mowy")["taf"].mean().reindex(range(1, 13), fill_value=0.0)
                for b in range(n_actual)}

    rows = []
    for b in range(n_actual):
        wys = sup.loc[sup["bin"] == b, "wy"].unique()
        ari = float(idx.loc[idx.index.isin(wys)].mean())
        row = {"bin": b, "ari_taf": round(ari, 0), "n_wy": int(len(wys))}
        for mo in range(1, 13):
            row[MONTH_NAMES[mo - 1]] = round(float(permonth[b].iloc[mo - 1]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _print_supplemental_config_snippet(fit: pd.DataFrame) -> None:
    """Print the supplemental schedule as a config-ready ``supplemental_schedule`` block."""
    bps = [int(round(v)) for v in fit["ari_taf"].tolist()]
    print("\n=== config supplemental_schedule (transcribe into folsom_complex.yml) ===")
    print("  supplemental_schedule:")
    for mo in range(1, 13):
        vals = [round(float(fit.iloc[c][MONTH_NAMES[mo - 1]]), 2) for c in range(len(fit))]
        print(f"    - [{bps}, {vals}]   # {mo:>2} {MONTH_NAMES[mo - 1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET, help="DV parquet path.")
    parser.add_argument("--supp-bins", type=int, default=7,
                        help="Number of equal-count ARI quantile bins for the supplemental "
                             "sliding scale.")
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        raise FileNotFoundError(
            f"DV parquet not found: {args.parquet}\n"
            "Run under the pyCalSim env and ensure pyCalSim-replica is checked out."
        )

    all_bparts = (
        [ARC_MIF, ARC_SUPPLEMENTAL, ARC_SPILL, ARC_CHANNEL, ARC_INDEX]
        + UPSTREAM_TOP_NODES
        + [n + "_WR" for n in UPSTREAM_TOP_NODES]
        + LOWER_MI_NODES
    )
    df = _load(args.parquet, all_bparts)

    decomp = build_decomposition(df)
    os.makedirs(os.path.dirname(DECOMP_OUT), exist_ok=True)
    decomp.to_csv(DECOMP_OUT, index=False)

    supp_fit = build_supplemental_fit(df, args.supp_bins)
    supp_fit.to_csv(SUPPLEMENTAL_OUT, index=False)

    # ---- report ----
    ann = decomp.sum(numeric_only=True)
    print("=== Decomposition (TAF/yr, WY%d-%d) ===" % (WY_START, WY_END))
    print(f"  min_instream (MIF) ........ {ann['min_instream']:7.0f}")
    print(f"  supplemental (ADD1) ....... {ann['supplemental']:7.0f}")
    print(f"  spill_power (ADD2, excl) .. {ann['spill_power']:7.0f}")
    print(f"  channel_total (C_NTOMA) ... {ann['channel_total']:7.0f}")
    print(f"  MIF+ADD1+ADD2 ............. {ann['min_instream']+ann['supplemental']+ann['spill_power']:7.0f}")
    print(f"  upstream_mi total ......... {ann['upstream_mi_total']:7.0f}"
          f"  (senior {ann['upstream_mi_senior']:.0f} / junior {ann['upstream_mi_junior']:.0f})")
    print(f"  lower_mi .................. {ann['lower_mi']:7.0f}")
    sfrac = ann["upstream_mi_senior"] / ann["upstream_mi_total"]
    print(f"  -> senior_frac {sfrac:.2f} / wf_cvp_frac {1-sfrac:.2f}")
    print()
    print("=== Supplemental fit (TAF/month by mowy x ARI bin) ===")
    print(supp_fit.to_string(index=False))
    print(f"  annual supplemental by bin (TAF/yr): " + ", ".join(
        f"b{int(r['bin'])}({r['ari_taf']:.0f})={sum(r[m] for m in MONTH_NAMES):.0f}"
        for _, r in supp_fit.iterrows()))
    _print_supplemental_config_snippet(supp_fit)
    print()
    print(f"Wrote {DECOMP_OUT}")
    print(f"Wrote {SUPPLEMENTAL_OUT}")


if __name__ == "__main__":
    main()
