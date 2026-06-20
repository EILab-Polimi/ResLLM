#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Back-calculate historical reservoir shortages from the observed release record.

Scores the observed Folsom release against the same CalSim-grounded additive demand stack
the complex simulation uses, so the historical baseline can be compared to the complex LLM
/ DP / MLP policies on identical shortage metrics.

Why this is well-posed: the ``outflow`` column of ``folsom_daily*.csv`` is a mass-balance
residual (``outflow = inflow - ΔS - evap``; verified corr 0.999 over WY1996-2016), so it
already contains every withdrawal from storage — exactly the single release ``rt`` the
complex model uses (``S = S + Q - rt``). The demand stack (minimum instream flow +
committed upstream M&I + supplemental downstream-flow demand) is therefore directly
comparable to the observed release.

Method (mirrors ``simulate.py`` / ``reservoir.py``):
  - The year-type index is the realized full-water-year inflow (perfect-hindsight: the
    requirement reflects the actual water-year type). The American River Index drops through
    the year as observed spills accumulate (releases above ``spill_threshold_cfs``), held
    per month like the simulation's monthly decision.
  - The minimum instream flow is ``Reservoir.compute_min_flow`` — the Lower American River
    Flow Management Standard Minimum Flows Requirement, refreshed once per month with
    ``Reservoir.update_monthly_min_flow`` and held across the month.
  - The observed release is attributed across the priority stack
    (MIF -> senior M&I -> junior M&I -> supplemental); each layer's shortfall is
    ``max(0, demand - served)``. The remainder above the full stack is split into supplemental vs
    flood spill (matching analysis/complex_helper._add_derived, historical branch): surplus is
    FLOOD SPILL only inside the flood season — Nov 1 (dowy 32) through June 15 (dowy 258) — AND
    within 100 TAF of / above the operating flood curve (``tocs_day``). Everything else is
    supplemental: the post-June-15 / October fall-drawdown release (no flood risk) and any day
    storage sits more than 100 TAF below the curve (not flood-proximate). ``delta_delivered`` stays
    at the modeled demand met on disk (the analysis layer folds the supplemental surplus into
    delivery when plotting). The transferred dynamic ``tocs`` gates the below-curve condition and
    is recorded as a storage-panel reference. (The agent's own run needs no rule here — its
    supplemental is the recorded allocation and all its surplus is dynamic-curve flood spill.)

Uses ONLY the observed ``outflow`` — never CalSim deliveries. CalSim only set the demand
magnitudes (read here from the complex config via the ``Reservoir``).

Run with the ``llm`` env (from anywhere):

    python data/backcalc_historical_shortage.py \
        --config folsom_complex.yml --inflow-file folsom_daily.csv \
        --start-year 1996 --end-year 2016 \
        --output data/historical_shortage_backcalc.csv

The output CSV mirrors the complex simulation-output schema (policy tag ``hist``) so it drops
into the figure pipeline: ``date, wy, mowy, dowy, qt, st, rt, dt, min_flow, min_flow_short,
delta_demand, delta_delivered, delta_short, committed_mi, junior_delivery_pct, junior_short,
senior_short, tocs, passthrough, spill, ari, hydro_class``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

# --- repo paths + complex-mode imports --------------------------------------------------
# This file lives at data/, so repo root is one level up.
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../data
_REPO = os.path.dirname(_HERE)              # repo root
_RESLLM = os.path.join(_REPO, "resllm")     # .../resllm
_DATA = _HERE                               # .../data
sys.path.insert(0, _RESLLM)

import src.utils as utils                   # noqa: E402
from src.reservoir import Reservoir         # noqa: E402


def build_reservoir(config_name: str, inflow_file: str, wy_forecast_file: str | None) -> Reservoir:
    """Build a complex-mode :class:`Reservoir` for its CalSim-grounded demand stack.

    Loads the same ``inflow_file`` and ``wy_forecast_file`` the simulation uses so that
    ``wy_inflow_index`` (and hence the ARI-driven minimum flow) reproduces the simulation's
    forward-looking index; only the config-driven demand methods are used. The flood gate is
    NOT computed here — the surplus is gated on the LLM run's recorded dynamic TOCS,
    transferred by date (see ``backcalc``), so both policies share the flood reference.
    """
    with open(os.path.join(_RESLLM, "configs", config_name)) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("complexity")
    if block is None:
        raise ValueError(f"{config_name} has no 'complexity' block (need the demand stack)")
    rc = cfg["folsom_reservoir"]
    chars = {
        "tocs": "fixed",
        "demand_file": os.path.join(_DATA, "demand.txt"),
        "inflow_file": os.path.join(_DATA, inflow_file),
        "wy_forecast_file": os.path.join(_DATA, wy_forecast_file) if wy_forecast_file else False,
        "wy_monthly_forecast_file": None,
        "operable_storage_max": rc["operable_storage_max"],
        "operable_storage_min": rc["operable_storage_min"],
        "max_safe_release": utils.cfs_to_taf(rc["max_safe_release"]),
        "sp_to_ep": rc["sp_to_ep"],
        "tp_to_tocs": rc["tp_to_tocs"],
        "sp_to_rp": rc["sp_to_rp"],
        "complexity": block,
        "complexity_mode": True,
    }
    return Reservoir(characteristics=chars)


def load_history(inflow_file: str, start_wy: int, end_wy: int) -> pd.DataFrame:
    """Load the observed daily record (date, inflow, outflow, storage) for the water-year
    window, dropping leap days (project convention) and assigning day/month-of-water-year."""
    df = pd.read_csv(os.path.join(_DATA, inflow_file), parse_dates=["date"])
    df = df.dropna(subset=["outflow"]).copy()
    df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))]
    df["wy"] = np.where(df["date"].dt.month >= 10, df["date"].dt.year + 1, df["date"].dt.year)
    df = df[(df["wy"] >= start_wy) & (df["wy"] <= end_wy)].sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No observed rows with WY in [{start_wy}, {end_wy}] in {inflow_file}")
    df["mowy"] = np.where(df["date"].dt.month > 9, df["date"].dt.month - 9, df["date"].dt.month + 3)
    df["dowy"] = df.groupby("wy").cumcount() + 1
    return df


def backcalc(R: Reservoir, hist: pd.DataFrame, dyn_tocs: dict[str, float] | None = None) -> pd.DataFrame:
    """Attribute each observed daily release across the priority demand stack and record the
    per-layer shortfalls.

    ``dyn_tocs`` maps ``YYYY-MM-DD`` -> the LLM run's recorded dynamic TOCS for that date;
    the surplus is gated on it so both policies share the flood reference. Dates missing
    from the map fall back to the static WCD curve.
    """
    spill_threshold_taf = utils.cfs_to_taf(R.complexity.get("spill_threshold_cfs", 8000))
    senior_frac = R._upstream_mi_senior_frac
    cvp_frac = R._upstream_mi_wf_cvp_frac
    demand = R.demand  # demand.txt (365 daily TAF) — recorded for reference only
    dyn_tocs = dyn_tocs or {}

    # Spill label: a calendar+storage rule (see attribution below), identical to
    # analysis/complex_helper._add_derived. The static WCM curve is kept only for the on-disk
    # ``tocs`` reference column (storage-panel dotted line) when no dynamic curve is transferred.
    tocs_tp, tocs_curve = R.characteristics["tp_to_tocs"]
    hist = hist.copy()
    _st = pd.to_numeric(hist["storage"], errors="coerce").ffill().bfill()
    hist["prev_storage"] = _st.shift(1).fillna(_st.iloc[0])  # end-of-prior-day storage (S_{t-1})

    rows = []
    for wy, g in hist.groupby("wy"):
        g = g.sort_values("dowy")
        spill_vol = 0.0
        wy_index = ari = None
        hydro_class = None
        cur_mowy = None
        for _, row in g.iterrows():
            mowy = int(row["mowy"])
            dowy = int(row["dowy"])
            outflow = float(row["outflow"])          # observed release == rt
            # Monthly decision point: refresh the forward-looking inflow index (observed-to-date
            # + remaining-WY forecast) and hold the ARI for the month, as simulate.py does on
            # day 1 from the spill accumulated so far.
            if mowy != cur_mowy:
                cur_mowy = mowy
                wy_index = R.wy_inflow_index(row["date"], int(wy), dowy)
                ari = R.american_river_index(wy_index, spill_vol)
                hydro_class = R.classify_water_year(wy_index)
                # Refresh the FMS minimum-flow state for the month. Observed month-start
                # storage drives the FRI / Jan-Feb triggers / off-ramp; the held wy_index
                # drives the SRI tier and IFII.
                st_month_start = float(row["storage"]) if not pd.isna(row.get("storage")) else 0.0
                R.update_monthly_min_flow(mowy, st_month_start, wy_index, wy=int(wy))

            instream = R.compute_min_flow()
            umi = R.upstream_mi_day(mowy)
            senior = umi * senior_frac
            junior_full = umi * cvp_frac
            # Supplemental indexed on the annual water-year hydrology (wy_index), not the
            # spill-decremented running ARI (the Delta obligation does not relax with spills).
            # Storage (start-of-day) applies the Nov-May low-storage relaxation, matching the sim.
            delta_demand = R.delta_demand_day(mowy, wy_index, st=float(row["prev_storage"]))

            # Attribute the observed release across the priority stack.
            served = outflow
            mif_served = min(served, instream);        served -= mif_served
            senior_served = min(served, senior);       served -= senior_served
            junior_served = min(served, junior_full);  served -= junior_served
            delta_served = min(served, delta_demand);  served -= delta_served
            surplus = served            # observed release above the full demand stack

            # Surplus split for the observed record (identical to analysis/complex_helper.
            # _add_derived, historical branch): above-demand release is FLOOD SPILL only inside the
            # flood season — Nov 1 (dowy 32) through June 15 (dowy 258) — AND within 100 TAF of /
            # above the operating flood curve (`tocs_day`). Everything else is SUPPLEMENTAL: the
            # post-June-15 / October fall-drawdown release (no flood risk) and any day storage sits
            # more than 100 TAF below the curve (not flood-proximate). delta_delivered stays at the
            # modeled demand met on disk; the analysis layer folds the supplemental surplus into
            # delivery when plotting. delta_short is the true supplemental shortfall (short days
            # carry no surplus). (The agent's own run needs no rule — its supplemental is the
            # recorded allocation and all its surplus is dynamic-curve flood spill.)
            tocs_day = float(dyn_tocs.get(row["date"].strftime("%Y-%m-%d"),
                                          np.interp(dowy, tocs_tp, tocs_curve)))
            cur_st = (float(row["storage"]) if not pd.isna(row.get("storage"))
                      else float(row["prev_storage"]))
            is_supp = (dowy < 32 or dowy > 258           # outside the Nov 1-June 15 flood season
                       or cur_st < tocs_day - 100.0)      # or not flood-proximate (below the curve)
            spill = 0.0 if is_supp else surplus
            passthrough = 0.0

            rows.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "wy": int(wy), "mowy": mowy, "dowy": dowy,
                "qt": float(row["inflow"]),
                "st": float(row["storage"]) if not pd.isna(row.get("storage")) else np.nan,
                "rt": outflow,
                "dt": float(demand[dowy - 1]) if dowy - 1 < len(demand) else np.nan,
                "min_flow": instream,
                "min_flow_short": max(0.0, instream - mif_served),
                "delta_demand": delta_demand,
                "delta_delivered": delta_served,
                "delta_short": max(0.0, delta_demand - delta_served),
                "committed_mi": senior_served + junior_served,
                "junior_delivery_pct": 100.0 * junior_served / junior_full if junior_full > 0 else 100.0,
                "junior_short": max(0.0, junior_full - junior_served),
                "senior_short": max(0.0, senior - senior_served),
                "tocs": tocs_day,           # dynamic TOCS (shared flood reference with the LLM run)
                "passthrough": passthrough,
                "spill": spill,
                "wy_index": wy_index,
                "ari": ari,
                "hydro_class": hydro_class,
            })
            # Accumulate spill for the ARI AFTER the release, mirroring the simulation.
            spill_vol += max(0.0, outflow - spill_threshold_taf)

    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> None:
    """Print annual shortage totals and flag years with an instream (regulatory) shortfall.

    The ``class`` column shows the settled (spring / mowy 8) forward-looking year-type; the
    October class is forecast-dominated and reads "normal" for every year.
    """
    print("\n== Historical back-calculated shortages (TAF/yr) ==")
    print(f"{'WY':>5} {'rt':>7} {'class':>8} {'min_flow_short':>15} "
          f"{'junior_short':>13} {'delta_short':>12} {'spill':>8}")
    for wy, g in df.groupby("wy"):
        spring = g.loc[g["mowy"] == 8, "hydro_class"]
        klass = spring.iloc[0] if not spring.empty else g["hydro_class"].iloc[-1]
        print(f"{int(wy):>5} {g['rt'].sum():>7.0f} {klass:>8} "
              f"{g['min_flow_short'].sum():>15.1f} {g['junior_short'].sum():>13.1f} "
              f"{g['delta_short'].sum():>12.1f} {g['spill'].sum():>8.0f}")
    tot = df[["min_flow_short", "junior_short", "delta_short"]].sum()
    print(f"{'ALL':>5} {df['rt'].sum():>7.0f} {'':>8} "
          f"{tot['min_flow_short']:>15.1f} {tot['junior_short']:>13.1f} {tot['delta_short']:>12.1f}")
    viol = sorted(int(w) for w, g in df.groupby("wy") if g["min_flow_short"].sum() > 1.0)
    print(f"\nWYs with an instream (regulatory) shortfall > 1 TAF: {viol or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="folsom_complex.yml",
                    help="Complex-mode config supplying the demand stack (default: folsom_complex.yml)")
    ap.add_argument("--inflow-file", default="folsom_daily.csv",
                    help="Observed daily record in data/ (date,inflow,outflow,storage,evap)")
    ap.add_argument("--wy-forecast-file", default="FOLC1_wy_hindcast.csv",
                    help="WY inflow forecast in data/ (drives the forward-looking index, "
                         "matching the simulation). Pass '' to fall back to the historical mean.")
    ap.add_argument("--dynamic-tocs-from",
                    default=os.path.join(_REPO, "analysis", "output", "resllm", "folsom_hist_complex",
                                         "deepseek-v4-pro-cloud_r-high_obj-balance-delivery_complex"
                                         "_simulation_output_n0.csv"),
                    help="LLM simulation-output CSV whose recorded dynamic `tocs` (by date) gates the "
                         "historical surplus — the shared flood reference. '' falls back to the static curve.")
    ap.add_argument("--start-year", type=int, default=1996, help="First water year (inclusive)")
    ap.add_argument("--end-year", type=int, default=2016, help="Last water year (inclusive)")
    ap.add_argument("--output", default=None,
                    help="Output CSV path (default: data/historical_shortage_backcalc.csv). "
                         "Never written under resllm/output/.")
    args = ap.parse_args()

    out = args.output or os.path.join(_DATA, "historical_shortage_backcalc.csv")
    R = build_reservoir(args.config, args.inflow_file, args.wy_forecast_file or None)
    hist = load_history(args.inflow_file, args.start_year, args.end_year)

    # Shared flood reference: the LLM run's RECORDED dynamic TOCS, by date.
    dyn_tocs = {}
    if args.dynamic_tocs_from and os.path.exists(args.dynamic_tocs_from):
        _sim = pd.read_csv(args.dynamic_tocs_from, usecols=["date", "tocs"])
        _sim = _sim.drop_duplicates("date", keep="last")
        dyn_tocs = dict(zip(_sim["date"].astype(str), _sim["tocs"].astype(float)))
        print(f"Dynamic-TOCS reference loaded: {len(dyn_tocs)} days from {args.dynamic_tocs_from}")
    else:
        print(f"WARNING: dynamic-TOCS reference not found ({args.dynamic_tocs_from!r}); "
              "gating on the static WCD curve instead.")
    df = backcalc(R, hist, dyn_tocs)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} daily rows (WY{args.start_year}-{args.end_year}) -> {out}")
    _summary(df)


if __name__ == "__main__":
    main()
