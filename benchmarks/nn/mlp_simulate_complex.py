"""
Simulation script using the trained MLP allocation model against the COMPLEX
demand stack (the year-specific, hydrology- and storage-responsive CalSim-grounded
four-layer demand stack the complex LLM operated against), instead of the fixed
``demand.txt`` profile.

The MLP keeps its single allocation lever (0-1, predicted monthly and held for the
month), but that lever now scales the *entire* demand stack each day:

    uu = (min_flow_full + umi_senior + junior_full + delta_demand)
           * allocation_percent / 100

i.e. one uniform fraction across the four priority layers. The firm floors
(min-flow regulatory floor and senior committed M&I) are still force-served by
``Reservoir.evaluate()`` (via ``min_flow`` / ``senior_floor`` / ``cap_protect``),
so when the MLP cuts below the floor the physics enforces it -- matching the
"single lever with force-served floors" behaviour.

The daily loop, demand-stack construction, evaporation, dynamic flood curve, ARI /
hydro-class / Water-Forum factor, FMS minimum-flow state, and recorded column
schema mirror ``resllm/simulate.py`` complex mode exactly, so the output is
directly post-classifiable by ``analysis_paper_v2/postclassify_mlp.py`` (which
already loads ``complex_helper`` and the historical back-calc) and comparable to
the complex LLM run column-for-column.

Usage:
    python mlp_simulate_complex.py
    python mlp_simulate_complex.py --start-year 1996 --end-year 2016 \
        --config folsom_complex --tocs dynamic_hist_cap \
        --inflow-file folsom_daily.csv --wy-forecast-file FOLC1_wy_hindcast.csv \
        --wy-monthly-forecast-file FOLC1_monthly_forecast.csv \
        --model-dir ./output --output-dir ./output
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
import yaml
import argparse

# Add resllm to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'resllm'))
from src.reservoir import Reservoir
import src.utils as utils


def load_model(model_dir='./output'):
    """Load the trained model, scalers, and metadata from ``model_dir``."""
    mlp = joblib.load(os.path.join(model_dir, 'mlp_allocation_model.pkl'))
    scalers = joblib.load(os.path.join(model_dir, 'mlp_allocation_scalers.pkl'))
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']
    with open(os.path.join(model_dir, 'mlp_allocation_metadata.json'), 'r') as f:
        metadata = json.load(f)
    return mlp, scaler_X, scaler_y, metadata


def prepare_features(storage, water_year_month, allocation_prev, inflow_ma):
    """Build the 5-feature MLP input (must match ``mlp_train.py`` exactly).

    Features: [storage, month_sin, month_cos, allocation_prev, inflow_ma]
    """
    storage = np.atleast_1d(storage)
    water_year_month = np.atleast_1d(water_year_month)
    allocation_prev = np.atleast_1d(allocation_prev)
    inflow_ma = np.atleast_1d(inflow_ma)
    month_sin = np.sin(2 * np.pi * water_year_month / 12)
    month_cos = np.cos(2 * np.pi * water_year_month / 12)
    X = np.column_stack([storage, month_sin, month_cos, allocation_prev, inflow_ma])
    return X


def predict_allocation(mlp, scaler_X, scaler_y, X):
    """Predict allocation (clipped to [0, 1])."""
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = mlp.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    return np.clip(y_pred, 0.0, 1.0)


def run_simulation(start_year=1996, end_year=2016, starting_storage=466.1,
                   config_name="folsom_complex", tocs="dynamic_hist_cap",
                   inflow_file="folsom_daily.csv",
                   wy_forecast_file="FOLC1_wy_hindcast.csv",
                   monthly_forecast_file="FOLC1_monthly_forecast.csv",
                   model_dir="./output"):
    """Run the complex-stack MLP simulation.

    Mirrors ``resllm/simulate.py`` complex mode. The MLP's single allocation lever
    scales the full four-layer demand stack (min flow + senior M&I + junior M&I +
    supplemental downstream flow); the firm floors are force-served by
    ``Reservoir.evaluate()``. The output schema matches the complex LLM run so
    ``postclassify_mlp.py`` works unchanged.
    """
    print("=" * 60)
    print("MLP Complex-Stack Reservoir Simulation")
    print("=" * 60)

    # --- MLP model -----------------------------------------------------------
    print("\nLoading MLP model...")
    mlp, scaler_X, scaler_y, metadata = load_model(model_dir)
    print(f"✓ Model loaded (Test R²: {metadata['performance']['test_r2']:.4f})")

    # --- Reservoir config (FULL complex mode) --------------------------------
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resllm',
                               'configs', f"{config_name}.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"✓ Configuration loaded: {config['config_name']}")

    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    res = config["folsom_reservoir"]
    R1_characteristics = {
        "tocs": tocs,
        "demand_file": os.path.join(data_dir, "demand.txt"),  # reference field (dt) only
        "inflow_file": os.path.join(data_dir, inflow_file),
        "wy_forecast_file": os.path.join(data_dir, wy_forecast_file),
        "wy_monthly_forecast_file": os.path.join(data_dir, monthly_forecast_file),
        "operable_storage_max": res["operable_storage_max"],
        "operable_storage_min": res["operable_storage_min"],
        "max_safe_release": utils.cfs_to_taf(res["max_safe_release"]),
        "sp_to_ep": res["sp_to_ep"],
        "tp_to_tocs": res["tp_to_tocs"],
        "sp_to_rp": res["sp_to_rp"],
        # Full complexity block -> compute_min_flow, delta_demand_day, upstream_mi_day,
        # ARI, hydro-class, Water-Forum factor, FMS chaining all active.
        "complexity": config["complexity"],
        "complexity_mode": True,
    }

    R1 = Reservoir(characteristics=R1_characteristics)
    print("✓ Complex-mode reservoir initialized")

    # Validate water-year range
    available_years = sorted(R1.inflows['water_year'].unique())
    min_wy, max_wy = available_years[0], available_years[-1]
    if start_year < min_wy or end_year > max_wy:
        raise ValueError(
            f"\nRequested water years {start_year}-{end_year} are outside available range.\n"
            f"Available data: water years {min_wy}-{max_wy}\n"
            f"(Date range: {R1.inflows['date'].iloc[0]} to {R1.inflows['date'].iloc[-1]})"
        )
    print(f"✓ Data validation passed (WY {min_wy}-{max_wy} available)")

    # --- Complex-mode constants (mirrors simulate.py) ------------------------
    spill_threshold_taf = utils.cfs_to_taf(
        R1.complexity.get("spill_threshold_cfs", 8000)
    )
    min_flow_floor_frac = float(
        R1.complexity.get("min_flow_decision", {}).get("floor_frac", 0.5)
    )

    # --- Simulation state ----------------------------------------------------
    ny = end_year - start_year + 1
    R1.record = pd.DataFrame(index=range(ny * 365))
    decision_record = []

    release_history = []   # for allocation_prev (30-day rolling release/demand)
    demand_history = []    # full-stack demand each day (denominator for allocation_prev)
    inflow_history = []     # for inflow_ma (120-day moving average)

    allocation_percent = 100.0
    allocation_prev = 1.0
    near_term = 0.0
    s0 = starting_storage
    t = 0

    # month-held complex state (refreshed on the 1st of each month)
    carryover_target = -1.0          # MLP sets no carryover target -> no cap
    junior_delivery_percent = 100.0  # MLP has no junior lever -> full commitment
    meet_min_flow = True             # MLP has no min-flow lever -> deliver full MIF
    hydro_class = None
    wy_idx = 0.0
    ari = 0.0
    wf_factor = 1.0
    spill_vol = 0.0

    print(f"\nSimulating water years {start_year} to {end_year}...")
    print(f"Starting storage: {starting_storage:.1f} TAF")

    # --- Period-of-record loop (mirrors simulate.py lines 144-322) ------------
    for wy in np.arange(start_year, end_year + 1):
        print(f"  Water year {wy}...")

        date_range = pd.date_range(start=f"{wy-1}-10-01", end=f"{wy}-09-30", freq="D")
        # drop leap day (project convention: 365 days/WY)
        if len(date_range) == 366:
            leap_day = (date_range.month == 2) & (date_range.day == 29)
            date_range = date_range[~leap_day]

        # ARI counts spills since October -> reset each WY
        spill_vol = 0.0

        for ty, d in enumerate(date_range):
            mowy = d.month - 9 if d.month > 9 else d.month + 3

            # previous day's end-of-day storage
            st_1 = s0 if t == 0 else R1.record.loc[t - 1, "st"]

            # --- MLP monthly decision (1st of each month) ---------------------
            if d.day == 1:
                # 120-day moving-average inflow (feature)
                if len(inflow_history) >= 120:
                    inflow_ma = np.mean(inflow_history[-120:])
                elif len(inflow_history) > 0:
                    inflow_ma = np.mean(inflow_history)
                else:
                    inflow_ma = 7.0

                X = prepare_features(
                    storage=st_1,
                    water_year_month=mowy,
                    allocation_prev=allocation_prev,
                    inflow_ma=inflow_ma,
                )
                allocation_predicted = predict_allocation(mlp, scaler_X, scaler_y, X)[0]
                allocation_percent = allocation_predicted * 100.0

                decision_record.append({
                    'date': d,
                    'wy': wy,
                    'mowy': mowy,
                    'dowy': ty + 1,
                    'storage': st_1,
                    'allocation_prev': allocation_prev,
                    'inflow_ma': inflow_ma,
                    'allocation_decision': allocation_predicted,
                    'allocation_percent': allocation_percent,
                })

                # Refresh the month's complex-mode drivers before building the stack,
                # so the held ARI / min flow / Water-Forum factor match the daily physics
                # (same ordering as simulate.py lines 172-182).
                wy_idx = R1.wy_inflow_index(d, wy, ty + 1)
                ari = R1.american_river_index(wy_idx, spill_vol)
                hydro_class = R1.classify_water_year(wy_idx)
                near_term = R1.near_term_inflow(d)
                wf_factor = R1.compute_water_forum_factor(wy_idx)
                # FMS minimum-flow state for the month (uses month-start storage).
                R1.update_monthly_min_flow(mowy, st_1, wy_idx, wy=int(wy))

            # --- Downstream demand -------------------------------------------
            # dt = demand.txt is carried as a reference field only (same as complex sim;
            # the actual demand lives in the stack built below).
            dt = R1.demand[ty]

            # inflow row for this date
            inflow_rows = R1.inflows.loc[
                (R1.inflows["water_year"] == wy)
                & (R1.inflows["month"] == d.month)
                & (R1.inflows["day"] == d.day)
            ]
            if inflow_rows.empty:
                raise ValueError(
                    f"Missing inflow for date={d.strftime('%Y-%m-%d')} (WY={wy})"
                )
            qt = float(inflow_rows["inflow"].iloc[0])

            # observed evaporative loss (complex mode only; NaN -> 0)
            evap_t = 0.0
            if "evap" in inflow_rows.columns:
                _e = inflow_rows["evap"].iloc[0]
                evap_t = float(_e) if pd.notna(_e) else 0.0

            # --- Demand stack (mirrors simulate.py lines 252-269) ------------
            min_flow = R1.compute_min_flow()
            tocs_day = R1.compute_tocs(
                dowy=ty + 1, date=d.strftime("%Y-%m-%d"), near_term=near_term,
            )
            # No carryover cap (MLP sets no target) -> release_cap stays inf.
            release_cap = float("inf")

            min_flow_full = min_flow                                     # required MIF (FMS)
            min_flow_floor = min_flow_full * min_flow_floor_frac         # firm floor (forced)
            min_flow_target = min_flow_full if meet_min_flow else min_flow_floor
            umi = R1.upstream_mi_day(mowy, ty + 1)
            umi_senior = umi * R1._upstream_mi_senior_frac                # firm, always served
            junior_full = umi * R1._upstream_mi_wf_cvp_frac              # full junior commitment
            junior_target = junior_full * junior_delivery_percent / 100.0
            delta_demand = R1.delta_demand_day(mowy, wy_idx, st=st_1)    # supplemental (hydro+storage)
            firm_floor = min_flow_floor + umi_senior                      # protected (floor + senior)

            # MLP single lever scales the ENTIRE stack uniformly:
            #   uu = (min_flow_full + umi_senior + junior_full + delta_demand) * alloc%
            # The firm floors are still force-served by evaluate(), so a sub-floor
            # allocation is raised to the floor in the realized release (matching the
            # chosen "single lever, force-served floors" semantics).
            full_stack = min_flow_full + umi_senior + junior_full + delta_demand
            uu = full_stack * allocation_percent / 100.0

            # --- Evaluate (complex path, mirrors simulate.py lines 271-284) ---
            rt, st = R1.evaluate(
                st_1=st_1, qt=qt, uu=uu, tocs=tocs_day,
                min_flow=min_flow_floor, release_cap=release_cap,
                senior_floor=firm_floor,
                cap_protect=min_flow_target + umi_senior,
                evaporation=evap_t,
            )

            # --- Attribute the realized release across the priority stack -----
            # (MIF -> senior M&I -> junior M&I -> supplemental; remainder = flood spill)
            served = rt
            mif_served = min(served, min_flow_target);   served -= mif_served
            senior_served = min(served, umi_senior);      served -= senior_served
            junior_served = min(served, junior_target);  served -= junior_served
            delta_served = min(served, delta_demand);     served -= delta_served
            min_flow_short = max(0.0, min_flow_full - mif_served)
            junior_short = max(0.0, junior_full - junior_served)
            delta_short = max(0.0, delta_demand - delta_served)
            committed_mi = senior_served + junior_served
            flood_release = max(0.0, rt - uu)
            spill_vol += max(0.0, rt - spill_threshold_taf)

            # --- Record (full complex-mode schema) --------------------------
            R1.record_timestep(
                idx=t, date=d, wy=wy, mowy=mowy, dowy=ty + 1, qt=qt, st=st, rt=rt,
                dt=dt, uu=uu, min_flow=min_flow_full, tocs=tocs_day,
                hydro_class=hydro_class, delta_demand=delta_demand,
                delta_delivered=delta_served, delta_short=delta_short,
                committed_mi=committed_mi, junior_delivery_pct=junior_delivery_percent,
                junior_short=junior_short, near_term=near_term, release_cap=release_cap,
                flood_release=flood_release, wf_factor=wf_factor, ari=ari,
                min_flow_short=min_flow_short, evap=evap_t,
            )

            # --- Rolling histories for the MLP features ---------------------
            release_history.append(rt)
            demand_history.append(full_stack)
            inflow_history.append(qt)

            if len(release_history) >= 30:
                _r = np.array(release_history[-30:]).clip(max=10)
                _d = np.array(demand_history[-30:])
                allocation_prev = float(np.mean(np.clip(_r / _d, 0.0, 1.0)))

            t += 1

    print(f"✓ Simulation complete ({t} timesteps)")
    decision_df = pd.DataFrame(decision_record)
    return R1.record, decision_df


def main():
    """Run the complex-stack MLP simulation."""
    parser = argparse.ArgumentParser(
        description='Run MLP-based reservoir simulation against the complex demand stack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Historical complex simulation (1996-2016):
  python mlp_simulate_complex.py
  # With a different inflow/forecast file:
  python mlp_simulate_complex.py --inflow-file folsom_daily.csv \\
      --wy-forecast-file FOLC1_wy_hindcast.csv
""",
    )
    parser.add_argument('--config', type=str, default='folsom_complex',
                        help='Reservoir config supplying physical+complexity params '
                             '(default: folsom_complex)')
    parser.add_argument('--tocs', type=str, default='dynamic_hist_cap',
                        choices=['fixed', 'historical', 'dynamic', 'dynamic_hist_cap'],
                        help='TOCS / flood-curve mode (default: dynamic_hist_cap)')
    parser.add_argument('--inflow-file', type=str, default='folsom_daily.csv',
                        help='Inflow file in ../../data (must have a storage column for '
                             'dynamic_hist_cap)')
    parser.add_argument('--wy-forecast-file', type=str, default='FOLC1_wy_hindcast.csv',
                        help='Full-WY inflow forecast file in ../../data (drives the '
                             'year-specific ARI and hydrology-indexed demand)')
    parser.add_argument('--wy-monthly-forecast-file', type=str,
                        default='FOLC1_monthly_forecast.csv',
                        help='Monthly near-term forecast (nt3_mean) for the dynamic curve')
    parser.add_argument('--model-dir', type=str, default='./output',
                        help='Directory with the MLP .pkl/.json artifacts (default: ./output)')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to write simulation/decision CSVs (default: ./output)')
    parser.add_argument('--start-year', type=int, default=1996,
                        help='Starting water year (default: 1996)')
    parser.add_argument('--end-year', type=int, default=2016,
                        help='Ending water year (default: 2016)')
    parser.add_argument('--starting-storage', type=float, default=466.1,
                        help='Initial storage in TAF (default: 466.1)')
    parser.add_argument('--fix-tocs', action='store_true', default=False,
                        help='Override --tocs to "fixed" (back-compat; default: False)')
    args = parser.parse_args()
    tocs_mode = 'fixed' if args.fix_tocs else args.tocs

    simulation_df, decision_df = run_simulation(
        start_year=args.start_year,
        end_year=args.end_year,
        starting_storage=args.starting_storage,
        config_name=args.config,
        tocs=tocs_mode,
        inflow_file=args.inflow_file,
        wy_forecast_file=args.wy_forecast_file,
        monthly_forecast_file=args.wy_monthly_forecast_file,
        model_dir=args.model_dir,
    )

    # --- Save outputs --------------------------------------------------------
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    start_year = args.start_year
    end_year = args.end_year
    period = 'paleo' if 'paleo' in args.config else 'hist'
    tocs_tag = tocs_mode.replace('_', '')

    simulation_file = os.path.join(
        args.output_dir,
        f'mlp_simulation_output_complex_{period}_{tocs_tag}_{start_year}_{end_year}.csv',
    )
    decision_file = os.path.join(
        args.output_dir,
        f'mlp_decision_output_complex_{period}_{tocs_tag}_{start_year}_{end_year}.csv',
    )

    simulation_df.to_csv(simulation_file, index=False)
    print(f"✓ Simulation output saved to: {simulation_file}")
    print(f"  Shape: {simulation_df.shape}")
    decision_df.to_csv(decision_file, index=False)
    print(f"✓ Decision output saved to: {decision_file}")
    print(f"  Shape: {decision_df.shape}")

    # --- Summary statistics --------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    print(f"\nStorage Performance:")
    print(f"  Mean storage: {simulation_df['st'].mean():.1f} TAF")
    print(f"  Min storage: {simulation_df['st'].min():.1f} TAF")
    print(f"  Max storage: {simulation_df['st'].max():.1f} TAF")
    print(f"  Final storage: {simulation_df['st'].iloc[-1]:.1f} TAF")

    print(f"\nMonthly Decisions:")
    print(f"  Mean allocation decision: {decision_df['allocation_decision'].mean():.3f}")
    print(f"  Min allocation decision: {decision_df['allocation_decision'].min():.3f}")
    print(f"  Max allocation decision: {decision_df['allocation_decision'].max():.3f}")
    print(f"  Shortage months (< 100%): {(decision_df['allocation_decision'] < 1.0).sum()}")

    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
