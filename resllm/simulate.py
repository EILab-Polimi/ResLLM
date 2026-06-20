#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import threading
import yaml
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# resllm imports
from src.reservoir import Reservoir
from src.operator import build_operator
from src.model_config import RunIntent, resolve_model_config
from src.ablation import ABLATION_TYPES
import src.utils as utils

# dotenv imports
from dotenv import load_dotenv
load_dotenv(verbose=True)


def run_single_sample(
    n: int,
    args,
    resolved_model_config,
    model: str,
    R1_characteristics: dict,
    file_dir: str,
    print_lock: threading.Lock,
) -> None:
    """Run a single simulation sample end-to-end.

    Parameters:
        n: Sample index.
        args: Parsed CLI arguments.
        resolved_model_config: Resolved model configuration.
        model: Model name string (used for output filenames).
        R1_characteristics: Reservoir characteristics dict.
        file_dir: Absolute path to the resllm/ directory.
        print_lock: Threading lock for serialized console output.
    """
    import src.utils as utils  # noqa: F401 (ensure available in thread)

    start_wy = args.start_year
    end_wy = args.end_year
    s0 = args.starting_storage

    # Output filenames (needed before resume detection)
    safe_model_name = model.replace(":", "-").replace("/", "_")
    safe_reasoning_effort = (args.reasoning_effort or "none").strip().lower().replace(" ", "-")
    output_stem = f"{safe_model_name}_r-{safe_reasoning_effort}"
    if args.objective != "minimize-shortages":
        output_stem += f"_obj-{args.objective}"
    if R1_characteristics.get("complexity_mode"):
        output_stem += "_complex"
    # Tag ablated runs so they never collide with baseline output filenames.
    if getattr(args, "ablation_type", None):
        safe_ablation = args.ablation_type.strip().lower().replace("_", "-")
        output_stem += f"_abl-{safe_ablation}"
    output_dir = os.path.join(file_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    simulation_output_file = os.path.join(
        output_dir, f"{output_stem}_simulation_output_n{n}.csv"
    )
    decision_output_file = os.path.join(
        output_dir, f"{output_stem}_decision_output_n{n}.csv"
    )

    # Resume: detect last complete water year from existing output
    if args.resume and os.path.exists(simulation_output_file):
        sim_existing = pd.read_csv(simulation_output_file)
        by_wy = sim_existing.groupby("wy").size()
        complete_wys = by_wy[by_wy == 365]
        if not complete_wys.empty:
            last_complete_wy = int(complete_wys.index[-1])
            if last_complete_wy < end_wy:
                resume_s0 = float(sim_existing[sim_existing["wy"] == last_complete_wy]["st"].iloc[-1])
                resume_start_wy = last_complete_wy + 1
                # Drop partial-WY rows from both outputs before appending fresh ones
                sim_existing[sim_existing["wy"] <= last_complete_wy].to_csv(
                    simulation_output_file, index=False
                )
                if os.path.exists(decision_output_file):
                    dec_existing = pd.read_csv(decision_output_file)
                    dec_existing[dec_existing["wy"] <= last_complete_wy].to_csv(
                        decision_output_file, index=False
                    )
                start_wy = resume_start_wy
                s0 = resume_s0
                with print_lock:
                    print(f"  [n={n}] Resuming from WY {start_wy} (s0={s0:.2f} TAF)")

    ny = end_wy - start_wy + 1

    # Each sample gets its own Reservoir and Operator
    R1 = Reservoir(characteristics=R1_characteristics)
    R1_agent = build_operator(
        resolved_model_config,
        R1,
        include_red_herring=args.include_red_herring,
        debug_response=args.debug_response,
        objective=args.objective,
        ablation_type=getattr(args, "ablation_type", None),
    )

    # record dataframes (one row per simulated day)
    R1.record = pd.DataFrame(index=range(ny * 365))
    R1_agent.record = pd.DataFrame(index=range(ny * 365))

    # initial allocation decision
    allocation_percent = 100
    t = 0

    # complex-mode state, held between monthly decisions
    complexity_mode = bool(getattr(R1, "complexity_mode", False))
    carryover_target = -1.0
    junior_delivery_percent = 100.0   # agent-set junior committed-M&I delivery (default full)
    meet_min_flow = True              # agent-set: deliver full min flow (default) or curtail to floor
    hydro_class = None
    near_term = 0.0
    wy_idx = 0.0
    ari = 0.0
    wf_factor = 1.0
    spill_vol = 0.0        # cumulative spill volume since Oct (TAF), reset each WY
    # release rate (TAF/day) above which releases count as spills for the ARI
    spill_threshold_taf = (
        utils.cfs_to_taf(R1.complexity.get("spill_threshold_cfs", 8000))
        if complexity_mode else 0.0
    )
    # Fraction of the required minimum flow that is a firm regulatory floor (always
    # released); the agent's meet_min_flow governs the discretionary remainder above it.
    min_flow_floor_frac = (
        float(R1.complexity.get("min_flow_decision", {}).get("floor_frac", 0.5))
        if complexity_mode else 0.0
    )

    prefix = f"[n={n}]"

    # period-of-record loop
    with print_lock:
        print(f"\n══ Simulation Start (sample {n}, WY {start_wy}\u2013{end_wy}) ═" + "═" * 20)
    for wy in np.arange(start_wy, end_wy + 1):
        with print_lock:
            print(f"  {prefix} Water year {wy}")
        # days of the water year (Oct 1 – Sep 30)
        date_range = pd.date_range(start=f"{wy-1}-10-01", end=f"{wy}-09-30", freq="D")
        # drop leap day (365 days/WY)
        if len(date_range) == 366:
            leap_day = (date_range.month == 2) & (date_range.day == 29)
            date_range = date_range[~leap_day]

        # ARI counts spills since October, so reset the running spill volume each WY
        spill_vol = 0.0

        for ty, d in enumerate(date_range):
            # month of water year (1 = October)
            mowy = d.month - 9 if d.month > 9 else d.month + 3

            # previous day's end-of-day storage
            st_1 = s0 if t == 0 else R1.record.loc[t - 1, "st"]

            # LLM decision on the first day of each month
            if d.day == 1 and args.model != 'release-demand':
                with print_lock:
                    print(f"    {prefix} Month {mowy:>2} \u2014 requesting allocation decision")

                # Compute the month-held forecast drivers before building the observation,
                # so the prompt shows the same ARI, min flow, and Water-Forum cutback the
                # daily physics will use.
                if complexity_mode:
                    wy_idx = R1.wy_inflow_index(d, wy, ty + 1)
                    ari = R1.american_river_index(wy_idx, spill_vol)
                    hydro_class = R1.classify_water_year(wy_idx)
                    near_term = R1.near_term_inflow(d)
                    wf_factor = R1.compute_water_forum_factor(wy_idx)
                    # Refresh the month's FMS minimum-flow state before the prompt and daily
                    # physics read it. Uses month-start storage (st_1 here = end of previous
                    # month) for the FRI / Jan-Feb triggers / off-ramp, and the held wy_index
                    # for the SRI tier and IFII.
                    R1.update_monthly_min_flow(mowy, st_1, wy_idx, wy=int(wy))

                R1_agent.set_observation(
                    idx=t, date=d, wy=wy, mowy=mowy, dowy=ty + 1,
                    alloc_1=allocation_percent, st_1=st_1,
                    spill_vol=spill_vol,
                )
                allocation_percent = R1_agent.make_allocation_decision(idx=t)

                # Secondary decision fields, held for the month's physics.
                if complexity_mode:
                    carryover_target = float(
                        getattr(R1_agent.last_decision, "carryover_target_taf", -1.0)
                    )
                    junior_delivery_percent = float(
                        getattr(R1_agent.last_decision, "junior_delivery_percent", 100.0)
                    )
                    meet_min_flow = bool(
                        getattr(R1_agent.last_decision, "meet_min_flow", True)
                    )

            # Downstream demand. demand.txt drives the simple baseline; in complex mode the
            # demand stack is built below from CalSim climatology, but dt is still recorded
            # for reference.
            dt = R1.demand[ty]
            if not complexity_mode:
                uu = dt * allocation_percent / 100.0

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
            # Observed evaporative loss (TAF/day), applied to the mass balance in complex mode
            # only. Missing/NaN -> 0.0 so inflow files without an `evap` column and
            # out-of-record dates fall back to the no-evap balance.
            evap_t = 0.0
            if complexity_mode and "evap" in inflow_rows.columns:
                _e = inflow_rows["evap"].iloc[0]
                evap_t = float(_e) if pd.notna(_e) else 0.0

            # daily physics
            if complexity_mode:
                min_flow = R1.compute_min_flow()
                tocs = R1.compute_tocs(
                    dowy=ty + 1, date=d.strftime("%Y-%m-%d"),
                    near_term=near_term,
                )
                # Carryover defense rations drawdown to the end of the active window: the
                # end-of-May fill peak in Feb–May (mowy 5–8), else the end of the WY.
                if 5 <= mowy <= 8:
                    window_end_dowy = sum(R1._WY_MONTH_DAYS[:8])
                else:
                    window_end_dowy = 365
                days_left = window_end_dowy - (ty + 1) + 1
                # Climatological inflow expected over the remaining window. Rationing the cap
                # NET of it front-loads the throttle less, so the discretionary supplemental
                # layer is delivered evenly rather than ramping up through the month (see
                # Reservoir.compute_carryover_release_cap).
                expected_window_inflow = R1.expected_window_inflow(ty + 1, window_end_dowy)
                release_cap = R1.compute_carryover_release_cap(
                    mowy, st_1, ari, days_left, carryover_target,
                    expected_inflow=expected_window_inflow,
                )
                # Demand stack (CalSim-grounded, additive — every layer served by the single
                # release rt). Priority order: minimum instream flow, firm senior committed
                # M&I, junior committed M&I, supplemental downstream-flow. The min flow splits
                # into a firm regulatory floor (always released) and a discretionary remainder
                # the agent may curtail (meet_min_flow); the agent also sets junior M&I and
                # supplemental delivery. All layers are demands, so under-delivery is a
                # shortage. (Lower-American M&I diverts below the dam, not a storage draw.)
                min_flow_full = min_flow                                     # required MIF (FMS)
                min_flow_floor = min_flow_full * min_flow_floor_frac         # firm floor (forced)
                min_flow_target = min_flow_full if meet_min_flow else min_flow_floor  # agent choice
                umi = R1.upstream_mi_day(mowy, ty + 1)
                umi_senior = umi * R1._upstream_mi_senior_frac               # firm, always served
                junior_full = umi * R1._upstream_mi_wf_cvp_frac              # full junior commitment
                junior_target = junior_full * junior_delivery_percent / 100.0   # agent-set delivery
                delta_demand = R1.delta_demand_day(mowy, wy_idx, st=st_1)  # supplemental: hydrology-indexed + low-storage relaxation (Nov-May)
                delta_target = delta_demand * allocation_percent / 100.0    # agent-set delivery
                firm_floor = min_flow_floor + umi_senior                    # protected (floor + senior)
                uu = min_flow_target + umi_senior + junior_target + delta_target  # total release target

                rt, st = R1.evaluate(
                    st_1=st_1, qt=qt, uu=uu, tocs=tocs,
                    min_flow=min_flow_floor, release_cap=release_cap, senior_floor=firm_floor,
                    # The carryover cap defends the target by curtailing the two lowest-priority
                    # supply layers in priority order — supplemental first, then junior committed
                    # M&I — down to cap_protect (the agent's chosen min flow + senior M&I, which
                    # the cap never cuts). Because the realized rt is attributed min flow → senior
                    # → junior → supplemental below, a binding cap zeroes supplemental before it
                    # touches junior. The target is breached only when even zero supplemental and
                    # zero junior cannot hold it.
                    cap_protect=min_flow_target + umi_senior,
                    # observed evaporative loss from storage (S = S_1 + Q − rt − E)
                    evaporation=evap_t,
                )
                # Attribute the realized release across the priority stack
                # (MIF → senior M&I → junior M&I → supplemental); any remainder is flood spill.
                served = rt
                mif_served = min(served, min_flow_target);   served -= mif_served
                senior_served = min(served, umi_senior);     served -= senior_served
                junior_served = min(served, junior_target);  served -= junior_served
                delta_served = min(served, delta_target);    served -= delta_served
                # Shortfalls measured against the FULL demands, so the agent's own curtailment
                # counts as a shortage (min_flow_short includes the discretionary cut plus any
                # dead-pool cut).
                min_flow_short = max(0.0, min_flow_full - mif_served)
                junior_short = max(0.0, junior_full - junior_served)
                delta_short = max(0.0, delta_demand - delta_served)
                committed_mi = senior_served + junior_served               # delivered upstream M&I
                # Release forced ABOVE the agent's target uu — only the flood curve (dynamic-TOCS
                # evacuation) or gross-pool capacity can do this. Logs the model's own dynamic-curve
                # flood evacuation, distinct from the regulatory fixed-WCD-gate spill computed in
                # the analysis layer.
                flood_release = max(0.0, rt - uu)
                # Accumulate spill volume (releases above the spill threshold) for the ARI.
                spill_vol += max(0.0, rt - spill_threshold_taf)
                R1.record_timestep(
                    idx=t, date=d, wy=wy, mowy=mowy, dowy=ty + 1, qt=qt, st=st, rt=rt, dt=dt, uu=uu,
                    min_flow=min_flow_full, tocs=tocs, hydro_class=hydro_class,
                    delta_demand=delta_demand, delta_delivered=delta_served,
                    delta_short=delta_short, committed_mi=committed_mi,
                    junior_delivery_pct=junior_delivery_percent, junior_short=junior_short,
                    near_term=near_term, release_cap=release_cap, flood_release=flood_release,
                    wf_factor=wf_factor, ari=ari, min_flow_short=min_flow_short, evap=evap_t,
                )
            else:
                tocs = R1.compute_tocs(dowy=ty + 1, date=d.strftime("%Y-%m-%d"))
                rt, st = R1.evaluate(st_1=st_1, qt=qt, uu=uu, tocs=tocs)
                R1.record_timestep(
                    idx=t, date=d, wy=wy, mowy=mowy, dowy=ty + 1, qt=qt, st=st, rt=rt, dt=dt, uu=uu
                )

            t += 1

            # On the last day of the month, append that month's records to the output files
            if d.day == date_range[date_range.month == d.month][-1].day:
                days_in_current_month = (date_range.month == d.month).sum()
                month_start_idx = t - days_in_current_month

                R1.record.loc[month_start_idx:t].dropna().to_csv(
                    simulation_output_file, index=False, mode='a',
                    header=not os.path.exists(simulation_output_file)
                )
                if args.model != "release-demand":
                    R1_agent.record.loc[month_start_idx:t].dropna(
                        subset=["allocation_percent"]
                    ).to_csv(
                        decision_output_file, quotechar='"', index=False, mode='a',
                        header=not os.path.exists(decision_output_file)
                    )

    with print_lock:
        print(f"  {prefix} Simulation complete")


def main():
    args = parse_args()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(file_dir, "..", "data")

    config_path = os.path.join(file_dir, "configs", args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    nsample = args.nsample

    # Resolve model/server selection and provider capabilities
    resolved_model_config = resolve_model_config(
        RunIntent(
            model_server=args.model_server,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
        )
    )
    model_server = resolved_model_config.model_server
    model = resolved_model_config.model
    model_kwargs = resolved_model_config.model_kwargs

    # Print resolved config (mask API keys)
    safe_model_kwargs = {
        key: ("***" if key.lower().endswith("_key") else value)
        for key, value in model_kwargs.items()
    }
    print("\n" + "─" * 60)
    print(f"  Model Server:  {model_server}")
    print(f"  Model:         {model}")
    for k, v in safe_model_kwargs.items():
        if k not in ("api_key",):
            print(f"  {k}: {v}")
    print("─" * 60)

    for warning in resolved_model_config.warnings:
        print(f"\n⚠  {warning}")
    if resolved_model_config.warnings:
        print()

    if args.tocs in ['fixed', 'historical', 'dynamic', 'dynamic_hist_cap']:
        tocs = args.tocs
    else:
        raise ValueError("TOCS must be 'fixed', 'historical', 'dynamic', or 'dynamic_hist_cap'")

    # complexity mode: CLI override, else config's complexity.enabled
    complexity_block = config.get("complexity")
    if args.complexity is None:
        complexity_mode = bool(complexity_block and complexity_block.get("enabled", False))
    else:
        complexity_mode = args.complexity
    if complexity_mode and complexity_block is None:
        raise ValueError("--complexity requires a 'complexity' block in the config")
    if tocs in ("dynamic", "dynamic_hist_cap"):
        if not complexity_mode:
            raise ValueError(f"--tocs {tocs} requires complexity mode (a 'complexity' config block)")
        if args.wy_forecast_file is None:
            raise ValueError(f"--tocs {tocs} requires --wy-forecast-file")
        if args.wy_monthly_forecast_file is None:
            raise ValueError(f"--tocs {tocs} requires --wy-monthly-forecast-file")
    if complexity_mode:
        print(f"  Complexity mode: ENABLED")

    # Objective default depends on mode: complex mode uses balance-delivery (delivery-first,
    # treats excess carryover as a cost rather than a co-equal goal, which had biased the
    # agent toward over-carrying water); simple mode keeps the legacy minimize-shortages.
    if args.objective is None:
        args.objective = "balance-delivery" if complexity_mode else "minimize-shortages"
    print(f"  Objective: {args.objective}")

    # --- RESERVOIR --- #
    R1_characteristics = {
        "tocs": tocs,
        "demand_file": os.path.join(data_dir, args.demand_file),
        "inflow_file": os.path.join(data_dir, args.inflow_file),
        "wy_forecast_file": os.path.join(data_dir, args.wy_forecast_file) if args.wy_forecast_file is not None else False,
        "wy_monthly_forecast_file": os.path.join(data_dir, args.wy_monthly_forecast_file) if args.wy_monthly_forecast_file is not None else None,
        "operable_storage_max": config["folsom_reservoir"]["operable_storage_max"],
        "operable_storage_min": config["folsom_reservoir"]["operable_storage_min"],
        "max_safe_release": utils.cfs_to_taf(config["folsom_reservoir"]["max_safe_release"]),
        "sp_to_ep": config["folsom_reservoir"]["sp_to_ep"],
        "tp_to_tocs": config["folsom_reservoir"]["tp_to_tocs"],
        "sp_to_rp": config["folsom_reservoir"]["sp_to_rp"],
        "complexity": complexity_block,
        "complexity_mode": complexity_mode,
    }

    # --- SIMULATION --- #
    print_lock = threading.Lock()

    if args.parallel is not None and nsample > 1:
        max_workers = args.parallel or nsample
        print(f"\nRunning {nsample} samples in parallel (max_workers={max_workers})")
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for n in range(args.sample_start, args.sample_start + nsample):
                future = executor.submit(
                    run_single_sample,
                    n, args, resolved_model_config, model,
                    R1_characteristics, file_dir, print_lock,
                )
                futures[future] = n
            for future in as_completed(futures):
                n = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"\n[n={n}] Sample failed with error: {exc}")
                    raise
    else:
        for n in range(args.sample_start, args.sample_start + nsample):
            run_single_sample(
                n, args, resolved_model_config, model,
                R1_characteristics, file_dir, print_lock,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Run reservoir simulation with LLM operator.")
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Name of configuration YAML file.",
    )
    parser.add_argument(
        "-n", "--nsample",
        type=int,
        default=1,
        help="Number of simulation samples to run.",
    )
    parser.add_argument(
        "--start-year",
        required=True,
        type=int,
        help="Start year of simulation (YYYY).",
    )
    parser.add_argument(
        "--end-year",
        required=True,
        type=int,
        help="End year of simulation (YYYY).",
    )
    parser.add_argument(
        "--demand-file",
        type=str,
        default="demand.txt",
        help="Demand file name (in ./data).",
    )
    parser.add_argument(
        "--inflow-file",
        type=str,
        default="folsom_daily.csv",
        help="Inflow file name (in ./data).",
    )
    parser.add_argument(
        "--wy-forecast-file",
        type=str,
        default=None,
        help="Water year forecast file name (in ./data).",
    )
    parser.add_argument(
        "--starting-storage",
        type=float,
        required=True,
        help="Initial storage level for the reservoir (TAF).",
    )
    parser.add_argument(
        "--tocs",
        type=str,
        default="fixed",
        help="How to handle TOCS (fixed, historical, dynamic, or dynamic_hist_cap). 'dynamic' and "
             "'dynamic_hist_cap' require complexity mode plus --wy-forecast-file and "
             "--wy-monthly-forecast-file. 'dynamic_hist_cap' adds a deep-winter relaxation that "
             "raises the limit to min(static WCD, observed storage) on flood-proximate days.",
    )
    parser.add_argument(
        "--wy-monthly-forecast-file",
        type=str,
        default=None,
        help="Near-term monthly inflow-outlook file name (in ./data); drives the "
             "dynamic flood curve in complexity mode.",
    )
    parser.add_argument(
        "--complexity",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable complex operating context (overrides the config's "
             "complexity.enabled). Default: config decides.",
    )
    parser.add_argument(
        "--model-server",
        required=True,
        default=None,
        help="Model server to use (Ollama or OpenAI)."
    )
    parser.add_argument(
        "--model",
        default=None,
        required=True,
        help="Model name/version.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        help="Reasoning effort for supported models: none, minimal, low, medium, high (Default: high).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default=None,
        choices=["minimize-shortages", "minimize-large-shortages-carryover", "balance-delivery"],
        help="Operator objective stated in the system prompt. Default: minimize-shortages in "
             "simple mode, balance-delivery in complex mode (delivery-first — meet supply/"
             "environmental demands as fully as possible, conserving only as needed for "
             "reliability, with excess carryover treated as a cost rather than a co-equal goal).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for local models.",
    )
    parser.add_argument(
        "--include-red-herring",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Include red herring in the context (Default: True).",
    )
    parser.add_argument(
        "--ablation-type",
        type=str,
        default=None,
        choices=list(ABLATION_TYPES),
        help=(
            "Option A: run the full sequential simulation with one element removed from "
            "every monthly prompt (e.g. min_flow, release_structure, forecasts, "
            "current_storage). Output filenames carry an 'abl-<type>' tag. Default: None "
            "(no ablation)."
        ),
    )
    parser.add_argument(
        "--debug-response",
        default=False,
        action="store_true",
        help="Capture raw model response payloads for inspection (Default: False).",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Resume from last complete water year in existing output files (Default: False).",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help="Starting sample index (Default: 0).",
    )
    parser.add_argument(
        "--parallel",
        nargs="?",
        const=0,
        default=None,
        type=int,
        metavar="N",
        help=(
            "Run samples in parallel. Optionally pass N to cap the worker thread "
            "count (Default: sequential; --parallel alone uses nsample workers)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
