#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import numpy as np
import pandas as pd

# resllm imports
from src.reservoir import Reservoir
from src.operator import build_operator
from src.model_config import RunIntent, resolve_model_config
import src.utils as utils

# dotenv imports
from dotenv import load_dotenv
load_dotenv(verbose=True)

def main():
    args = parse_args()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(file_dir, "..", "data")

    # Resolve config path
    config_path = os.path.join(file_dir, "configs", args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Samples
    nsample = args.nsample

    # Model/server selection and provider capability resolution
    resolved_model_config = resolve_model_config(
        RunIntent(
            model_server=args.model_server,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
            include_logprobs=args.include_logprobs,
        )
    )
    model_server = resolved_model_config.model_server
    model = resolved_model_config.model
    model_kwargs = resolved_model_config.model_kwargs

    # Print resolved config
    safe_model_kwargs = {
        key: ("***" if "key" in key.lower() else value)
        for key, value in model_kwargs.items()
    }
    print("\n" + "─" * 60)
    print(f"  Model Server:  {model_server}")
    print(f"  Model:         {model}")
    print(f"  Logprobs:      {resolved_model_config.top_logprobs or 'off'}")
    for k, v in safe_model_kwargs.items():
        if k not in ("api_key",):
            print(f"  {k}: {v}")
    print("─" * 60)

    # Print warnings after config summary
    for warning in resolved_model_config.warnings:
        print(f"\n⚠  {warning}")
    if resolved_model_config.warnings:
        print()

    # tocs option
    if args.tocs in ['fixed','historical']:
        tocs = args.tocs
    else:
        raise ValueError("TOCS must be either 'fixed' or 'historical'")
    
    # --- RESERVOIR --- #
    R1_characteristics = {
        "tocs": tocs,
        "demand_file": os.path.join(data_dir, args.demand_file),
        "inflow_file": os.path.join(data_dir, args.inflow_file),
        "wy_forecast_file": os.path.join(data_dir, args.wy_forecast_file) if args.wy_forecast_file is not None else False,
        "operable_storage_max": config["folsom_reservoir"]["operable_storage_max"],
        "operable_storage_min": config["folsom_reservoir"]["operable_storage_min"],
        "max_safe_release": utils.cfs_to_taf(config["folsom_reservoir"]["max_safe_release"]),
        "sp_to_ep": config["folsom_reservoir"]["sp_to_ep"],
        "tp_to_tocs": config["folsom_reservoir"]["tp_to_tocs"],
        "sp_to_rp": config["folsom_reservoir"]["sp_to_rp"],
    }

    # Create reservoir
    R1 = Reservoir(characteristics=R1_characteristics)

    # --- AGENT --- #
    R1_agent = build_operator(
        resolved_model_config,
        R1,
        include_red_herring=args.include_red_herring,
        debug_response=args.debug_response,
    )

    # --- SIMULATION --- #
    start_wy = args.start_year
    end_wy = args.end_year
    s0 = args.starting_storage
    ny = end_wy - start_wy + 1
    for n in range(nsample):
        # simulation dataframes
        R1.record = pd.DataFrame(index=range(ny * 365))
        R1_agent.record = pd.DataFrame(index=range(ny * 365))

        # set the initial allocation decision
        allocation_percent = 100
        t = 0

        # Setup output filenames
        safe_model_name = model.replace(":", "-")
        output_dir = os.path.join(file_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        simulation_output_file = os.path.join(
            output_dir, f"{safe_model_name}_simulation_output_n{n}.csv"
        )
        decision_output_file = os.path.join(
            output_dir, f"{safe_model_name}_decision_output_n{n}.csv"
        )
        logprobs_output_file = os.path.join(
            output_dir, f"{safe_model_name}_logprobs_output_n{n}.csv"
        )

        # period of record loop
        print(f"\n══ Simulation Start (sample {n}, WY {start_wy}–{end_wy}) ═" + "═" * 20)
        for wy in np.arange(start_wy, end_wy + 1):
            print(f"  Water year {wy}")
            # date range for the water year
            date_range = pd.date_range(start=f"{wy-1}-10-01", end=f"{wy}-09-30", freq="D")
            # remove leap day
            if len(date_range) == 366:
                leap_day = (date_range.month == 2) & (date_range.day == 29)
                date_range = date_range[~leap_day]

            # loop through the days of the water year
            for ty, d in enumerate(date_range):
                # get the month of the water year
                mowy = d.month - 9 if d.month > 9 else d.month + 3

                # SIMULATE RESERVOIR
                # - Observation
                st_1 = s0 if t == 0 else R1.record.loc[t - 1, "st"]

                # - LLM Decision at the start of each month
                if d.day == 1 and not args.model=='release-demand':
                    print(f"    Month {mowy:>2} — requesting allocation decision")
                    R1_agent.set_observation(
                        idx=t, date=d, wy=wy, mowy=mowy, dowy=ty + 1, alloc_1=allocation_percent, st_1=st_1
                    )
                    allocation_percent, _, _ = R1_agent.make_allocation_decision(idx=t)

                # current downstream demand
                dt = R1.demand[ty]
                # set target demand from allocation decision
                uu = dt * allocation_percent / 100.0

                # inflow
                inflow_rows = R1.inflows.loc[
                    (R1.inflows["water_year"] == wy)
                    & (R1.inflows["month"] == d.month)
                    & (R1.inflows["day"] == d.day),
                    "inflow",
                ]
                if inflow_rows.empty:
                    raise ValueError(
                        f"Missing inflow for date={d.strftime('%Y-%m-%d')} (WY={wy})"
                    )
                qt = float(inflow_rows.iloc[0])

                # TOCS and evaluate
                tocs = R1.compute_tocs(dowy=ty + 1, date=d.strftime("%Y-%m-%d"))
                rt, st = R1.evaluate(st_1=st_1, qt=qt, uu=uu, tocs=tocs)

                # record the timestep
                R1.record_timestep(
                    idx=t, date=d, wy=wy, mowy=mowy, dowy=ty + 1, qt=qt, st=st, rt=rt, dt=dt, uu=uu
                )

                # increment timestep
                t += 1

                # Save outputs at the end of each month
                if d.day == date_range[date_range.month == d.month][-1].day:
                    # Calculate the start index of the current month
                    days_in_current_month = (date_range.month == d.month).sum()
                    month_start_idx = t - days_in_current_month
                    
                    # Append only the current month's records
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
                    logprobs_df = R1_agent.pop_logprobs_record()
                    if not logprobs_df.empty:
                        logprobs_df.to_csv(
                            logprobs_output_file, index=False, mode='a',
                            header=not os.path.exists(logprobs_output_file)
                        )

        # Outputs are already saved incrementally at the end of each month
        print("Simulation complete")


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
        help="How to handle TOCS (fixed or historical).",
    )
    parser.add_argument(
        "--model-server",
        required=True,
        default=None,
        help="Model server to use (e.g., Ollama, OpenAI, xAI)."
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
        "--temperature",
        type=float,
        default=None,
        help="Temperature for local models.",
    )
    parser.add_argument(
        "--include-red-herring",
        default=True,
        action="store_true",
        help="Include red herring in the context (Default: True).",
    )
    parser.add_argument(
        "--debug-response",
        default=False,
        action="store_true",
        help="Capture raw model response payloads for inspection (Default: False).",
    )
    parser.add_argument(
        "--include-logprobs",
        default=None,
        type=int,
        help="Include top N log probabilities in output (OpenAI 0-5, Ollama 0-20, Default: None).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
