#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor

# resllm multi-agent imports
from multi_agent_simulation import MultiAgentZambeziSimulationRunner
# dotenv imports
from dotenv import load_dotenv
load_dotenv(verbose=True)

def run_simulation_wrapper(n, args, config, data_dir):
    """
    Wrapper function to initialize the physical reservoirs and run the multi-agent simulation.
    """
    try:
        sim = MultiAgentZambeziSimulationRunner(
            config=config, 
            simulation_number=n, 
            data_dir=data_dir, 
            policy=args.policy,
            hydro_goal=args.hydro_goal,
            irrigation_goal=args.irrigation_goal
        )
        sim.run()
        
    except Exception as e:
        print(f"[Sample {n}] Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Starting Multi-Agent Zambezi Reservoir Simulation...")
    args = parse_args()

    if not args.hydro_goal and not args.irrigation_goal:
        raise ValueError("At least one of --hydro-goal or --irrigation-goal must be True.")

    file_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(file_dir, "..", "data")

    config_path = os.path.join(file_dir, "configs", args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Ensure output directory exists
    if not os.path.exists("./output"):
        os.makedirs("./output")
    if not os.path.exists(f"./output/{args.extra_name}"):
        os.makedirs(f"./output/{args.extra_name}")

    nsample = args.nsample
    print(f"Running {nsample} simulation samples with policy: '{args.policy}'...")
    
    # Run Logic
    start_n = 0 
    
    if nsample == 1:
        print("Running a single simulation sample...")
        run_simulation_wrapper(start_n, args, config, data_dir)
    elif nsample > 1:
        print("Running simulations in parallel...")
        print(f"Using up to {args.max_workers} workers for parallel execution.")
        
        futures = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for i in range(start_n, start_n + nsample):
                futures.append(
                    executor.submit(
                        run_simulation_wrapper, i, args, config, data_dir
                    )
                )
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
        
    print("All simulations completed.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Multi-Agent Zambezi Reservoir Simulation.")
    
    # Execution & Config Settings
    parser.add_argument("-c", "--config", required=True, help="Name of configuration YAML file.")
    parser.add_argument("-n", "--nsample", type=int, default=1, help="Number of simulation samples to run.")
    parser.add_argument("--extra-name", type=str, default="", help="Extra name to append to output files.")
    parser.add_argument("--max-workers", type=int, default=6, help="Maximum number of parallel workers.")
    
    # Multi-Agent Policy Setting
    parser.add_argument(
        "--policy", 
        type=str, 
        default="self-interest", 
        choices=["self-interest", "collaborative-information-exchange", "collaborative-global"],
        help="The negotiation policy for the agents."
    )
    parser.add_argument(
        "--hydro-goal", 
        default=True, 
        help="Whether the agents prioritize hydroelectric power generation as their main goal."
    )
    parser.add_argument(
        "--irrigation-goal", 
        default=False, 
        help="Whether the agents prioritize irrigation as their main goal."
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()