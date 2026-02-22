#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import date
import pickle
from concurrent.futures import ThreadPoolExecutor

# resllm imports
from src.reservoir import Reservoir
from src.operator import ReservoirAllocationOperator
import src.utils as utils

cum_days = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

# dotenv imports
from dotenv import load_dotenv
load_dotenv(verbose=True)

class SimulationRunner:
    """
    Handles the simulation state, checkpointing (pickle), and resuming logic.
    """
    def __init__(self, n, args, config, data_dir, model_server, model_kwargs, tocs):
        # 1. STORE CONFIGURATION
        self.n = n
        self.args = args
        self.config = config
        self.data_dir = data_dir
        self.model_server = model_server
        self.model_kwargs = model_kwargs
        self.tocs = tocs
        
        # 2. INITIALIZE STATE VARIABLES
        self.t = 0
        self.allocation_percent = 100
        self.current_wy = args.start_year
        self.next_resume_date = None  
        
        # 3. SETUP OUTPUT PATHS
        safe_model_name = args.model.replace(":", "-")
        self.sim_out_file = f"./output/{args.extra_name}/{safe_model_name}_simulation_output_{args.extra_name}_n{n}.csv"
        self.dec_out_file = f"./output/{args.extra_name}/{safe_model_name}_decision_output_{args.extra_name}_n{n}.csv"
        self.checkpoint_file = f"./output/{args.extra_name}/checkpoint_n{n}.pkl"
        
        # 4. INITIALIZE RESERVOIR & AGENT
        self._init_system()
        
        # 5. INITIALIZE RECORDS
        ny = args.end_year - args.start_year + 1
        total_steps = ny * args.steps_per_year
        self.R1.record = pd.DataFrame(index=range(total_steps))
        self.R1_agent.record = pd.DataFrame(index=range(total_steps))

    def _init_system(self):
        """Helper to set up R1 and Agent. Kept separate for clarity."""
        R1_characteristics = {
            "steps_per_year": self.args.steps_per_year,
            "tocs": self.tocs,
            "demand_file": os.path.join(self.data_dir, self.args.demand_file),
            "inflow_file": os.path.join(self.data_dir, self.args.inflow_file),
            "wy_forecast_file": os.path.join(self.data_dir, self.args.wy_forecast_file) if self.args.wy_forecast_file else False,
            "operable_storage_max": self.config["reservoir"]["operable_storage_max"],
            "operable_storage_min": self.config["reservoir"]["operable_storage_min"],
            "max_safe_release": utils.cfs_to_taf(self.config["reservoir"]["max_safe_release"]),
            "sp_to_ep": self.config["reservoir"]["sp_to_ep"],
            "tp_to_tocs": self.config["reservoir"]["tp_to_tocs"],
            "sp_to_rp_min": self.config["reservoir"]["sp_to_rp_min"],
            "sp_to_rp_max": self.config["reservoir"]["sp_to_rp_max"],
            "forecast_name": self.args.forecast_name,
            "forecast_locations": self.args.forecast_locations,
        }
        self.R1 = Reservoir(characteristics=R1_characteristics)
        
        self.R1_agent = ReservoirAllocationOperator(
            model_and_version=[self.model_server, self.args.model],
            reservoir=self.R1,
            model_kwargs=self.model_kwargs,
            include_double_check=self.args.include_double_check,
            include_num_history=self.args.include_num_history,
            include_red_herring=self.args.include_red_herring,
        )

    # --- MAGIC METHODS FOR PICKLING (THE FIX) ---
    def __getstate__(self):
        """Called when pickling. We remove the unpicklable Agent object."""
        state = self.__dict__.copy()
        
        # 1. Remove the objects that contain locks (Agent and Reservoir)
        del state['R1']
        del state['R1_agent']
        
        # 2. But SAVE their data (DataFrames) so we don't lose progress
        state['saved_R1_record'] = self.R1.record
        state['saved_agent_record'] = self.R1_agent.record
        
        return state

    def __setstate__(self, state):
        """Called when unpickling (resuming). We rebuild the Agent."""
        # 1. Restore the configuration
        self.__dict__.update(state)
        
        # 2. Re-initialize the Agent and Reservoir (creates new fresh connections)
        self._init_system()
        
        # 3. Restore the history (DataFrames) into the new objects
        self.R1.record = state['saved_R1_record']
        self.R1_agent.record = state['saved_agent_record']
        
        # 4. Clean up the temporary storage keys
        del self.saved_R1_record
        del self.saved_agent_record
    # --------------------------------------------

    @staticmethod
    def load_or_create(n, args, config, data_dir, model_server, model_kwargs, tocs):
        checkpoint_path = f"./output/{args.extra_name}/checkpoint_n{n}.pkl"
        safe_model_name = args.model.replace(":", "-")
        sim_out_file = f"./output/{args.extra_name}/{safe_model_name}_simulation_output_{args.extra_name}_n{n}.csv"
        
        if os.path.exists(checkpoint_path):
            print(f"[Sample {n}] 🔄 Found checkpoint. Resuming...")
            try:
                with open(checkpoint_path, 'rb') as f:
                    instance = pickle.load(f)
                return instance
            except Exception as e:
                print(f"[Sample {n}] ⚠️ Failed to load checkpoint ({e}). Checking if run is complete...")

        if os.path.exists(sim_out_file):
            print(f"[Sample {n}] ⚠️ No checkpoint found but output file exists. Checking if run is complete...")
            print(f"[Sample {n}] ⚠️ {sim_out_file} exists. Attempting to read last water year...")
            try:
                df = pd.read_csv(sim_out_file)
                if not df.empty and "wy" in df.columns:
                    last_wy = df["wy"].iloc[-1]
                    if last_wy >= args.end_year: 
                        print(f"[Sample {n}] ✅ Output file appears complete. Ending date {last_wy}. Skipping.")
                        return None 
            except Exception:
                pass 

        print(f"[Sample {n}] ⚡ Starting fresh simulation...")
        return SimulationRunner(n, args, config, data_dir, model_server, model_kwargs, tocs)

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(self, f)

    def run(self):
        freq = "D" if self.args.steps_per_year == 365 else "MS"
        
        for wy in np.arange(self.current_wy, self.args.end_year + 1):
            self.current_wy = wy 
            print(f"[Sample {self.n}] Simulating water year {wy}")

            date_range = pd.date_range(start=f"{wy}-01-01", end=f"{wy}-12-31", freq=freq)
            if self.args.steps_per_year == 365 and len(date_range) == 366:
                leap_day = (date_range.month == 2) & (date_range.day == 29)
                date_range = date_range[~leap_day]

            if self.next_resume_date is not None:
                resume_ts = pd.Timestamp(self.next_resume_date)
                if resume_ts.year == wy:
                    date_range = date_range[date_range >= resume_ts]
                if not date_range.empty:
                    self.next_resume_date = None

            for ty, d in enumerate(date_range):
                self._step(ty, d, wy, freq)
                
                should_save = (self.args.steps_per_year == 12) or (d.day == d.days_in_month)
                
                if should_save:
                    self._save_csvs(d)
                    if freq == "D":
                        self.next_resume_date = d + pd.Timedelta(days=1)
                    else:
                        self.next_resume_date = d + pd.DateOffset(months=1)
                    self.save_checkpoint()

        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        print(f"[Sample {self.n}] Simulation complete.")

    def _step(self, ty, d, wy, freq):
        mowy = d.month

        st_1 = self.args.starting_storage if self.t == 0 else self.R1.record.loc[self.t - 1, "st"]
        is_decision_step = (self.args.steps_per_year == 12) or (d.day == 1)

        days_passed = ty if self.args.steps_per_year == 365 else cum_days[mowy]
        tocs = self.R1.compute_tocs(dowy=days_passed + 1, date=d.strftime("%Y-%m-%d")) + self.R1.characteristics["operable_storage_min"]
        max_safe_release = utils.m3s_to_m3(self.R1.compute_max_release(st_1), freq=freq[0], month=mowy, year=wy)

        if is_decision_step and not self.args.model=='release-demand':
            print(f"[Sample {self.n}] Setting allocation decision for month {mowy} of water year {wy}")
            self.R1_agent.set_observation(
                idx=self.t, date=d, wy=wy, mowy=mowy, dowy=ty + 1, alloc_1=self.allocation_percent, 
                st_1=st_1, tocs=tocs, max_safe_release=max_safe_release
            )
            self.allocation_percent, _, _ = self.R1_agent.make_allocation_decision(idx=self.t)

        dt = self.R1.demand[ty] 
        uu = dt * self.allocation_percent / 100.0
        qt = self.R1.inflows.loc[
            (self.R1.inflows["water_year"] == wy)
            & (self.R1.inflows["month"] == d.month)
            & (self.R1.inflows["day"] == d.day),
            "inflow",
        ].values[0]

        rt, st = self.R1.evaluate(st_1=st_1, qt=qt, uu=uu, tocs=tocs, freq=freq[0], month=mowy, year=wy)

        if freq[0] == "M":
            dowy_rec = date(wy, mowy, 1).timetuple().tm_yday
        else:
            dowy_rec = ty + 1

        self.R1.record_timestep(
            idx=self.t, date=d, wy=wy, mowy=mowy, dowy=dowy_rec, qt=qt, st=st, rt=rt, dt=dt, uu=uu
        )
        self.t += 1

    def _save_csvs(self, d):
        start_slice = self.t - 1 if self.args.steps_per_year == 12 else self.t - d.days_in_month
        start_slice = max(0, start_slice)
        
        self.R1.record.loc[start_slice:self.t-1].dropna().to_csv(
            self.sim_out_file, index=False, mode='a', 
            header=not os.path.exists(self.sim_out_file)
        )
        
        if self.args.model != "release-demand":
            self.R1_agent.record.loc[start_slice:self.t-1].dropna().to_csv(
                self.dec_out_file, quotechar='"', index=False, mode='a', 
                header=not os.path.exists(self.dec_out_file)
            )

def run_simulation_wrapper(n, args, config, data_dir, model_server, model_kwargs, tocs):
    """Wrapper function to initialize and run the simulation class."""
    sim = SimulationRunner.load_or_create(n, args, config, data_dir, model_server, model_kwargs, tocs)
    if sim is not None:
        sim.run()

def main():
    print("Starting reservoir simulation from the python script...")
    args = parse_args()

    file_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(file_dir, "..", "data")

    # Resolve config path
    config_path = os.path.join(file_dir, "configs", args.config)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Samples
    nsample = args.nsample

    # Model/server selection
    model_server = args.model_server
    if model_server=='Ollama':
        model_kwargs = {
            "temperature": args.temperature,
            "think": True,
            "timeout": 3600  # 60 minutes
        }
    elif model_server == "Google":
        model_kwargs = {
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "temperature": args.temperature,
            "thinking_level": args.reasoning_effort
        }
    elif model_server == "OpenAI":
        if args.reasoning_effort == "na":
            model_kwargs = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": args.temperature,
            }
        else:   
            model_kwargs = {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "reasoning": {"effort": args.reasoning_effort},
        }
    elif model_server == "xAI":
        model_kwargs = {
            "api_key": os.getenv("XAI_API_KEY"),
            "temperature": args.temperature
        }
    elif model_server == "Mistral":
        model_kwargs = {
            "api_key": os.getenv("MISTRAL_API_KEY"),
            "temperature": args.temperature
        }
    elif model_server == "DeepSeek_Local":
        model_kwargs = {
            "api_key": "EMPTY",
            "base_url": args.hosting_server,
            "temperature": args.temperature
        }
    else:
        raise ValueError(f"Unsupported model server: {model_server}")

    # tocs option
    if args.tocs in ['fixed','historical']:
        tocs = args.tocs
    else:
        raise ValueError("TOCS must be either 'fixed' or 'historical'")

    # Ensure output directory exists
    if not os.path.exists("./output"):
        os.makedirs("./output")
    if not os.path.exists(f"./output/{args.extra_name}"):
        os.makedirs(f"./output/{args.extra_name}")

    print(f"Running {nsample} simulation samples...")
    
    # Run Logic
    start_n = 0 # Default starting index
    
    if nsample == 1:
        print("Running a single simulation sample...")
        run_simulation_wrapper(start_n, args, config, data_dir, model_server, model_kwargs, tocs)
    elif nsample > 1:
        print("Running simulations in parallel...")
        print(f"Using up to {args.max_workers} workers for parallel execution.")
        
        futures = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for i in range(start_n, start_n + nsample):
                futures.append(
                    executor.submit(
                        run_simulation_wrapper, i, args, config, data_dir, model_server, model_kwargs, tocs
                    )
                )
            # Wait for all futures to complete
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
        
    print("All simulations completed.")

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
        "--extra-name",
        type=str,
        default="",
        help="Extra name to append to output files.",
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
        help="(OpenAI reasoning/hybrid models (e.g., GPT 5.1 or o4-mini) reasoning effort level: none (GPT 5.1), low, medium, high.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for local models.",
    )
    parser.add_argument(
        "--include-double-check",
        default=False,
        action="store_true",
        help="Include double check on the decision (Default: False).",
    )
    parser.add_argument(
        "--include-num-history",
        default=0,
        type=int,
        help="Include number of historical timesteps in the context (Default: 0).",
    )
    parser.add_argument(
        "--include-red-herring",
        default=True,
        action="store_true",
        help="Include red herring in the context (Default: True).",
    )
    parser.add_argument(
        "--steps-per-year",
        type=int,
        default=12,
        help="Number of simulation steps per year (e.g., 12 for monthly, 365 for daily).",
    )
    parser.add_argument(
        "--hosting-server",
        type=str,
        default=None,
        help="If using a hosted model server, provide the base URL (e.g., for DeepSeek_Local).",
    )
    parser.add_argument(
        "--forecast-name",
        type=str,
        default=None,
        help="Forecast possible names: None, 1a, 1b, 2b, 3a, 3b, 4a, 4b. Use space to separate multiple forecasts (e.g., '1a 3b').",
    )
    parser.add_argument(
        "--forecast-locations",
        type=str,
        default=None,
        help="Forecast locations possible names: None, Average, Kafue, Luangwa, Mangochi, Sum, Victoria, ZRB. Use space to separate multiple locations (e.g., 'Kafue Luangwa').",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Maximum number of parallel workers for multiple simulations.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    main()