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

# dotenv imports
from dotenv import load_dotenv
load_dotenv(verbose=True)

class ReservoirAgent:
    """
    Represents an individual reservoir agent that interacts with the environment and makes decisions based on its policy.
    """
    def __init__(self, name, config, reservoir, data_dir, start_date):
        self.config = config
        self.R = reservoir
        self.name = name
        self.storage = self.config['initial_storage']
        self.start_date = pd.to_datetime(start_date)
        self.current_commitment = None
        self.final_decision = False
        
        self.hydro_demand = np.loadtxt(self.config["demand_file"]) # demand, MWh
        self.hydro_demand = np.array([
                utils.twh_to_m3(d / 12, self.R.characteristics['turbines'], freq='M', month=i+1, year=2001) # Using a non-leap year for month length
                for i, d in enumerate(self.hydro_demand)
            ])  # demand per each month in cubic meters
        
        if self.config["wy_forecast_file"] is not False:
            self.forecasted_inflows = pd.read_csv(os.path.join(data_dir, self.config["wy_forecast_file"]))  # forecasted inflows, cubic meters
            self.forecasted_inflows["date"] = pd.to_datetime(self.forecasted_inflows["date"])
            if self.config['forecast_name'] is not False:
                forecasts = self.config['forecast_name'].split(' ')
                accepted_columns = [
                    col for col in self.forecasted_inflows.columns 
                    if any(f in col for f in forecasts)
                ]
                self.forecasted_inflows = self.forecasted_inflows[['date'] + accepted_columns]
                if self.config['forecast_locations'] is not False:
                    locations = self.config['forecast_locations'].split(' ')
                    selected_columns = [col for col in accepted_columns if any(loc in col for loc in locations)]
                    selected_columns.insert(0, 'date')  # Ensure 'date' column is included
                    self.forecasted_inflows = self.forecasted_inflows[selected_columns]
                else:
                    raise ValueError("Forecast locations must be provided if forecast name is specified.")
                print(f"Forecasted inflows filtered to columns: {self.forecasted_inflows.columns.tolist()}")
            else:
                print("No forecast name provided; using all available forecasts.")
            print(f"Forecasted inflows data loaded: {self.config['wy_forecast_file']}")
        
    def record_timestep(
        self,
        idx: int = 0,
        date: pd.Timestamp = None,
        wy: int = None,
        mowy: int = None,
        dowy: int = None,
        qt: float = None,
        st: float = None,
        rt: float = None,
        dt: float = None,
        uu: float = None,
    ):
        """
        Records the simulation output for the reservoir.
        """
        self.record.loc[idx, "date"] = date
        self.record.loc[idx, "wy"] = wy
        self.record.loc[idx, "mowy"] = mowy
        self.record.loc[idx, "dowy"] = dowy
        self.record.loc[idx, "qt"] = qt
        self.record.loc[idx, "st"] = st
        self.record.loc[idx, "rt"] = rt
        self.record.loc[idx, "dt"] = dt
        self.record.loc[idx, "uu"] = uu 
    
    def commit_action(self, release_volume_m3: float) -> str:
        """The tool the LLM calls to lock in its decision."""
        self.current_commitment = release_volume_m3
        return f"SUCCESS: {self.name} has committed to releasing {release_volume_m3} m3."

    def get_storage(self):
        return self.storage