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
from src.reservoirs.baseReservoir import BaseReservoir
import src.utils as utils
import autogen

# dotenv imports
from dotenv import load_dotenv
load_dotenv(verbose=True)

class ReservoirAgent:
    """
    Represents an individual reservoir agent that interacts with the environment and makes decisions based on its policy.
    """
    def __init__(
        self,
        name: str,
        reservoir_logic: BaseReservoir,
        data_dir: str,
        config: dict,
        forecasted_inflows: pd.DataFrame,
        upstream_inflows: list,
        llm_configs: dict,
    ):
    
        self.name = name
        self.reservoir_logic = reservoir_logic

        self.config = config

        self.monthly_demands = np.loadtxt(os.path.join(data_dir, self.config["demand_file"])) # TWh
        self.monthly_demands = self.monthly_demands = np.array([
                utils.twh_to_m3(d, self.reservoir_logic.characteristics['reservoir']["turbines"], freq='M', month=i+1, year=2001) # Using a non-leap year for month length
                for i, d in enumerate(self.monthly_demands)
            ])  # demand, cubic meters

        available_forecasts = self.config['forecast_name'].split(' ')
        available_locations = self.config['forecast_locations'].split(' ')
        self.forecasted_inflows = forecasted_inflows[
            ['date'] + [col for col in forecasted_inflows.columns if any(f in col for f in available_forecasts) and any(loc in col for loc in available_locations)]
        ]

        self.inflows = pd.DataFrame(columns=['date', 'total_inflow'] + upstream_inflows)  # This will hold the actual inflows for the current water year, updated monthly
        self.storage = self.config['starting_storage']
        self.allocations = pd.DataFrame(columns=['date', 'allocation'])  # This will hold the agent's monthly allocations/commitments
    
        self.current_commitment = None

        self.llm_agent = autogen.AssistantAgent(
            name=name,
            system_message=self.reservoir_logic.system_message,
            llm_config=llm_configs,
        )
    
    def get_monthly_demands(self):
        return self.monthly_demands

    def get_forecasts_for_date(self, date):
        forecasts_for_date = self.forecasted_inflows[self.forecasted_inflows['date'] == date]
        if forecasts_for_date.empty:
            return {}
        else:
            return forecasts_for_date.to_dict(orient='records')[0]
        
    def get_total_monthly_past_inflows(self, wy):
        average_monthly_inflows = self.inflows[self.inflows['date'].dt.year == wy].groupby(self.inflows['date'].dt.month).mean()
        return average_monthly_inflows.to_dict(orient='index')

    def set_monthly_inflow(self, date, inflow_dict):
        self.inflows = self.inflows.append({
            'date': date,
            'total_inflow': sum(inflow_dict[key] for key in inflow_dict),
            **inflow_dict
        }, ignore_index=True)

    def set_allocation(self, date, allocation):
        self.allocations = self.allocations.append({
            'date': date,
            'allocation': allocation
        }, ignore_index=True)
        self.current_commitment = allocation
    
    def get_previous_allocations(self, num_months=3):
        return self.allocations.tail(num_months)['allocation'].values

        