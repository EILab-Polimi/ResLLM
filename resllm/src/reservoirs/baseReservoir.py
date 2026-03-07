#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.reservoir

Reservoir simulation classes for the resllm project.

"""

import os

import numpy as np
import pandas as pd
import src.utils as utils
import yaml
import autogen

class BaseReservoir:
    """
    Basic reservoir simulation
    """

    def __init__(
        self,
        name,
        config_path,
        prompt_config,
        model_and_version,
        policy,
    ):
        with open(os.path.join("configs", config_path), 'r') as f:
            characteristics = yaml.safe_load(f)
        self.characteristics = characteristics

        with open(os.path.join("configs", prompt_config), 'r') as f:
            self.prompt_config = yaml.safe_load(f)
            
        sys_parts = [
            self.prompt_config['persona'],
            self.prompt_config['hydro_goal'],
            self.prompt_config['location'],
            self.prompt_config['water_year'],
        ]
        self.system_message = "\n".join(filter(None, sys_parts))
        
        self.model_and_version = model_and_version
        self.policy = policy
        
        self.record = pd.DataFrame()

    def compute_max_release(self, S):
        """
        Computes the maximum release from the reservoir based on the current storage.
        Parameters:
            S (float): The storage in the reservoir.
        Returns:
            float: The maximum release from the reservoir.
        """
        # Step 1: Interpolate current elevation from current storage volume
        vol_array = self.characteristics["sp_to_ep"][0]
        elev_array = self.characteristics["sp_to_ep"][1]
        current_elevation = np.interp(S, vol_array, elev_array)

        # Step 2: Interpolate max release from the calculated elevation
        sp_elev = self.characteristics["sp_to_rp_max"][0]
        rp_release = self.characteristics["sp_to_rp_max"][1]
        
        return np.interp(current_elevation, sp_elev, rp_release) # m3/s

    def compute_min_release(self, S):
        """
        Computes the minimum release from the reservoir based on the current storage.
        Parameters:
            S (float): The storage in the reservoir.
        Returns:
            float: The minimum release from the reservoir.
        """
       # Step 1: Interpolate current elevation from current storage volume
        vol_array = self.characteristics["sp_to_ep"][0]
        elev_array = self.characteristics["sp_to_ep"][1]
        current_elevation = np.interp(S, vol_array, elev_array)

        # Step 2: Interpolate min release from the calculated elevation
        sp_elev = self.characteristics["sp_to_rp_min"][0]
        rp_release = self.characteristics["sp_to_rp_min"][1]
        
        return np.interp(current_elevation, sp_elev, rp_release) # m3/s

    def volume_to_height(self, S):
        """
        Converts the storage in m3 to height in feet based on storage-elevation relationship.
        Parameters:
            S (float): The storage in m3.
        Returns:
            float: The height in feet.
        """
        sp = self.characteristics["sp_to_ep"][0]
        ep = self.characteristics["sp_to_ep"][1]
        return np.interp(S, sp, ep)
    
    def volume_to_surface_area(self, S):
        """
        Converts the storage in m3 to surface area in m2 based on storage-surface area relationship.
        Parameters:
            S (float): The storage in m3.
        Returns:
            float: The surface area in m2.
        """
        sp = self.characteristics["sp_to_aa"][0]
        aa = self.characteristics["sp_to_aa"][1]
        return np.interp(S, sp, aa)

    def actual_release_daily(self, desired_release_m3s, S_m3, cmonth, n_sim_m3s, mef_m3s):
        """
        Calculates the physically permissible release for a SINGLE DAY.
        Everything here is strictly in m3/s.
        """
        qm = self.compute_min_release(S_m3)
        qM = self.compute_max_release(S_m3)

        # Constrain target between physical bounds
        rr = min(qM, max(qm, desired_release_m3s))

        # Base MEF logic (Meet environmental flows without exceeding max physical capacity)
        rr = min(qM, max(rr, mef_m3s))

        return rr

    def integration_daily(self, days_in_month, st_1_m3, desired_release_m3, n_sim_m3, current_month, evap_rate_mm=0.0, mef_m3s=0.0):
        """
        The core physics loop. 
        Takes inputs in total m3, does the daily math in m3/s, and returns totals in m3.
        """
        seconds_in_day = 86400
        seconds_in_month = days_in_month * seconds_in_day
        
        # 1. Convert Monthly Volumes (m3) to flat target Flow Rates (m3/s)
        desired_release_m3s = desired_release_m3 / seconds_in_month
        n_sim_m3s = n_sim_m3 / seconds_in_month

        # 2. Initialize tracking arrays
        s = np.zeros(days_in_month + 1)
        r_m3s = np.zeros(days_in_month)
        e_m3s = np.zeros(days_in_month)
        
        s[0] = st_1_m3 # Starting storage
        
        # 3. Step through every day of the month
        for i in range(days_in_month):
            
            # Get the physical release for today (in m3/s)
            r_m3s[i] = self.actual_release_daily(desired_release_m3s, s[i], current_month, n_sim_m3s, mef_m3s)
            
            # Calculate evaporation for today (in m3/s)
            if evap_rate_mm > 0:
                surface_area_m2 = self.volume_to_surface_area(s[i])
                e_m3s[i] = (evap_rate_mm / 1000.0) * surface_area_m2 / seconds_in_month
            else:
                e_m3s[i] = 0.0
                
            # System Transition: Storage tomorrow = Storage today + (Inflow - Outflow)*Seconds
            s[i+1] = s[i] + seconds_in_day * (n_sim_m3s - r_m3s[i] - e_m3s[i])
            
            # Failsafe
            if s[i+1] < 0:
                s[i+1] = 0.0

        # 4. Convert average daily rates back to total monthly volume (m3)
        final_storage_m3 = s[-1]
        total_release_m3 = np.mean(r_m3s) * seconds_in_month
        total_evap_m3 = np.mean(e_m3s) * seconds_in_month

        return final_storage_m3, total_release_m3, total_evap_m3
    
    def calculate_power(self, storage):
        raise NotImplementedError("Power calculation not implemented. Override this method if power generation is to be calculated based on storage.")
    
    def generate_context(self, average_cumulative_inflow_by_month, average_remaining_demand_by_month, total_demand, forecast_names):
        inflow_str = " | ".join([f"Month {m+1}: {val:,} cubic meters" for m, val in enumerate(average_cumulative_inflow_by_month)])
        demand_str = " | ".join([f"Month {m+1}: {val:,} cubic meters" for m, val in enumerate(average_remaining_demand_by_month)])

        context_message = pol["single_objective_instructions"] + "\n"
        context_message += pol["information"].format(
            max_storage=self.characteristics["operable_storage_max"],
            min_storage=self.characteristics["operable_storage_min"],
            total_demand=total_demand,
            inflow_string=inflow_str,
            demand_string=demand_str
        )

        if self.include_red_herring:
            context_message += "\n" + pol["red_herring"] + "\n"

        # --- CONSTRUCT TASK / JSON ROUTING ---
        concept_map = utils.concept_map.copy()
        for col in forecast_names:
            concept_map[col] = "<rank 0-4>"

        json_structure = json.dumps({
            "allocation_reasoning": "<string justification>",
            "allocation_percent": "<number 0-100>",
            "allocation_concept_importance": concept_map
        }, indent=4).replace('"<number 0-100>"', '<number 0-100>').replace('"<rank 0-4>"', '<rank 0-4>')

        task_cfg = self.prompt_config["task"]
        if self.model_and_version[0] == "DeepSeek_Local":
            context_message += "\n" + task_cfg["deepseek_instructions"].format(json_structure=json_structure)
        else:
            context_message += "\n" + task_cfg["standard_instructions"]
        return context_message
    
    def generate_observation(self, monthly_demands, forecast_dict, mowy, monthly_past_inflows, current_storage, max_safe_release, previous_allocation):

        current_demand = monthly_demands[mowy-1]
        total_demand = sum(monthly_demands)
        average_remaining_demand_by_month = [total_demand - sum(monthly_demands[:m]) for m in range(1, len(monthly_demands)+1)] 
        #qwyaccum = sum(monthly_past_inflows)

        context = self.generate_context(monthly_past_inflows, average_remaining_demand_by_month, total_demand, forecast_dict.keys())
        if self.model_and_version[0] == "DeepSeek_Local":
            obs_cfg = self.prompt_config["observation"]
            obs_parts = []

            obs_parts.append(obs_cfg["beginning_of_month"].format(mowy=int(mowy) if mowy is not None else 0))

            # if mowy > 1:
            #     obs_parts.append(obs_cfg["cumulative_inflow"].format(qwyaccum=int(qwyaccum) if qwyaccum is not None else 0))
            
            obs_parts.append(obs_cfg["current_storage"].format(st_1=int(current_storage) if st_1 is not None else 0))

            if max_safe_release is not None:
                obs_parts.append(obs_cfg["max_safe_release"].format(max_safe_release=int(max_safe_release)))

            if forecast_dict:
                for forecast_key, forecast_value in forecast_dict.items():
                    obs_parts.append(obs_cfg["forecast_item"].format(
                        forecast_key=forecast_key,
                        forecast_def=utils.forecast_columns_definitions.get(forecast_key, [""])[0],
                        forecast_val=float(forecast_value) if forecast_value is not None else 0,
                        forecast_unit=utils.forecast_columns_definitions.get(forecast_key, ["", ""])[1]
                    ))

            obs_parts.append(obs_cfg["current_month_demand"].format(current_demand=current_demand))

            obs_parts.append(obs_cfg["remaining_demand"].format(d_wy_rem=int(average_remaining_demand_by_month[mowy - 1]) if average_remaining_demand_by_month and mowy <= len(average_remaining_demand_by_month) else 0))

            if mowy >= 9:
                next_year_demand = int(sum(monthly_demands[:3]))
                obs_parts.append(obs_cfg["approaching_next_year"].format(next_year_demand=next_year_demand))

            obs_parts.append(obs_cfg["instruction"].format(alloc_1=int(previous_allocation) if alloc_1 is not None else 0))

        else:
            raise NotImplementedError("Only DeepSeek_Local model is currently supported in BaseReservoir. Please specify model_and_version as ('DeepSeek_Local', 'vX') where X is the version number.")
            

