#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.reservoir

Reservoir simulation classes for the resllm project.

"""

import numpy as np
import pandas as pd
import src.utils as utils


class BaseReservoir:
    """
    Basic reservoir simulation
    """

    def __init__(
        self,
        characteristics: dict = {},
    ):
        self.characteristics = characteristics

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

    def compute_tocs(self, dowy, date=None):
        """
        Computes the Top of Conservation Storage (TOCS) based on the current day of the water year
        or, if using the historical option, the date.
        Parameters:
            dowy (int): The current day of the water year (1-365/366).
            date (str): The date in 'YYYY-MM-DD' format.
        Returns:
            float: The Top of Conservation Storage (TOCS) in TAF.
        """
        tp = self.characteristics["tp_to_tocs"][0]
        tocs = self.characteristics["tp_to_tocs"][1]
        tocs = np.interp(dowy, tp, tocs)
        if self.tocs == "historical":
            hist_st = self.inflows.loc[
                (self.inflows["date"] == date), "storage"
            ].values[0]
            return np.max([tocs, hist_st])
        elif self.tocs == "fixed":
            return tocs
        else:
            return self.characteristics["operable_storage_max"]

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

    def compute_average_cumulative_inflow_by_month(self):
        raise NotImplementedError("Not needed.")
 
    def compute_average_remaining_demand_by_month(self):
        raise NotImplementedError("Not needed.")

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
