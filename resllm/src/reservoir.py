#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.reservoir

Reservoir simulation classes for the resllm project.

"""

import numpy as np
import pandas as pd
import src.utils as utils


class Reservoir:
    """
    Basic reservoir simulation
    """

    def __init__(
        self,
        characteristics: dict = {},
    ):
        """
        Initializes the reservoir simulation.
        Parameters:
            characteristics (dict): A dictionary containing reservoir characteristics.
                - steps_per_year (int): Number of time steps per year.
                - inflow_file (str): Path to the inflow data file.
                - demand_file (str): Path to the demand data file.
                - operable_storage_max (float): Maximum operable storage in cubic meters.
                - operable_storage_min (float): Minimum operable storage in cubic meters.
                - max_safe_release (float):  Maximum safe release in cubic meters.
                - sp_to_rp_min (list): List of storage to release points (minimum).
                - sp_to_rp_max (list): List of storage to release points (maximum).
                - sp_to_ep (list): List of storage to elevation points.
                - tp_to_tocs (list): List of top of conservation storage points.
        """

        self.steps_per_year = characteristics["steps_per_year"]

        self.tocs = characteristics["tocs"]
        self.demand = np.loadtxt(characteristics["demand_file"]) # demand, cubic meters
        print(f"Demand data loaded: {characteristics['demand_file']}")

        self.inflows = pd.read_csv(characteristics["inflow_file"])  # inflow, cubic meters
        if characteristics["wy_forecast_file"] is not False:
            self.forecasted_inflows = pd.read_csv(characteristics["wy_forecast_file"])  # forecasted inflows, cubic meters
            self.forecasted_inflows["date"] = pd.to_datetime(self.forecasted_inflows["date"])
            if characteristics['forecast_name'] is not False:
                print(f"Filtering forecasted inflows for forecasts: {characteristics['forecast_name']}")
                forecasts = characteristics['forecast_name'].split(' ')
                accepted_columns = [
                    col for col in self.forecasted_inflows.columns 
                    if any(f in col for f in forecasts)
                ]
                self.forecasted_inflows = self.forecasted_inflows[['date'] + accepted_columns]
                if characteristics['forecast_locations'] is not False:
                    locations = characteristics['forecast_locations'].split(' ')
                    selected_columns = [col for col in accepted_columns if any(loc in col for loc in locations)]
                    selected_columns.insert(0, 'date')  # Ensure 'date' column is included
                    self.forecasted_inflows = self.forecasted_inflows[selected_columns]
                else:
                    raise ValueError("Forecast locations must be provided if forecast name is specified.")
                print(f"Forecasted inflows filtered to columns: {self.forecasted_inflows.columns.tolist()}")
            else:
                print("No forecast name provided; using all available forecasts.")
            print(f"Forecasted inflows data loaded: {characteristics['wy_forecast_file']}")
            

        # apply date metadata to inflow data
        self.inflows["date"] = pd.to_datetime(self.inflows["date"])
        self.inflows["year"] = self.inflows["date"].dt.year
        self.inflows["month"] = self.inflows["date"].dt.month
        self.inflows["day"] = self.inflows["date"].dt.day
        self.inflows["doy"] = self.inflows["date"].dt.dayofyear
        self.inflows["dowy"] = self.inflows.apply(
            lambda row: utils.water_day(row["doy"]) + 1, axis=1
        )
        if self.steps_per_year == 365:
            self.inflows["week"] = self.inflows["dowy"].apply(
                lambda x: int((x - 1) / 7) + 1
            )
            self.inflows.loc[
                (self.inflows["month"] == 10) & (self.inflows["day"] == 1), "week"
            ] = 1
        else:
            self.inflows["week"] = 1 # Placeholder for monthly

        # Water year from Oct 1 to Sept 30
        # self.inflows["water_year"] = np.where(
        #     self.inflows["month"] >= 10,
        #     self.inflows["year"] + 1,
        #     self.inflows["year"],
        # )

        # Water year from Jan 1 to Dec 31
        self.inflows["water_year"] = self.inflows["year"]
        self.inflows["date"] = self.inflows["date"].dt.strftime("%Y-%m-%d")
        print(f"Inflow data loaded: {characteristics["inflow_file"]}")

        self.characteristics = characteristics
        self.characteristics["average_water_year_total_demand"] = int(self.demand.sum())
        self.characteristics["average_remaining_demand_by_month"] = (
            self.compute_average_remaining_demand_by_month()
        )
        self.characteristics["average_cumulative_inflow_by_month"] = (
            self.compute_average_cumulative_inflow_by_month()
        )

        self.record = pd.DataFrame()

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

    def evaluate(self, st_1, qt, uu, tocs, freq, month=None, year=None):
        """
        Evaluates the reservoir release and storage based on the current state, inflow,
        target release, and day of water year (DOWY).
        Parameters:
            st_1 (float): The storage in the reservoir at the end of the previous time step.
            qt (float): The inflow to the reservoir during the current time step.
            uu (float): The desired release from the reservoir.
            dowy (int): The current day of the water year (1-365/366).
        Returns:
            list: A list containing:
                - rt (float): The actual release from the reservoir.
                - st (float): The storage in the reservoir at the end of the current time step.
        Notes:
            - The target release is adjusted to release flood water
              in accordance with the Top of Conservation Storage (TOCS) constraint.
        """
        K = self.characteristics["operable_storage_max"]

        # constrain by TOCS
        rt = max(0.2 * (qt + st_1 - tocs), uu)
        # constrain by max safe release
        rt = min(rt, utils.m3s_to_m3(self.compute_max_release(st_1), freq=freq, month=month, year=year))
        # constrain by min release
        rt = max(rt, utils.m3s_to_m3(self.compute_min_release(st_1), freq=freq, month=month, year=year))
        # compute ending storage
        st = st_1 + qt - rt

        return [rt, st]

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
        
        return np.interp(current_elevation, sp_elev, rp_release)

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
        
        return np.interp(current_elevation, sp_elev, rp_release)

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
        Converts the storage in TAF to height in feet based on storage-elevation relationship.
        Parameters:
            S (float): The storage in TAF.
        Returns:
            float: The height in feet.
        """
        sp = self.characteristics["sp_to_ep"][0]
        ep = self.characteristics["sp_to_ep"][1]
        return np.interp(S, sp, ep)

    def compute_average_cumulative_inflow_by_month(self):
        """
        Returns the average cumulative inflow by beinning of month over the water year.
        Parameters:
            inflow (pd.DataFrame): DataFrame containing inflow data with columns 'water_year', 'month', and 'inflow'.
            start_wy (int): Start water year for averaging.
            end_wy (int): End water year for averaging.
        Returns:
            np.ndarray: Array of average cumulative inflow by month of the water year.
        """
        # get monthly sums
        monthly_inflow = (
            self.inflows[["water_year", "month", "inflow"]]
            .groupby(["water_year", "month"], as_index=False)
            .sum()
        )

        # accumulate average monthly inflows
        cumulative_inflow_by_month = np.zeros(12)
        for i in range(11):
            cumulative_inflow_for_month = (
                monthly_inflow.loc[
                    (monthly_inflow["month"] == i + 1),
                    "inflow",
                ]
                .mean()
                .astype(int)
            )
            cumulative_inflow_by_month[i + 1] = (
                cumulative_inflow_by_month[i] + cumulative_inflow_for_month
            )

        return cumulative_inflow_by_month

    def compute_average_remaining_demand_by_month(self):
        """
        Returns the average remaining demand by month of the water year.
        Parameters:
            demand (np.ndarray): Array of demand data.
        Returns:
            np.ndarray: Array of average remaining demand by month of the water year.
        """
        # get annual total
        total_demand = self.demand.sum()

        remaining_demand_by_month = np.zeros(12)
        remaining_demand_by_month[0] = int(total_demand)
        
        # Logic branch based on resolution
        if self.steps_per_year == 12:
            # MONTHLY LOGIC
            # Assuming self.demand is a 1D array of monthly values repeated over years
            # We need to compute the *average* demand for month i across all years
            
            # Reshape demand into (Years, 12) to get monthly averages
            # Note: This assumes self.demand length is a multiple of 12
            num_years = len(self.demand) // 12
            reshaped_demand = self.demand[:num_years*12].reshape((num_years, 12))
            avg_monthly_demand = reshaped_demand.mean(axis=0)

            # Note: The original function accumulates logic differently (subtracting averages)
            # We mimic the original logic:
            current_rem = total_demand
            
            # We iterate 0..10. For month i (0=Jan/Oct depending on start), we subtract that month's avg demand
            # However, typical usage is: remaining at START of month.
            
            # Since self.demand is usually just one single year repeated or a timeseries,
            # let's assume it follows the structure of the input file.
            
            # IF self.demand is just ONE year of data (12 values):
            if len(self.demand) == 12:
                 avg_monthly_demand = self.demand
            else:
                 # Calculate average for each month index 0..11
                 avg_monthly_demand = [np.mean(self.demand[i::12]) for i in range(12)]

            running_demand = total_demand
            # Loop to compute remaining demand at the START of next month
            for i in range(11):
                # Subtract the average demand of the current month (i)
                # Note: The original code used 30*i : 30*i+30. 
                # This implies i=0 is the first month in the array.
                
                # Check mapping: In water year, i=0 is usually Oct.
                # Assuming data is ordered by Water Year or Calendar Year consistently.
                
                running_demand -= avg_monthly_demand[i]
                remaining_demand_by_month[i + 1] = int(running_demand)
                
        else:
            # DAILY LOGIC (Original)
            # Original code hardcoded 30 days. Better to be flexible, but keeping logic for 365.
            for i in range(11):
                # Note: This approximation (30 * i) drifts from reality (365 days).
                # But keeping original logic for consistency if in daily mode.
                total_demand -= self.demand[30 * i : 30 * i + 30].sum()
                remaining_demand_by_month[i + 1] = int(total_demand)

        return remaining_demand_by_month
