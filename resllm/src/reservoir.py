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
                - inflow_file (str): Path to the inflow data file.
                - demand_file (str): Path to the demand data file.
                - operable_storage_max (float): Maximum operable storage in TAF.
                - operable_storage_min (float): Minimum operable storage in TAF.
                - max_safe_release (float):  Maximum safe release in TAF.
                - sp_to_rp (list): List of storage to release points.
                - sp_to_ep (list): List of storage to elevation points.
                - tp_to_tocs (list): List of top of conservation storage points.
        """

        self.tocs = characteristics["tocs"]
        self.demand = np.loadtxt(characteristics["demand_file"]) # demand, TAF
        print(f"Demand data loaded: {characteristics['demand_file']}")
        self.inflows = pd.read_csv(characteristics["inflow_file"])  # inflow, TAF
        if characteristics["wy_forecast_file"] is not False:
            self.forecasted_inflows = pd.read_csv(characteristics["wy_forecast_file"])  # forecasted inflows, TAF
            self.forecasted_inflows["date"] = pd.to_datetime(self.forecasted_inflows["date"])
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
        self.inflows["week"] = self.inflows["dowy"].apply(
            lambda x: int((x - 1) / 7) + 1
        )
        self.inflows.loc[
            (self.inflows["month"] == 10) & (self.inflows["day"] == 1), "week"
        ] = 1
        self.inflows["water_year"] = np.where(
            self.inflows["month"] >= 10,
            self.inflows["year"] + 1,
            self.inflows["year"],
        )
        self.inflows["date"] = self.inflows["date"].dt.strftime("%Y-%m-%d")
        print(f"Inflow data loaded: {characteristics['inflow_file']}")

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
        uu: float = None
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

    def evaluate(self, st_1, qt, uu, tocs):
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
        rt = min(rt, utils.cfs_to_taf(self.compute_max_release(st_1)))
        # constrain by min release
        rt = min(rt, st_1 + qt)
        # add any spill
        rt += max(st_1 + qt - rt - K, 0)
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
        sp = self.characteristics["sp_to_rp"][0]
        rp = self.characteristics["sp_to_rp"][1]
        return np.interp(S, sp, rp)

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

        # subtract monthly demand from total demand
        remaining_demand_by_month = np.zeros(12)
        remaining_demand_by_month[0] = int(total_demand)
        for i in range(11):
            total_demand -= self.demand[30 * i : 30 * i + 30].sum()
            remaining_demand_by_month[i + 1] = int(total_demand)

        return remaining_demand_by_month
