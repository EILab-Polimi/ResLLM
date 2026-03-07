#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resllm.utils

Utility functions for the resllm project.

"""
import json

forecast_columns_definitions = {
    # --- GloFAS Inflow Forecasts (cubic meters) ---
    '1a_GloFAS_Seas_1m_EnsM_Average': ['The forecasted monthly average inflow across the 4 main sub-basins is', 'cubic meters.'],
    '1a_GloFAS_Seas_1m_EnsM_Kafue': ['The forecasted monthly average inflow to Itezhi-tezhi Dam/Kafue is', 'cubic meters.'],
    '1a_GloFAS_Seas_1m_EnsM_Luangwa': ['The forecasted monthly average inflow from Luangwa River is', 'cubic meters.'],
    '1a_GloFAS_Seas_1m_EnsM_Mangochi': ['The forecasted monthly average inflow to Mangochi/Shire is', 'cubic meters.'],
    '1a_GloFAS_Seas_1m_EnsM_Sum': ['The forecasted sum of monthly average inflows for all sub-basins is', 'cubic meters.'],
    '1a_GloFAS_Seas_1m_EnsM_Victoria': ['The forecasted monthly average inflow to Kariba/Victoria Falls is', 'cubic meters.'],
    '1a_GloFAS_Seas_1m_EnsM_ZRB': ['The forecasted total monthly average inflow for the Zambezi Basin is', 'cubic meters.'],

    # --- HYPE Inflow Forecasts (cubic meters) ---
    '1b_HYPE_Seas_1m_EnsM_Kafue': ['The forecasted monthly average inflow to Itezhi-tezhi Dam/Kafue is', 'cubic meters.'],
    '1b_HYPE_Seas_1m_EnsM_Luangwa': ['The forecasted monthly average inflow from Luangwa River is', 'cubic meters.'],
    '1b_HYPE_Seas_1m_EnsM_Mangochi': ['The forecasted monthly average inflow to Mangochi/Shire is', 'cubic meters.'],
    '1b_HYPE_Seas_1m_EnsM_Victoria': ['The forecasted monthly average inflow to Kariba/Victoria Falls is', 'cubic meters.'],
    '1b_HYPE_Seas_1m_EnsM_ZRB': ['The forecasted total monthly average inflow for the Zambezi Basin is', 'cubic meters.'],

    # --- HYPE Flow Anomalies (Standard Deviations) ---
    '2b_HYPE_Seas_1m_Anom_Kafue': ['The forecasted flow anomaly for Kafue River is', 'std deviation from the mean.'],
    '2b_HYPE_Seas_1m_Anom_Luangwa': ['The forecasted flow anomaly for Luangwa River is', 'std deviation from the mean.'],
    '2b_HYPE_Seas_1m_Anom_Mangochi': ['The forecasted flow anomaly for Shire River is', 'std deviation from the mean.'],
    '2b_HYPE_Seas_1m_Anom_Victoria': ['The forecasted flow anomaly for Victoria Falls is', 'std deviation from the mean.'],

    # --- GloFAS Drought Probabilities (0-1) ---
    '3a_GloFAS_Seas_3m_prob20_Kafue': ['The probability of low flows (lowest 20%) over next 3 months for Kafue is', 'probability (0-1).'],
    '3a_GloFAS_Seas_3m_prob20_Luangwa': ['The probability of low flows (lowest 20%) over next 3 months for Luangwa is', 'probability (0-1).'],
    '3a_GloFAS_Seas_3m_prob20_Mangochi': ['The probability of low flows (lowest 20%) over next 3 months for Shire is', 'probability (0-1).'],
    '3a_GloFAS_Seas_3m_prob20_Victoria': ['The probability of low flows (lowest 20%) over next 3 months for Victoria Falls is', 'probability (0-1).'],
    '3a_GloFAS_Seas_3m_prob20_ZRB': ['The probability of low flows (lowest 20%) over next 3 months for Zambezi Basin is', 'probability (0-1).'],

    # --- HYPE Drought Probabilities (0-1) ---
    '3b_HYPE_Seas_3m_prob20_Kafue': ['The probability of low flows (lowest 20%) over next 3 months for Kafue is', 'probability (0-1).'],
    '3b_HYPE_Seas_3m_prob20_Luangwa': ['The probability of low flows (lowest 20%) over next 3 months for Luangwa is', 'probability (0-1).'],
    '3b_HYPE_Seas_3m_prob20_Mangochi': ['The probability of low flows (lowest 20%) over next 3 months for Shire is', 'probability (0-1).'],
    '3b_HYPE_Seas_3m_prob20_Victoria': ['The probability of low flows (lowest 20%) over next 3 months for Victoria Falls is', 'probability (0-1).'],
    '3b_HYPE_Seas_3m_prob20_ZRB': ['The probability of low flows (lowest 20%) over next 3 months for Zambezi Basin is', 'probability (0-1).'],

    # --- GloFAS Standardized Runoff Index (Index Value) ---
    '4a_GloFAS_SRI_1m_Kafue': ['The Standardized Runoff Index (SRI) drought value for Kafue is', '.'],
    '4a_GloFAS_SRI_1m_Luangwa': ['The Standardized Runoff Index (SRI) drought value for Luangwa is', '.'],
    '4a_GloFAS_SRI_1m_Mangochi': ['The Standardized Runoff Index (SRI) drought value for Shire is', '.'],
    '4a_GloFAS_SRI_1m_Victoria': ['The Standardized Runoff Index (SRI) drought value for Victoria Falls is', '.'],
    '4a_GloFAS_SRI_1m_ZRB': ['The Standardized Runoff Index (SRI) drought value for Zambezi Basin is', '.'],

    # --- HYPE Standardized Runoff Index (Index Value) ---
    '4b_HYPE_SRI_1m_Kafue': ['The Standardized Runoff Index (SRI) drought value for Kafue is', '.'],
    '4b_HYPE_SRI_1m_Luangwa': ['The Standardized Runoff Index (SRI) drought value for Luangwa is', '.'],
    '4b_HYPE_SRI_1m_Mangochi': ['The Standardized Runoff Index (SRI) drought value for Shire is', '.'],
    '4b_HYPE_SRI_1m_Victoria': ['The Standardized Runoff Index (SRI) drought value for Victoria Falls is', '.'],
    '4b_HYPE_SRI_1m_ZRB': ['The Standardized Runoff Index (SRI) drought value for Zambezi Basin is', '.'],
}

concept_map = {
    "environment_setting": "<rank 0-4>",
    "goal": "<rank 0-4>",
    "operational_limits": "<rank 0-4>",
    "average_cumulative_inflow_by_month": "<rank 0-4>",
    "average_remaining_demand_by_month": "<rank 0-4>",
    "previous_allocation": "<rank 0-4>",
    "current_month": "<rank 0-4>",
    "current_storage": "<rank 0-4>",
    "current_cumulative_observed_inflow": "<rank 0-4>",
    "current_water_year_remaining_demand": "<rank 0-4>",
    "next_water_year_demand": "<rank 0-4>",
    "puppies": "<rank 0-4>"
}

import numpy as np
import calendar

def twh_to_m3(demand_twh, turbines, freq='M', month=None, year=None):
    """Convert demand from megawatt-hours (MWh) to cubic meters (m³).

    Parameters:
        demand_twh (float): Energy demand in terawatt-hours.
        turbines (dict): Dictionary containing turbine characteristics. 
                         Expected keys per turbine: 'max_discharge' (m³/s), 
                         'efficiency' (float 0-1), 'head' (m), 'quantity' (int).
        freq (str): Time frequency ('D' for Day, 'M' for Month).

    Returns:
        float: Volume of water in cubic meters required to meet the demand.
    """
    demand_mwh = demand_twh * 1e6  # Convert TWh to MWh

    if freq == 'D':
        hours_per_period = 24
    elif freq == 'M':
        if month is not None and year is not None:
            days_in_month = calendar.monthrange(year, month)[1]
            hours_per_period = 24 * days_in_month
        else:
            hours_per_period = 24 * 30  # Assuming a standard 30-day month
    else:
        raise ValueError("Frequency must be 'D' or 'M'")

    total_max_power_mw = 0
    total_max_flow_m3s = 0

    # Calculate system-wide power and flow to find average yield
    for name, turbine in turbines.items():
        # Calculate total maximum flow for this set of turbines (Q)
        Q_total = turbine['quantity'] * turbine['capacity'] # m³/s
        
        # Calculate Power in MW: P = (eta * rho * g * h * Q) / 10^6
        P_mw = (turbine['efficiency'] * 1000 * 9.81 * turbine['head'] * Q_total) / 10**6
        
        total_max_power_mw += P_mw
        total_max_flow_m3s += Q_total 

    # Calculate the system's average energy yield (MWh per m³ of water)
    # Power (MW) / Flow (m³/s) = MW per m³/s. Divide by 3600 to get MWh per m³.
    avg_mwh_per_m3 = (total_max_power_mw / total_max_flow_m3s) / 3600

    # 4. Convert the energy demand directly to water volume
    required_volume_m3 = demand_mwh / avg_mwh_per_m3

    # 5. Feasibility Check: Can the dam actually produce this much in the given time?
    max_possible_mwh = total_max_power_mw * hours_per_period
    
    if demand_mwh > max_possible_mwh:
        print(f"WARNING: Demand ({demand_mwh:.2f} MWh) exceeds the maximum possible "
              f"generation for this period ({max_possible_mwh:.2f} MWh).")

    return required_volume_m3


def m3s_to_m3(m3s, freq='D', month=None, year=None):
    """Convert cubic meters per second to cubic meters per specified frequency.

    Parameters:
        m3s (float): Flow rate in cubic meters per second.
        freq (str): Frequency for conversion. Options are 'D' (daily), 'H' (hourly), 'M' (monthly), 'Y' (yearly).

    Returns:
        float: Flow volume in cubic meters for the specified frequency.
    """
    seconds_per_unit = {
        'H': 3600,           # seconds in an hour
        'D': 86400,          # seconds in a day
        'M': 2592000,        # average seconds in a month (30 days)
        'Y': 31536000        # seconds in a year
    }

    if month is not None and freq == 'M':
        # Adjust for actual number of days in the month
        if year is None:
            year = 2015  # Default to a non-leap year for February
        days_in_month = [31, 29 if calendar.isleap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        seconds_per_unit['M'] = days_in_month[month - 1] * seconds_per_unit['D']

    if freq not in seconds_per_unit:
        raise ValueError("Invalid frequency. Choose from 'D', 'H', 'M', or 'Y'.")

    return m3s * seconds_per_unit[freq]

def m3_to_m3s(m3, freq='D'):
    """Convert cubic meters to cubic meters per second based on specified frequency.

    Parameters:
        m3 (float): Flow volume in cubic meters.
        freq (str): Frequency for conversion. Options are 'D' (daily), 'H' (hourly), 'M' (monthly), 'Y' (yearly).

    Returns:
        float: Flow rate in cubic meters per second.
    """
    seconds_per_unit = {
        'D': 86400,          # seconds in a day
        'H': 3600,           # seconds in an hour
        'M': 2592000,        # average seconds in a month (30 days)
        'Y': 31536000        # seconds in a year
    }

    if freq not in seconds_per_unit:
        raise ValueError("Invalid frequency. Choose from 'D', 'H', 'M', or 'Y'.")

    return m3 / seconds_per_unit[freq]


def cfs_to_taf(cfs):
    return cfs * 2.29568411 * 10**-5 * 86400 / 1000


def taf_to_cfs(taf):
    return taf * 1000 / 86400 * 43560


def select_closest_choice(numbers):
    """
    Given a list of numbers, select the closest choice from a predefined set
    (10, 20, 30, 40, 50, 60, 70, 80, 90, 100) based on a weighted scoring system.
    The scoring system assigns weights to distances from the choices:
    - 5: 10 points
    - 10: 5 points
    - 30: 3 points
    - 60: 2 points
    - 90: 1 point
    The choice with the highest score is returned.
    Parameters:
        numbers (list): A list of numbers to compare against the choices.
    Returns:
        int: The choice with the highest score.
    Example:
    >>> select_closest_choice([15, 25, 35, 40])
        30
    >>> select_closest_choice([5, 15, 25, 30])
        20
    """
    choices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    weights = {5: 10, 10: 5, 30: 3, 60: 2, 90: 1}

    def calculate_weighted_score(choice):
        score = 0
        for num in numbers:
            distance = abs(num - choice)
            # print(f"Choice: {choice}, Number: {num}, Distance: {distance}")
            for threshold, weight in weights.items():
                if distance <= threshold:
                    # print(f"Threshold: {threshold}, Weight: {weight}")
                    score += weight
                    break
        return score

    # Calculate scores for each choice and select the one with the highest score
    best_choice = max(choices, key=calculate_weighted_score)
    return best_choice


def water_day(d):
    """
    Converts a day of the year to a water day.
    Parameters:
        d (int): Day of the year (1-365/366).
    Returns:
        int: Water day (1-365/366).
    """
    return d - 274 if d >= 274 else d + 91
