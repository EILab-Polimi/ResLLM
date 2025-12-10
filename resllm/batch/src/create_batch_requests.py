#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create batch requests for OpenAI API to generate allocation decisions
for the historical period (1996-2016).
"""

import os
import json
import yaml
import pandas as pd
import numpy as np
import argparse
from textwrap import dedent
from datetime import datetime

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create batch requests for OpenAI API")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples to generate for each date (default: 1)"
    )
    args = parser.parse_args()
    
    n_samples = args.n_samples
    print(f"Generating {n_samples} sample(s) per date...")
    
    # Setup paths
    file_dir = os.path.dirname(os.path.abspath(__file__))  # resllm/batch/src
    batch_dir = os.path.join(file_dir, "..")  # resllm/batch
    resllm_dir = os.path.join(batch_dir, "..")  # resllm
    repo_root = os.path.join(resllm_dir, "..")  # ResLLM
    data_dir = os.path.join(repo_root, "data")  # ResLLM/data
    
    # Load configuration
    config_path = os.path.join(resllm_dir, "configs", "folsom_hist_forecast.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load data files
    demand = np.loadtxt(os.path.join(data_dir, config["folsom_reservoir"]["demand_file"]))
    inflows = pd.read_csv(os.path.join(data_dir, config["folsom_reservoir"]["inflow_file"]))
    forecasts = pd.read_csv(os.path.join(data_dir, config["folsom_reservoir"]["wy_forecast_file"]))
    
    # Process inflow data
    inflows["date"] = pd.to_datetime(inflows["date"])
    inflows["year"] = inflows["date"].dt.year
    inflows["month"] = inflows["date"].dt.month
    inflows["day"] = inflows["date"].dt.day
    inflows["water_year"] = np.where(
        inflows["month"] >= 10,
        inflows["year"] + 1,
        inflows["year"],
    )
    
    # Process forecast data
    forecasts["date"] = pd.to_datetime(forecasts["date"])
    
    # Compute average cumulative inflow by month
    monthly_inflow = (
        inflows[["water_year", "month", "inflow"]]
        .groupby(["water_year", "month"], as_index=False)
        .sum()
    )
    
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
    
    # Compute average remaining demand by month
    total_demand = demand.sum()
    remaining_demand_by_month = np.zeros(12)
    remaining_demand_by_month[0] = int(total_demand)
    for i in range(11):
        total_demand -= demand[30 * i : 30 * i + 30].sum()
        remaining_demand_by_month[i + 1] = int(total_demand)
    
    # System message
    system_message = dedent(
        """\
        You are a water reservoir operator.
        Your goal is to minimize shortages to downstream water supply by releasing water from the reservoir.
        The reservoir is located in a region with a Mediterranean climate, characterized by hot, dry summers and highly variable wet winters.
        The reservoir is operated to meet the municipal and agricultural water supply needs of the region while also maintaining flood control and environmental flow requirements.
        The water year is defined as the period from October through September.
        """
    )
    
    # Instructions
    instructions = dedent(
        """\
        - You are tasked with determining the percent allocation of water demand to release from the reservoir.
        - At the beginning of each month, you will be asked to update the percent allocation decision based on your current observations.
        - In your determination, consider the volume currently in storage, inflow to date compared to expected inflows, and the need to balance meeting current demands against conserving water for future demands.
        - Note that a shortage is calculated by demand x (100 - percent allocation).
        You have the following information about the reservoir:
        - The maximum operable storage level is {} TAF.
        - The minimum operable storage level is {} TAF.
        """
    ).format(
        config["folsom_reservoir"]["operable_storage_max"],
        config["folsom_reservoir"]["operable_storage_min"],
    )
    
    avg_demand = int(demand.sum())
    instructions += f"- The average total water year demand: {avg_demand}\n"
    
    instructions += """- The average cumulative inflow by beginning of month of the water year: """
    for month in range(12):
        instructions += f"month {month + 1}: {int(cumulative_inflow_by_month[month])} TAF; "
    instructions += "\n"
    
    instructions += """- The average remaining demand by beginning of month of the water year: """
    for month in range(12):
        instructions += f"month {month + 1}: {int(remaining_demand_by_month[month])} TAF; "
    instructions += "\n"
    
    instructions += dedent(
        """\
        - Starting in month 4 of the water year, you have access to a probabilistic forecast of inflows for the remainder of the water year.
        - The probabilistic forecast includes the ensemble mean, and 10th and 90th percentile expected water year inflow.
        - Use this forecast to inform your allocation decision.
        """
    )
    
    # Response schema
    response_schema = {
        "type": "object",
        "properties": {
            "allocation_reasoning": {
                "type": "string",
                "description": "A brief justification of the percent allocation decision."
            },
            "allocation_percent": {
                "type": "number",
                "description": "The percent allocation to release from the reservoir."
            },
        },
        "required": ["allocation_reasoning", "allocation_percent"],
        "additionalProperties": False
    }
    
    # Generate batch requests
    batch_requests = []
    request_id = 0

    # Filter to first day of each month from 1995-10-01 to 2016-09-01
    start_date = pd.Timestamp("1995-10-01")
    end_date = pd.Timestamp("2016-09-01")
    
    # Get all first days of month in the period
    first_days = inflows[
        (inflows["date"] >= start_date) & 
        (inflows["date"] <= end_date) &
        (inflows["day"] == 1)
    ].copy()
    
    # Sort by date
    first_days = first_days.sort_values("date")
    
    # Calculate historical allocation time series (outflow / demand)
    # This matches the logic in mlp_allocation.py
    
    # Calculate water year day (1-366) for each date
    def get_water_year_day(date):
        water_year = date.year if date.month < 10 else date.year + 1
        oct_1 = pd.Timestamp(f'{water_year - 1}-10-01')
        return (date - oct_1).days + 1
    
    inflows['water_year_day'] = inflows['date'].apply(get_water_year_day)
    
    # Map demand to each day using water year day
    # Handle leap year day 366 by using day 365 value if needed
    inflows['demand'] = inflows['water_year_day'].apply(lambda wd: demand[min(wd - 1, len(demand) - 1)])
    
    # Calculate 30-day backward-looking moving averages of outflow and demand
    inflows['outflow_ma30'] = inflows['outflow'].clip(upper=10).rolling(window=25, min_periods=1).mean().shift(-13)
    inflows['allocation'] = (inflows['outflow_ma30'] / inflows['demand']).clip(upper=1.0)

    # Calculate allocation as 30-day average outflow / 30-day average demand
    # This represents the average allocation over the past month
    inflows['allocation_ma30'] = inflows['allocation'].rolling(window=30, min_periods=1).mean()
    
    
    
    for idx, row in first_days.iterrows():
        date = row["date"]
        wy = row["water_year"]
        month = row["month"]
        
        # Calculate month of water year (1-12)
        mowy = month - 9 if month > 9 else month + 3
        
        # Get storage for this date
        st_1 = row["storage"]
        
        # Calculate cumulative inflow for the water year to date
        wy_start = pd.Timestamp(f"{wy-1}-10-01")
        qwyaccum = inflows[
            (inflows["date"] >= wy_start) &
            (inflows["date"] < date)
        ]["inflow"].sum()
        
        # Calculate remaining demand for the water year
        # Days elapsed in the water year
        days_elapsed = (date - wy_start).days
        # Remaining demand
        d_wy_rem = demand[days_elapsed:].sum()
        
        # Get forecast data for this date
        forecast_row = forecasts[forecasts["date"] == date]
        if mowy > 3:
            # Mean forecast (QCYFHM_tot)
            qwy_forecast_mean = forecast_row["QCYFHM"].values[0]
            # 10th percentile forecast (QCYFH1 - driest)
            qwy_forecast_10 = forecast_row["QCYFH1"].values[0]
            # 90th percentile forecast (QCYFH9 - wettest)
            qwy_forecast_90 = forecast_row["QCYFH9"].values[0]
        else:
            # If no forecast available, use None
            qwy_forecast_mean = None
            qwy_forecast_10 = None
            qwy_forecast_90 = None

        # Previous allocation decision (from historical data)
        # For the first date, assume 100% allocation
        if request_id == 0:
            previous_allocation = 100.0
        else:
            previous_allocation = inflows.loc[inflows.index[inflows["date"] == date][0] - 1, "allocation_ma30"] * 100.0
        
        # Construct observation message
        observation = dedent(
            f"""\
            It is the beginning of month {int(mowy)} of the water year.
            """
        )
        
        if mowy > 1:
            observation += dedent(
                f"""\
                So far this water year, {int(qwyaccum)} TAF of reservoir inflow has been observed.
                """
            )
        
        observation += dedent(
            f"""\
            There is currently {int(st_1)} TAF in storage.
            """
        )
        
        if qwy_forecast_mean is not None:
            observation += dedent(
                f"""\
                The probabilistic forecasted inflows for the remainder of the water year are:
                - Mean (expected): {int(qwy_forecast_mean)} TAF
                - 10th percentile: {int(qwy_forecast_10)} TAF
                - 90th percentile: {int(qwy_forecast_90)} TAF
                """
            )
        
        observation += dedent(
            f"""\
            There is approximately {int(d_wy_rem)} TAF of water demand to meet over the remainder of the water year.
            """
        )
        
        if mowy >= 9:
            observation += dedent(
                f"""\
                Also, note that next water year is approaching and the first three months have a demand of {int(demand[:90].sum())} TAF.
                """
            )
        
        observation += dedent(
            f"""\
            The previous percent allocation decision was {int(previous_allocation)} percent.
            Provide a percent allocation decision (from 0-100 percent) which continues or updates the allocation.
            """
        )
        
        # Create n duplicate requests for this date (for sampling variability)
        for sample_num in range(n_samples):
            # Create the batch request
            batch_request = {
                "custom_id": f"request-{request_id}_n{sample_num}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "o4-mini-2025-04-16",
                    "messages": [
                        {
                            "role": "system",
                            "content": system_message + instructions
                        },
                        {
                            "role": "user",
                            "content": observation
                        }
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "allocation_decision",
                            "strict": True,
                            "schema": response_schema
                        }
                    },
                    "reasoning_effort":  "high"
                }
            }
            
            batch_requests.append(batch_request)
        
        # Store metadata for later matching
        # We'll create a separate CSV file to map request IDs to dates and other info
        
        # Update previous_allocation to the actual historical 30-day moving average allocation 
        # from the previous month
        # Find the previous month's first day
        if month == 1:
            prev_month = 12
            prev_year = row["date"].year - 1
        else:
            prev_month = month - 1
            prev_year = row["date"].year
        
        prev_month_date = pd.Timestamp(f"{prev_year}-{prev_month:02d}-15")
        
        # Get the 30-day moving average allocation for that date from the historical data
        prev_month_data = inflows[inflows["date"] == prev_month_date]
        if len(prev_month_data) > 0:
            # Convert to percentage (0-100)
            previous_allocation = prev_month_data["allocation_ma30"].values[0] * 100
        # else: keep the current value (handles edge case for first date)
        
        request_id += 1
    
    # Write batch requests to JSONL file
    output_dir = os.path.join(batch_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Include n_samples in filename if > 1
    if n_samples > 1:
        batch_file = os.path.join(output_dir, f"batch_requests_1996_2016_n{n_samples}.jsonl")
        metadata_file_name = f"batch_requests_metadata_1996_2016_n{n_samples}.csv"
    else:
        batch_file = os.path.join(output_dir, "batch_requests_1996_2016.jsonl")
        metadata_file_name = "batch_requests_metadata_1996_2016.csv"
    
    with open(batch_file, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
    
    print(f"\nCreated {len(batch_requests)} batch requests ({len(first_days)} dates × {n_samples} samples)")
    print(f"Batch file saved to: {batch_file}")
    
    # Create metadata file to map request IDs to dates
    metadata = []
    request_id = 0
    for idx, row in first_days.iterrows():
        date = row["date"]
        wy = row["water_year"]
        month = row["month"]
        mowy = month - 9 if month > 9 else month + 3
        st_1 = row["storage"]
        
        wy_start = pd.Timestamp(f"{wy-1}-10-01")
        qwyaccum = inflows[
            (inflows["date"] >= wy_start) & 
            (inflows["date"] < date)
        ]["inflow"].sum()
        
        days_elapsed = (date - wy_start).days
        d_wy_rem = demand[days_elapsed:].sum()
        
        forecast_row = forecasts[forecasts["date"] == date]
        if len(forecast_row) > 0:
            qwy_forecast_mean = forecast_row["QCYFHM"].values[0]
            qwy_forecast_10 = forecast_row["QCYFH1"].values[0]
            qwy_forecast_90 = forecast_row["QCYFH9"].values[0]
        else:
            qwy_forecast_mean = np.nan
            qwy_forecast_10 = np.nan
            qwy_forecast_90 = np.nan
        
        # Create metadata for each sample
        for sample_num in range(n_samples):
            metadata.append({
                "custom_id": f"request-{request_id}_n{sample_num}",
                "date": date.strftime("%Y-%m-%d"),
                "water_year": wy,
                "month_of_water_year": mowy,
                "sample_number": sample_num,
                "storage_taf": st_1,
                "cumulative_inflow_taf": qwyaccum,
                "remaining_demand_taf": d_wy_rem,
                "forecast_mean_taf": qwy_forecast_mean,
                "forecast_10th_taf": qwy_forecast_10,
                "forecast_90th_taf": qwy_forecast_90
            })
        
        request_id += 1
    
    metadata_df = pd.DataFrame(metadata)
    metadata_file = os.path.join(output_dir, metadata_file_name)
    metadata_df.to_csv(metadata_file, index=False)
    print(f"Metadata saved to: {metadata_file}")
    
    print("\nNext steps:")
    print("1. Upload the batch file to OpenAI:")
    print(f"   openai api files.create -f {batch_file} -p batch")
    print("2. Create a batch job with the file ID")
    print("3. Wait for the batch to complete")
    print("4. Download and process the results")


if __name__ == "__main__":
    main()
