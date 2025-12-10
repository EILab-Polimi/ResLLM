#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process batch results from OpenAI API and create analysis-ready output.
"""

import os
import json
import pandas as pd
import numpy as np
import argparse


def process_batch_results(batch_results_file, metadata_file, output_file=None):
    """
    Process batch results and combine with metadata.
    
    Args:
        batch_results_file: Path to the JSONL file with batch results
        metadata_file: Path to the CSV file with request metadata
        output_file: Path to save the processed results (optional)
    
    Returns:
        DataFrame with processed results
    """
    
    # Load metadata
    print(f"Loading metadata from: {metadata_file}")
    metadata = pd.read_csv(metadata_file)
    print(f"  Found {len(metadata)} requests in metadata")
    
    # Load batch results
    print(f"\nLoading batch results from: {batch_results_file}")
    results = []
    with open(batch_results_file, "r") as f:
        for line in f:
            results.append(json.loads(line))
    print(f"  Found {len(results)} results")
    
    # Process results
    processed_results = []
    
    # Detect metadata format (ablation vs original)
    is_ablation = "ablation_type" in metadata.columns
    
    for result in results:
        custom_id = result["custom_id"]
        
        # Get metadata for this request
        meta_row = metadata[metadata["custom_id"] == custom_id]
        if len(meta_row) == 0:
            print(f"Warning: No metadata found for {custom_id}")
            continue
        
        meta_row = meta_row.iloc[0]
        
        # Get sample_number - handle both formats
        if is_ablation:
            sample_number = meta_row.get("sample_num", 0)
            month_of_water_year = meta_row.get("month", None)
            # Ablation metadata doesn't have these detailed fields
            storage_taf = None
            cumulative_inflow_taf = None
            remaining_demand_taf = None
            forecast_mean_taf = None
            forecast_10th_taf = None
            forecast_90th_taf = None
            ablation_type = meta_row.get("ablation_type", None)
            original_allocation = meta_row.get("allocation_percent", None)
        else:
            sample_number = meta_row.get("sample_number", 0)
            month_of_water_year = meta_row.get("month_of_water_year", None)
            storage_taf = meta_row.get("storage_taf", None)
            cumulative_inflow_taf = meta_row.get("cumulative_inflow_taf", None)
            remaining_demand_taf = meta_row.get("remaining_demand_taf", None)
            forecast_mean_taf = meta_row.get("forecast_mean_taf", None)
            forecast_10th_taf = meta_row.get("forecast_10th_taf", None)
            forecast_90th_taf = meta_row.get("forecast_90th_taf", None)
            ablation_type = None
            original_allocation = None
        
        # Extract response
        if result.get("error"):
            # Error in the request
            result_data = {
                "custom_id": custom_id,
                "date": meta_row["date"],
                "water_year": meta_row["water_year"],
                "month_of_water_year": month_of_water_year,
                "sample_number": sample_number,
                "error": str(result["error"]),
                "allocation_percent": None,
                "allocation_reasoning": None
            }
            if is_ablation:
                result_data["ablation_type"] = ablation_type
                result_data["original_allocation_percent"] = original_allocation
            else:
                result_data["storage_taf"] = storage_taf
                result_data["cumulative_inflow_taf"] = cumulative_inflow_taf
                result_data["remaining_demand_taf"] = remaining_demand_taf
                result_data["forecast_mean_taf"] = forecast_mean_taf
                result_data["forecast_10th_taf"] = forecast_10th_taf
                result_data["forecast_90th_taf"] = forecast_90th_taf
            processed_results.append(result_data)
        elif result["response"]["status_code"] == 200:
            # Successful response
            response_body = result["response"]["body"]
            
            # Extract the content or refusal
            message = response_body["choices"][0]["message"]
            content = message.get("content")
            refusal = message.get("refusal")
            
            # Handle refusal case
            if content is None and refusal is not None:
                # Model refused to respond
                result_row = {
                    "custom_id": custom_id,
                    "date": meta_row["date"],
                    "water_year": meta_row["water_year"],
                    "month_of_water_year": month_of_water_year,
                    "sample_number": sample_number,
                    "allocation_percent": np.nan,
                    "allocation_reasoning": refusal,
                    "error": "refusal"
                }
                
                if is_ablation:
                    result_row["ablation_type"] = ablation_type
                    result_row["original_allocation_percent"] = original_allocation
                else:
                    result_row["storage_taf"] = storage_taf
                    result_row["cumulative_inflow_taf"] = cumulative_inflow_taf
                    result_row["remaining_demand_taf"] = remaining_demand_taf
                    result_row["forecast_mean_taf"] = forecast_mean_taf
                    result_row["forecast_10th_taf"] = forecast_10th_taf
                    result_row["forecast_90th_taf"] = forecast_90th_taf
                
                processed_results.append(result_row)
                continue
            
            # Parse JSON content
            try:
                decision = json.loads(content)
                
                # Handle both normal schema and bare_minimal schema
                if "allocation_percent" in decision:
                    # Normal schema
                    allocation_percent = decision.get("allocation_percent")
                    allocation_reasoning = decision.get("allocation_reasoning", "")
                elif "percent" in decision:
                    # Bare minimal schema
                    allocation_percent = decision.get("percent")
                    allocation_reasoning = decision.get("reasoning", "")
                else:
                    # Fallback
                    allocation_percent = None
                    allocation_reasoning = str(decision)
                
                concept_importance = decision.get("allocation_concept_importance", {})
                
                # Build result row
                result_row = {
                    "custom_id": custom_id,
                    "date": meta_row["date"],
                    "water_year": meta_row["water_year"],
                    "month_of_water_year": month_of_water_year,
                    "sample_number": sample_number,
                    "allocation_percent": allocation_percent,
                    "allocation_reasoning": allocation_reasoning,
                    "error": None
                }
                
                if is_ablation:
                    result_row["ablation_type"] = ablation_type
                    result_row["original_allocation_percent"] = original_allocation
                else:
                    result_row["storage_taf"] = storage_taf
                    result_row["cumulative_inflow_taf"] = cumulative_inflow_taf
                    result_row["remaining_demand_taf"] = remaining_demand_taf
                    result_row["forecast_mean_taf"] = forecast_mean_taf
                    result_row["forecast_10th_taf"] = forecast_10th_taf
                    result_row["forecast_90th_taf"] = forecast_90th_taf
                
                # Add concept importance scores
                for concept, importance in concept_importance.items():
                    result_row[f"importance_{concept}"] = importance
                
                processed_results.append(result_row)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON for {custom_id}: {e}")
                result_data = {
                    "custom_id": custom_id,
                    "date": meta_row["date"],
                    "water_year": meta_row["water_year"],
                    "month_of_water_year": month_of_water_year,
                    "sample_number": sample_number,
                    "error": f"JSON parse error: {e}",
                    "allocation_percent": None,
                    "allocation_reasoning": content
                }
                if is_ablation:
                    result_data["ablation_type"] = ablation_type
                    result_data["original_allocation_percent"] = original_allocation
                else:
                    result_data["storage_taf"] = storage_taf
                    result_data["cumulative_inflow_taf"] = cumulative_inflow_taf
                    result_data["remaining_demand_taf"] = remaining_demand_taf
                    result_data["forecast_mean_taf"] = forecast_mean_taf
                    result_data["forecast_10th_taf"] = forecast_10th_taf
                    result_data["forecast_90th_taf"] = forecast_90th_taf
                processed_results.append(result_data)
        else:
            # Non-200 status code
            processed_results.append({
                "custom_id": custom_id,
                "date": meta_row["date"],
                "water_year": meta_row["water_year"],
                "month_of_water_year": meta_row["month_of_water_year"],
                "sample_number": sample_number,
                "storage_taf": meta_row["storage_taf"],
                "cumulative_inflow_taf": meta_row["cumulative_inflow_taf"],
                "remaining_demand_taf": meta_row["remaining_demand_taf"],
                "forecast_mean_taf": meta_row["forecast_mean_taf"],
                "forecast_10th_taf": meta_row["forecast_10th_taf"],
                "forecast_90th_taf": meta_row["forecast_90th_taf"],
                "error": f"Status code: {result['response']['status_code']}",
                "allocation_percent": None,
                "allocation_reasoning": None
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(processed_results)
    
    # Sort by date
    results_df = results_df.sort_values("date")
    
    print(f"\nProcessed {len(results_df)} results")
    print(f"  Successful: {results_df['error'].isna().sum()}")
    print(f"  Errors: {results_df['error'].notna().sum()}")
    
    if results_df["error"].notna().sum() > 0:
        print("\nErrors:")
        for idx, row in results_df[results_df["error"].notna()].iterrows():
            print(f"  {row['custom_id']} ({row['date']}): {row['error']}")
    
    # Save results
    if output_file:
        results_df.to_csv(output_file, index=False, quotechar='"')
        print(f"\nResults saved to: {output_file}")
    
    # Print summary statistics
    if results_df["allocation_percent"].notna().sum() > 0:
        print("\nAllocation decision summary:")
        print(results_df["allocation_percent"].describe())
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Process OpenAI batch results")
    parser.add_argument(
        "--results",
        required=True,
        help="Path to batch results JSONL file"
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata CSV file"
    )
    parser.add_argument(
        "--output",
        help="Path to save processed results CSV"
    )
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output:
        results_dir = os.path.dirname(args.results)
        results_basename = os.path.basename(args.results).replace(".jsonl", "")
        args.output = os.path.join(results_dir, f"{results_basename}_processed.csv")
    
    process_batch_results(args.results, args.metadata, args.output)


if __name__ == "__main__":
    # Example usage:
    # python process_batch_results.py \
    #   --results ./output/batch_results_batch_abc123.jsonl \
    #   --metadata ./output/batch_requests_metadata_1996_2016.csv \
    #   --output ./output/allocation_decisions_1996_2016.csv
    
    main()
