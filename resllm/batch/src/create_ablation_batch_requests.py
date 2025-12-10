#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create ablation batch requests for OpenAI API by removing specific elements
from existing observations in decision_output files (n0-n9) to study their 
impact on allocation decisions.
"""

import os
import json
import argparse
import re
import pandas as pd
from glob import glob


def split_system_and_user(observation):
    """
    Split observation into system message and user message.
    System message ends before "It is the beginning of month".
    
    Args:
        observation: The full observation text
        
    Returns:
        Tuple of (system_message, user_message)
    """
    lines = observation.split('\n')
    system_lines = []
    user_lines = []
    in_user_section = False
    
    for line in lines:
        if line.strip().startswith("It is the beginning of month"):
            in_user_section = True
        
        if in_user_section:
            user_lines.append(line)
        else:
            system_lines.append(line)
    
    return '\n'.join(system_lines), '\n'.join(user_lines)


def remove_element_from_observation(observation, ablation_type):
    """
    Remove a specific element from an observation string, including both
    the current observation and associated system instructions.
    Also removes importance ranking and puppies lines.
    
    Args:
        observation: The full observation text
        ablation_type: Which element to remove
        
    Returns:
        Modified observation with the specified element removed
    """
    lines = observation.split('\n')
    filtered_lines = []
    skip_forecast_data = False
    skip_forecast_instruction = False
    
    for i, line in enumerate(lines):
        # Remove importance ranking line (always)
        if "Assign an importance ranking" in line:
            continue
        # Remove puppies line (always)
        if "Puppies like to play" in line:
            continue
            
        # Skip lines based on ablation type
        if ablation_type == "current_storage":
            # Remove current observation
            if "There is currently" in line and "TAF in storage" in line:
                continue
            # Skip system instruction about storage in observations
            if "consider the volume currently in storage" in line:
                # Replace with version without storage mention
                line = line.replace(", the volume currently in storage,", "")
                line = line.replace("consider the volume currently in storage, inflow", "consider inflow")
                
        elif ablation_type == "forecasts":
            # Remove current forecast observation
            if "The probabilistic forecasted inflows for the remainder of the water year are:" in line:
                skip_forecast_data = True
                continue
            if skip_forecast_data:
                if line.strip().startswith("- Mean") or line.strip().startswith("- 10th") or line.strip().startswith("- 90th"):
                    continue
                else:
                    skip_forecast_data = False
            # Remove system instructions about forecasts
            if "Starting in month 4 of the water year, you have access to a probabilistic forecast" in line:
                skip_forecast_instruction = True
                continue
            if skip_forecast_instruction:
                if "- The probabilistic forecast includes" in line or "- Use this forecast to inform" in line:
                    continue
                else:
                    skip_forecast_instruction = False
                    
        elif ablation_type == "forecast_p10":
            # Remove only the 10th percentile forecast from observations
            if "The probabilistic forecasted inflows for the remainder of the water year are:" in line:
                skip_forecast_data = True
            elif skip_forecast_data:
                if line.strip().startswith("- 10th"):
                    continue
                elif line.strip().startswith("- Mean") or line.strip().startswith("- 90th"):
                    # Keep these lines and continue skipping
                    pass
                elif line.strip() and not line.strip().startswith("-"):
                    # We've moved past the forecast section
                    skip_forecast_data = False
            # Modify system instruction to remove 10th percentile mention
            if "- The probabilistic forecast includes the ensemble mean, and 10th and 90th percentile" in line:
                line = line.replace("the ensemble mean, and 10th and 90th percentile", "the ensemble mean and 90th percentile")
                
        elif ablation_type == "forecast_mean":
            # Remove only the mean forecast from observations
            if "The probabilistic forecasted inflows for the remainder of the water year are:" in line:
                skip_forecast_data = True
            elif skip_forecast_data:
                if line.strip().startswith("- Mean"):
                    continue
                elif line.strip().startswith("- 10th") or line.strip().startswith("- 90th"):
                    # Keep these lines and continue skipping
                    pass
                elif line.strip() and not line.strip().startswith("-"):
                    # We've moved past the forecast section
                    skip_forecast_data = False
            # Modify system instruction to remove mean mention
            if "- The probabilistic forecast includes the ensemble mean, and 10th and 90th percentile" in line:
                line = line.replace("the ensemble mean, and 10th and 90th percentile", "10th and 90th percentile")
                
        elif ablation_type == "forecast_p90":
            # Remove only the 90th percentile forecast from observations
            if "The probabilistic forecasted inflows for the remainder of the water year are:" in line:
                skip_forecast_data = True
            elif skip_forecast_data:
                if line.strip().startswith("- 90th"):
                    continue
                elif line.strip().startswith("- Mean") or line.strip().startswith("- 10th"):
                    # Keep these lines and continue skipping
                    pass
                elif line.strip() and not line.strip().startswith("-"):
                    # We've moved past the forecast section
                    skip_forecast_data = False
            # Modify system instruction to remove 90th percentile mention
            if "- The probabilistic forecast includes the ensemble mean, and 10th and 90th percentile" in line:
                line = line.replace("the ensemble mean, and 10th and 90th percentile", "the ensemble mean and 10th percentile")
                    
        elif ablation_type == "previous_allocation":
            # Remove current observation
            if "The previous percent allocation decision was" in line:
                continue
        
        elif ablation_type == "demand":
            # Remove current remaining demand observation
            if "There is approximately" in line and "TAF of water demand to meet over the remainder" in line:
                continue
            # Remove next water year demand note if present
            if "Also, note that next water year is approaching" in line:
                continue
            # Remove average remaining demand by month statement
            if "The average remaining demand by beginning of month of the water year:" in line:
                continue
            # Remove average total water year demand statement
            if "The average total water year demand:" in line:
                continue
            # Skip system instruction mentioning demands
            if "balance meeting current demands against conserving water for future demands" in line:
                line = line.replace(", inflow to date compared to expected inflows, and the need to balance meeting current demands against conserving water for future demands", 
                                   " and inflow to date compared to expected inflows")
                
        elif ablation_type == "current_month":
            # Remove current month information
            if "It is the beginning of month" in line and "of the water year." in line:
                continue
                
        elif ablation_type == "cumulative_inflow":
            # Remove current observation
            if "So far this water year," in line and "TAF of reservoir inflow has been observed" in line:
                continue
            # Remove average cumulative inflow reference in system context
            if "The average cumulative inflow by beginning of month of the water year:" in line:
                continue
            # Modify system instruction to remove inflow mention
            if "inflow to date compared to expected inflows" in line:
                line = line.replace(", inflow to date compared to expected inflows,", "")
                line = line.replace("consider the volume currently in storage, inflow to date compared to expected inflows, and the need", 
                                   "consider the volume currently in storage and the need")
                                   
        elif ablation_type == "storage_and_inflow":
            # Remove current storage observation
            if "There is currently" in line and "TAF in storage" in line:
                continue
            # Remove average cumulative inflow reference in system context
            if "The average cumulative inflow by beginning of month of the water year:" in line:
                continue
            # Remove cumulative inflow observation
            if "So far this water year," in line and "TAF of reservoir inflow has been observed" in line:
                continue
            # Modify system instruction to remove both storage and inflow mentions
            if "consider the volume currently in storage, inflow to date compared to expected inflows" in line:
                line = line.replace("consider the volume currently in storage, inflow to date compared to expected inflows, and the need", 
                                   "consider the volume currently in storage and the need")
        
        elif ablation_type == "no_system":
            # For no_system ablation, we skip all lines in the system section
            # The split_system_and_user function will handle separating system from user
            # Here we just mark that we want to remove all system lines
            pass
        
        elif ablation_type == "minimal":
            # For minimal ablation, we keep only the final request line
            # Everything else will be filtered out in the message construction
            pass
        
        elif ablation_type == "bare_minimal":
            # For bare_minimal ablation, everything is removed
            # Message construction will use generic prompt and schema
            pass
        
        elif ablation_type == "default":
            # Default ablation leaves the observation unchanged
            pass
        
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create ablation batch requests by removing specific observation elements from decision_output files"
    )
    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="Month of the water year to generate requests for (1-12)"
    )
    parser.add_argument(
        "--ablation-type",
        type=str,
        required=True,
        choices=[
            "current_storage",
            "forecasts",
            "forecast_p10",
            "forecast_mean",
            "forecast_p90",
            "previous_allocation",
            "demand",
            "current_month",
            "cumulative_inflow",
            "storage_and_inflow",
            "no_system",
            "minimal",
            "bare_minimal",
            "default"
        ],
        help="Which observation element to remove"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing decision_output files (default: analysis/output/resllm/folsom_hist_forecast_1996_2016/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save batch requests (default: resllm/batch/output/)"
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="o4-mini-2025-04-16",
        help="Model prefix for decision_output files to use (default: o4-mini-2025-04-16)"
    )
    args = parser.parse_args()
    
    month = args.month
    ablation_type = args.ablation_type
    model_prefix = args.model_prefix
    
    if month < 1 or month > 12:
        raise ValueError("Month must be between 1 and 12")
    
    print(f"Generating ablation requests for month {month}")
    print(f"Ablation type: {ablation_type}")
    
    # Setup paths
    file_dir = os.path.dirname(os.path.abspath(__file__))  # resllm/batch/src
    batch_dir = os.path.join(file_dir, "..")  # resllm/batch
    repo_root = os.path.join(batch_dir, "..", "..")  # ResLLM
    
    # Set default input directory if not provided
    if args.input_dir is None:
        input_dir = os.path.join(repo_root, "analysis", "output", "resllm", "folsom_hist_forecast_1996_2016")
    else:
        input_dir = args.input_dir
    
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Set default output directory if not provided
    if args.output_dir is None:
        output_dir = os.path.join(batch_dir, "output")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading decision outputs from: {input_dir}")
    
    # Find all decision_output files (n0-n9) for the specified model
    decision_files = glob(os.path.join(input_dir, f"{model_prefix}_decision_output_n[0-9].csv"))
    
    if not decision_files:
        raise ValueError(f"No decision_output files found in {input_dir}")
    
    print(f"Found {len(decision_files)} decision_output files")
    
    # Load all decision_output files and filter for the specified month
    all_observations = []
    
    for file_path in sorted(decision_files):
        df = pd.read_csv(file_path)
        
        # Filter for the specified month
        df_month = df[df['mowy'] == month].copy()
        
        print(f"  {os.path.basename(file_path)}: {len(df_month)} rows for month {month}")
        
        # Extract sample number from filename (e.g., n0, n1, ..., n9)
        filename = os.path.basename(file_path)
        sample_match = re.search(r'_n(\d+)\.csv$', filename)
        sample_num = int(sample_match.group(1)) if sample_match else 0
        
        # Extract observations
        for idx, row in df_month.iterrows():
            all_observations.append({
                'date': row['date'],
                'observation': row['observation'],
                'allocation_percent': int(row['allocation_percent']),
                'source_file': filename,
                'sample_num': sample_num,
                'wy': row['wy'],
                'mowy': row['mowy']
            })
    
    print(f"\nTotal observations collected: {len(all_observations)}")
    
    # Sort observations by sample number (n0-n9) and then by date
    all_observations.sort(key=lambda x: (x['sample_num'], x['date']))
    
    # Response schema (default)
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
    
    # Generic response schema for bare_minimal (no reservoir mention)
    bare_minimal_schema = {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "A brief justification of the decision."
            },
            "percent": {
                "type": "number",
                "description": "The percent value."
            },
        },
        "required": ["reasoning", "percent"],
        "additionalProperties": False
    }
    
    # Generate batch requests
    batch_requests = []
    metadata = []
    
    for i, obs_data in enumerate(all_observations):
        date = obs_data['date']
        observation = obs_data['observation']
        sample_num = obs_data['sample_num']
        allocation_percent = obs_data['allocation_percent']
        
        # Split into system and user messages first (before ablation)
        system_message, user_message = split_system_and_user(observation)
        
        # Apply ablation to the appropriate part
        system_message = remove_element_from_observation(system_message, ablation_type)
        user_message = remove_element_from_observation(user_message, ablation_type)
        
        # Create request
        custom_id = f"ablation_{ablation_type}_month{month}_n{sample_num}_date{date.replace('-', '')}_alloc{allocation_percent}"
        
        # Build messages list based on ablation type
        if ablation_type == "no_system":
            # No system prompt, only user message
            messages = [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            schema = response_schema
        elif ablation_type == "minimal":
            # Minimal ablation: no system prompt, only the final request line
            minimal_prompt = "Provide a percent allocation decision (from 0-100 percent) which continues or updates the allocation."
            messages = [
                {
                    "role": "user",
                    "content": minimal_prompt
                }
            ]
            schema = response_schema
        elif ablation_type == "bare_minimal":
            # Bare minimal: no context, no observations, generic prompt and schema
            bare_prompt = "Provide a percent value from 0 to 100."
            messages = [
                {
                    "role": "user",
                    "content": bare_prompt
                }
            ]
            schema = bare_minimal_schema
        else:
            # Normal case: both system and user messages
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            schema = response_schema
        
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "o4-mini-2025-04-16",
                "reasoning_effort": "high",
                "messages": messages,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "allocation_decision" if ablation_type != "bare_minimal" else "decision",
                        "strict": True,
                        "schema": schema
                    }
                }
            }
        }
        batch_requests.append(request)
        
        # Add to metadata
        metadata.append({
            "custom_id": custom_id,
            "date": date,
            "water_year": obs_data['wy'],
            "month": month,
            "sample_num": sample_num,
            "allocation_percent": allocation_percent,
            "ablation_type": ablation_type,
            "source_file": obs_data['source_file'],
            "index": i
        })
    
    # Write batch requests to JSONL file
    filename = f"ablation_{ablation_type}_month{month}_requests.jsonl"
    batch_file = os.path.join(output_dir, filename)
    
    with open(batch_file, "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")
    
    print(f"\nCreated {len(batch_requests)} batch requests")
    print(f"Batch file saved to: {batch_file}")
    
    # Create metadata file
    metadata_df = pd.DataFrame(metadata)
    metadata_file = os.path.join(output_dir, f"ablation_{ablation_type}_month{month}_metadata.csv")
    metadata_df.to_csv(metadata_file, index=False)
    
    print(f"Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    main()
