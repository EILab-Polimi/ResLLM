#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create OpenAI Batch API ablation requests by removing one element from the recorded
observations in decision_output files (n0-n9), to study each element's impact on the
allocation decision.

The text-surgery (``split_system_and_user`` / ``remove_element_from_observation``) is shared
with the online ablation paths via ``src.ablation`` — this script is the OpenAI-Batch consumer
of that single source of truth. It runs in simple (non-complex) mode and passes
``extra_strip=("importance_ranking", "red_herring")`` to reproduce its reduced-schema output
(no concept-importance, no puppies) byte-for-byte.
"""

import os
import sys
import json
import argparse
import re
import pandas as pd
from glob import glob

# Reach the resllm/ package root so the shared ablation module is importable, then keep
# batch/src (this script's dir, already on sys.path[0]) for the local ``schemas`` import.
_HERE = os.path.dirname(os.path.abspath(__file__))            # resllm/batch/src
_RESLLM = os.path.abspath(os.path.join(_HERE, "..", ".."))    # resllm
if _RESLLM not in sys.path:
    sys.path.append(_RESLLM)

from schemas import RESPONSE_SCHEMA
from src.ablation import (
    SIMPLE_ABLATION_TYPES,
    remove_element_from_observation,
    split_system_and_user,
)

# Reduced-schema behavior for every ablation: drop the concept-importance ranking line and
# the puppies red herring (reproduces the historical batch output).
_BATCH_EXTRA_STRIP = ("importance_ranking", "red_herring")


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
        choices=list(SIMPLE_ABLATION_TYPES),
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

    # Response schema (default) — shared with create_batch_requests.py
    response_schema = RESPONSE_SCHEMA

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

        # Split into system and user messages before ablating.
        system_message, user_message = split_system_and_user(observation)

        # Ablate each part (simple mode; _BATCH_EXTRA_STRIP also drops the importance-ranking
        # and red-herring lines for the reduced-schema output).
        system_message = remove_element_from_observation(
            system_message, ablation_type, complexity_mode=False, extra_strip=_BATCH_EXTRA_STRIP
        )
        user_message = remove_element_from_observation(
            user_message, ablation_type, complexity_mode=False, extra_strip=_BATCH_EXTRA_STRIP
        )

        # Create request
        custom_id = f"ablation_{ablation_type}_month{month}_n{sample_num}_date{date.replace('-', '')}_alloc{allocation_percent}"

        # Build the messages list per ablation type.
        if ablation_type == "no_system":
            # Drop the system prompt; user message only.
            messages = [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
            schema = response_schema
        elif ablation_type == "minimal":
            # No system prompt; the final request line only.
            minimal_prompt = "Provide a percent allocation decision (from 0-100 percent) which continues or updates the allocation."
            messages = [
                {
                    "role": "user",
                    "content": minimal_prompt
                }
            ]
            schema = response_schema
        elif ablation_type == "bare_minimal":
            # No context or observation; generic prompt and schema.
            bare_prompt = "Provide a percent value from 0 to 100."
            messages = [
                {
                    "role": "user",
                    "content": bare_prompt
                }
            ]
            schema = bare_minimal_schema
        else:
            # Normal case: both system and user messages.
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
