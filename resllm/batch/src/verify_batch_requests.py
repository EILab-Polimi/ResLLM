#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify batch requests file before submission.
"""

import json
import pandas as pd
import argparse


def verify_batch_requests(batch_file, metadata_file):
    """
    Verify batch requests file and metadata.
    
    Args:
        batch_file: Path to batch requests JSONL file
        metadata_file: Path to metadata CSV file
    """
    
    print("=" * 70)
    print("Batch Request Verification")
    print("=" * 70)
    
    # Load metadata
    print(f"\n1. Loading metadata from: {metadata_file}")
    metadata = pd.read_csv(metadata_file)
    print(f"   ✓ Found {len(metadata)} metadata records")
    
    # Load batch requests
    print(f"\n2. Loading batch requests from: {batch_file}")
    requests = []
    with open(batch_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                req = json.loads(line)
                requests.append(req)
            except json.JSONDecodeError as e:
                print(f"   ❌ Error on line {line_num}: {e}")
                return False
    
    print(f"   ✓ Found {len(requests)} valid JSON requests")
    
    # Check counts match
    print("\n3. Checking request counts...")
    if len(requests) != len(metadata):
        print(f"   ❌ Mismatch: {len(requests)} requests but {len(metadata)} metadata records")
        return False
    print(f"   ✓ Counts match: {len(requests)} requests")
    
    # Verify request structure
    print("\n4. Verifying request structure...")
    errors = []
    
    for i, req in enumerate(requests):
        # Check required fields
        if "custom_id" not in req:
            errors.append(f"Request {i}: Missing custom_id")
        if "method" not in req or req["method"] != "POST":
            errors.append(f"Request {i}: Invalid or missing method")
        if "url" not in req or req["url"] != "/v1/chat/completions":
            errors.append(f"Request {i}: Invalid or missing URL")
        
        # Check body
        if "body" not in req:
            errors.append(f"Request {i}: Missing body")
            continue
        
        body = req["body"]
        
        # Check model
        if "model" not in body or body["model"] != "o4-mini-2025-04-16":
            errors.append(f"Request {i}: Invalid or missing model")
        
        # Check messages
        if "messages" not in body or len(body["messages"]) != 2:
            errors.append(f"Request {i}: Invalid messages structure")
        else:
            if body["messages"][0]["role"] != "system":
                errors.append(f"Request {i}: First message should be system")
            if body["messages"][1]["role"] != "user":
                errors.append(f"Request {i}: Second message should be user")
        
        # Check response format
        if "response_format" not in body:
            errors.append(f"Request {i}: Missing response_format")
        
        # Check reasoning (optional - only for some models/endpoints)
        # if "reasoning" not in body or body["reasoning"].get("effort") != "high":
        #     errors.append(f"Request {i}: Missing or invalid reasoning config")
    
    if errors:
        print(f"   ❌ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"      - {error}")
        if len(errors) > 10:
            print(f"      ... and {len(errors) - 10} more")
        return False
    
    print(f"   ✓ All requests have valid structure")
    
    # Verify custom IDs match
    print("\n5. Verifying custom_id consistency...")
    request_ids = {req["custom_id"] for req in requests}
    metadata_ids = set(metadata["custom_id"].values)
    
    missing_in_metadata = request_ids - metadata_ids
    missing_in_requests = metadata_ids - request_ids
    
    if missing_in_metadata:
        print(f"   ❌ {len(missing_in_metadata)} IDs in requests but not in metadata")
        return False
    if missing_in_requests:
        print(f"   ❌ {len(missing_in_requests)} IDs in metadata but not in requests")
        return False
    
    print(f"   ✓ All custom_ids match")
    
    # Check date range
    print("\n6. Checking date range...")
    metadata["date"] = pd.to_datetime(metadata["date"])
    min_date = metadata["date"].min()
    max_date = metadata["date"].max()
    
    print(f"   Start date: {min_date.strftime('%Y-%m-%d')}")
    print(f"   End date:   {max_date.strftime('%Y-%m-%d')}")
    
    expected_start = pd.Timestamp("1996-10-01")
    expected_end = pd.Timestamp("2016-09-01")
    
    if min_date != expected_start:
        print(f"   ⚠️  Warning: Expected start date {expected_start.strftime('%Y-%m-%d')}")
    if max_date != expected_end:
        print(f"   ⚠️  Warning: Expected end date {expected_end.strftime('%Y-%m-%d')}")
    
    # Check for duplicates
    print("\n7. Checking for duplicate dates...")
    duplicates = metadata[metadata["date"].duplicated()]
    if len(duplicates) > 0:
        print(f"   ❌ Found {len(duplicates)} duplicate dates")
        return False
    print(f"   ✓ No duplicate dates")
    
    # Sample request inspection
    print("\n8. Sample request (first request):")
    sample = requests[0]
    print(f"   Custom ID: {sample['custom_id']}")
    print(f"   Model: {sample['body']['model']}")
    
    user_message = sample['body']['messages'][1]['content']
    print(f"   User message preview: {user_message[:200]}...")
    
    # Statistics
    print("\n9. Statistics:")
    print(f"   Total requests: {len(requests)}")
    print(f"   Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"   Duration: {(max_date - min_date).days} days")
    print(f"   Water years: {metadata['water_year'].min()} to {metadata['water_year'].max()}")
    
    # Estimate token count (rough)
    total_chars = sum(len(json.dumps(req)) for req in requests)
    estimated_tokens = total_chars / 4  # Rough estimate: 1 token ≈ 4 characters
    print(f"   Estimated total tokens: {int(estimated_tokens):,}")
    print(f"   Estimated cost (batch rates): ${estimated_tokens / 1_000_000 * 0.15:.2f}")
    
    print("\n" + "=" * 70)
    print("✓ Verification complete - All checks passed!")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify batch requests file")
    parser.add_argument(
        "--batch-file",
        default="./output/batch_requests_1996_2016.jsonl",
        help="Path to batch requests JSONL file"
    )
    parser.add_argument(
        "--metadata",
        default="./output/batch_requests_metadata_1996_2016.csv",
        help="Path to metadata CSV file"
    )
    
    args = parser.parse_args()
    
    success = verify_batch_requests(args.batch_file, args.metadata)
    
    if success:
        print("\nReady to submit!")
        print("\nNext step:")
        print(f"  python src/submit_batch.py upload --file {args.batch_file}")
    else:
        print("\n❌ Verification failed. Please fix errors before submitting.")
        return 1
    
    return 0


if __name__ == "__main__":
    # Example usage:
    # python verify_batch_requests.py
    # python verify_batch_requests.py --batch-file custom_batch.jsonl --metadata custom_metadata.csv
    
    exit(main())
