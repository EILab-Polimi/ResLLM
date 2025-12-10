#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submit batch requests to OpenAI API and retrieve results.
"""

import os
import json
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(verbose=True)


def upload_batch_file(client, batch_file_path):
    """Upload the batch input file to OpenAI."""
    print(f"Uploading batch file: {batch_file_path}")
    
    with open(batch_file_path, "rb") as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    print(f"File uploaded successfully!")
    print(f"File ID: {batch_input_file.id}")
    return batch_input_file.id


def create_batch(client, input_file_id, description=None):
    """Create a batch job."""
    print(f"\nCreating batch job with file ID: {input_file_id}")
    
    metadata = {}
    if description:
        metadata["description"] = description
    
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata
    )
    
    print(f"Batch created successfully!")
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created at: {batch.created_at}")
    print(f"Expires at: {batch.expires_at}")
    
    return batch.id


def check_batch_status(client, batch_id):
    """Check the status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    
    print(f"\nBatch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created at: {batch.created_at}")
    
    if batch.in_progress_at:
        print(f"Started at: {batch.in_progress_at}")
    if batch.completed_at:
        print(f"Completed at: {batch.completed_at}")
    if batch.failed_at:
        print(f"Failed at: {batch.failed_at}")
    if batch.expired_at:
        print(f"Expired at: {batch.expired_at}")
    
    print(f"\nRequest counts:")
    print(f"  Total: {batch.request_counts.total}")
    print(f"  Completed: {batch.request_counts.completed}")
    print(f"  Failed: {batch.request_counts.failed}")
    
    if batch.output_file_id:
        print(f"\nOutput file ID: {batch.output_file_id}")
    if batch.error_file_id:
        print(f"Error file ID: {batch.error_file_id}")
    
    return batch


def download_results(client, batch_id, output_dir="./output"):
    """Download the results of a completed batch job."""
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        print(f"Batch is not completed yet. Current status: {batch.status}")
        return None
    
    if not batch.output_file_id:
        print("No output file available.")
        return None
    
    print(f"\nDownloading results from file: {batch.output_file_id}")
    
    # Download the output file
    file_response = client.files.content(batch.output_file_id)
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"batch_results_{batch_id}.jsonl")
    
    with open(output_file, "w") as f:
        f.write(file_response.text)
    
    print(f"Results saved to: {output_file}")
    
    # Download error file if available
    if batch.error_file_id:
        print(f"\nDownloading errors from file: {batch.error_file_id}")
        error_response = client.files.content(batch.error_file_id)
        
        error_file = os.path.join(output_dir, f"batch_errors_{batch_id}.jsonl")
        with open(error_file, "w") as f:
            f.write(error_response.text)
        
        print(f"Errors saved to: {error_file}")
    
    return output_file


def list_batches(client, limit=10):
    """List recent batch jobs."""
    print(f"\nListing {limit} most recent batches:")
    
    batches = client.batches.list(limit=limit)
    
    for batch in batches.data:
        print(f"\nBatch ID: {batch.id}")
        print(f"  Status: {batch.status}")
        print(f"  Created: {batch.created_at}")
        print(f"  Requests: {batch.request_counts.total} total, "
              f"{batch.request_counts.completed} completed, "
              f"{batch.request_counts.failed} failed")


def cancel_batch(client, batch_id):
    """Cancel a batch job."""
    print(f"\nCancelling batch: {batch_id}")
    
    batch = client.batches.cancel(batch_id)
    
    print(f"Batch status: {batch.status}")
    return batch


def main():
    parser = argparse.ArgumentParser(description="Manage OpenAI batch jobs")
    parser.add_argument(
        "action",
        choices=["upload", "create", "status", "download", "list", "cancel"],
        help="Action to perform"
    )
    parser.add_argument(
        "--file",
        help="Path to batch input file (for upload action)"
    )
    parser.add_argument(
        "--file-id",
        help="File ID (for create action)"
    )
    parser.add_argument(
        "--batch-id",
        help="Batch ID (for status, download, cancel actions)"
    )
    parser.add_argument(
        "--description",
        help="Description for the batch job (for create action)"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for downloaded results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of batches to list (for list action)"
    )
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if args.action == "upload":
        if not args.file:
            print("Error: --file is required for upload action")
            return
        upload_batch_file(client, args.file)
    
    elif args.action == "create":
        if not args.file_id:
            print("Error: --file-id is required for create action")
            return
        create_batch(client, args.file_id, args.description)
    
    elif args.action == "status":
        if not args.batch_id:
            print("Error: --batch-id is required for status action")
            return
        check_batch_status(client, args.batch_id)
    
    elif args.action == "download":
        if not args.batch_id:
            print("Error: --batch-id is required for download action")
            return
        download_results(client, args.batch_id, args.output_dir)
    
    elif args.action == "list":
        list_batches(client, args.limit)
    
    elif args.action == "cancel":
        if not args.batch_id:
            print("Error: --batch-id is required for cancel action")
            return
        cancel_batch(client, args.batch_id)


if __name__ == "__main__":
    # Example usage:
    # 1. Upload batch file:
    #    python submit_batch.py upload --file ./output/batch_requests_1996_2016.jsonl
    #
    # 2. Create batch job (use file ID from step 1):
    #    python submit_batch.py create --file-id file-abc123 --description "Historical allocation decisions 1996-2016"
    #
    # 3. Check batch status:
    #    python submit_batch.py status --batch-id batch_abc123
    #
    # 4. List recent batches:
    #    python submit_batch.py list --limit 10
    #
    # 5. Download results when complete:
    #    python submit_batch.py download --batch-id batch_abc123 --output-dir ./output
    #
    # 6. Cancel a batch:
    #    python submit_batch.py cancel --batch-id batch_abc123
    
    main()
