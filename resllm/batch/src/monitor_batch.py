#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor OpenAI batch job status and automatically download results when complete.
"""

import os
import sys
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(verbose=True)


def monitor_batch(batch_id, check_interval=300, auto_download=True, output_dir="./output"):
    """
    Monitor a batch job and optionally auto-download results when complete.
    
    Args:
        batch_id: The batch ID to monitor
        check_interval: Seconds between status checks (default: 60 = 1 minute)
        auto_download: Whether to automatically download results when complete
        output_dir: Directory to save downloaded results
    """
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print(f"Monitoring batch: {batch_id}")
    print(f"Check interval: {check_interval} seconds ({check_interval/60:.1f} minutes)")
    print(f"Auto-download: {auto_download}")
    print("")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 70)
    
    try:
        while True:
            # Get batch status
            batch = client.batches.retrieve(batch_id)
            
            # Print status
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Status: {batch.status}")
            
            if batch.request_counts:
                print(f"  Requests - Total: {batch.request_counts.total}, "
                      f"Completed: {batch.request_counts.completed}, "
                      f"Failed: {batch.request_counts.failed}")
            
            if batch.status == "completed":
                print("\n✓ Batch completed successfully!")
                
                if batch.output_file_id:
                    print(f"  Output file ID: {batch.output_file_id}")
                
                if batch.error_file_id:
                    print(f"  Error file ID: {batch.error_file_id}")
                    print(f"  Warning: Some requests failed. Check error file.")
                
                if auto_download:
                    print(f"\nDownloading results to: {output_dir}")
                    
                    # Download output file
                    file_response = client.files.content(batch.output_file_id)
                    
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f"batch_results_{batch_id}.jsonl")
                    
                    with open(output_file, "w") as f:
                        f.write(file_response.text)
                    
                    print(f"  ✓ Results saved to: {output_file}")
                    
                    # Download error file if available
                    if batch.error_file_id:
                        error_response = client.files.content(batch.error_file_id)
                        error_file = os.path.join(output_dir, f"batch_errors_{batch_id}.jsonl")
                        
                        with open(error_file, "w") as f:
                            f.write(error_response.text)
                        
                        print(f"  ✓ Errors saved to: {error_file}")
                    
                    print("\nNext step: Process the results with:")
                    print(f"  python src/process_batch_results.py \\")
                    print(f"    --results {output_file} \\")
                    print(f"    --metadata ./output/batch_requests_metadata_1996_2016.csv \\")
                    print(f"    --output ./output/allocation_decisions_1996_2016.csv")

                break
            
            elif batch.status == "failed":
                print("\n❌ Batch failed!")
                if batch.errors:
                    print(f"  Error: {batch.errors}")
                break
            
            elif batch.status == "expired":
                print("\n⏱ Batch expired (did not complete within 24 hours)")
                break
            
            elif batch.status == "cancelled":
                print("\n🚫 Batch was cancelled")
                break
            
            elif batch.status in ["validating", "in_progress", "finalizing"]:
                # Still processing
                if batch.status == "validating":
                    print("  Validating input file...")
                elif batch.status == "in_progress":
                    if batch.request_counts:
                        progress = (batch.request_counts.completed / batch.request_counts.total * 100
                                  if batch.request_counts.total > 0 else 0)
                        print(f"  Progress: {progress:.1f}%")
                elif batch.status == "finalizing":
                    print("  Finalizing results...")
                
                # Wait before next check
                print(f"  Next check in {check_interval} seconds...")
                time.sleep(check_interval)
            
            else:
                print(f"  Unknown status: {batch.status}")
                time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        print(f"\nTo check status again later, run:")
        print(f"  python submit_batch.py status --batch-id {batch_id}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Monitor OpenAI batch job")
    parser.add_argument(
        "--batch-id",
        help="Batch ID to monitor (if not provided, reads from ./output/latest_batch_id.txt)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60 = 1 minute)"
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable automatic download of results when complete"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for downloaded results"
    )
    
    args = parser.parse_args()
    
    # Get batch ID
    batch_id = args.batch_id
    
    if not batch_id:
        # Try to read from latest_batch_id.txt
        batch_id_file = os.path.join(args.output_dir, "latest_batch_id.txt")
        if os.path.exists(batch_id_file):
            with open(batch_id_file, "r") as f:
                batch_id = f.read().strip()
            print(f"Using batch ID from {batch_id_file}: {batch_id}")
        else:
            print(f"Error: No batch ID provided and {batch_id_file} not found")
            print("\nUsage:")
            print("  python monitor_batch.py --batch-id batch_xxxxxxxx")
            print("  or")
            print("  python monitor_batch.py  # reads from ./output/latest_batch_id.txt")
            sys.exit(1)
    
    monitor_batch(
        batch_id=batch_id,
        check_interval=args.interval,
        auto_download=not args.no_auto_download,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    # Example usage:
    # python monitor_batch.py --batch-id batch_xxxxxxxx
    # python monitor_batch.py --batch-id batch_xxxxxxxx --interval 600  # check every 10 minutes
    # python monitor_batch.py  # uses latest batch ID from file
    
    main()
