#!/bin/bash
# Complete end-to-end workflow for OpenAI batch processing
# This script creates, submits, monitors, and processes batch requests
#
# Usage:
#   ./run_batch_workflow.sh [--n-samples N] [--skip-create] [--batch-id ID]
#
# Options:
#   --n-samples N      Number of samples per date (default: 1)
#   --skip-create      Skip batch creation step (use existing batch file)
#   --batch-id ID      Monitor existing batch instead of creating new one
#   --help             Show this help message

set -e  # Exit on error

# Default values
N_SAMPLES=1
SKIP_CREATE=false
BATCH_ID=""
MONITOR_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --skip-create)
            SKIP_CREATE=true
            shift
            ;;
        --batch-id)
            BATCH_ID="$2"
            MONITOR_ONLY=true
            shift 2
            ;;
        --help)
            echo "Usage: ./run_batch_workflow.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --n-samples N      Number of samples per date (default: 1)"
            echo "  --skip-create      Skip batch creation, use existing file"
            echo "  --batch-id ID      Monitor existing batch ID"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_batch_workflow.sh                    # Single sample batch"
            echo "  ./run_batch_workflow.sh --n-samples 3      # 3 samples per date"
            echo "  ./run_batch_workflow.sh --batch-id batch_xyz  # Monitor existing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to the batch directory
cd "$(dirname "$0")"

echo "==================================================================="
echo "OpenAI Batch Processing - Complete Workflow"
echo "==================================================================="
echo ""
echo "Configuration:"
echo "  Samples per date: $N_SAMPLES"
echo "  Skip creation: $SKIP_CREATE"
if [ "$MONITOR_ONLY" = true ]; then
    echo "  Mode: Monitor existing batch"
    echo "  Batch ID: $BATCH_ID"
else
    echo "  Mode: Create and submit new batch"
fi
echo ""
echo "==================================================================="
echo ""

# Determine file names based on n_samples
if [ "$N_SAMPLES" -gt 1 ]; then
    BATCH_FILE="./output/batch_requests_1996_2016_n${N_SAMPLES}.jsonl"
    METADATA_FILE="./output/batch_requests_metadata_1996_2016_n${N_SAMPLES}.csv"
    OUTPUT_FILE="./output/allocation_decisions_1996_2016_n${N_SAMPLES}.csv"
else
    BATCH_FILE="./output/batch_requests_1996_2016.jsonl"
    METADATA_FILE="./output/batch_requests_metadata_1996_2016.csv"
    OUTPUT_FILE="./output/allocation_decisions_1996_2016.csv"
fi

# If monitoring existing batch, skip to monitoring
if [ "$MONITOR_ONLY" = true ]; then
    echo "Skipping to monitoring step..."
    echo ""
    # Find the metadata file (try both naming conventions)
    if [ ! -f "$METADATA_FILE" ]; then
        METADATA_FILE="./output/batch_requests_metadata_1996_2016.csv"
    fi
    
    # Jump to monitoring section
    SKIP_CREATE=true
    # BATCH_ID is already set from command line
fi

# Step 1: Create batch requests (unless skipped)
if [ "$SKIP_CREATE" = false ]; then
    echo "Step 1: Creating batch requests..."
    echo "  Samples per date: $N_SAMPLES"
    
    if [ "$N_SAMPLES" -gt 1 ]; then
        python src/create_batch_requests.py --n-samples "$N_SAMPLES"
    else
        python src/create_batch_requests.py
    fi
    
    echo ""
    echo "✓ Batch requests created!"
    echo "  - Batch file: $BATCH_FILE"
    echo "  - Metadata:   $METADATA_FILE"
    echo ""
else
    echo "Step 1: Skipped (using existing batch file)"
    echo "  - Batch file: $BATCH_FILE"
    echo ""
fi

# Only proceed with upload/create if not monitoring existing batch
if [ "$MONITOR_ONLY" = false ]; then
    # Step 2: Upload batch file
    echo "Step 2: Uploading batch file to OpenAI..."
    echo ""
    
    FILE_OUTPUT=$(python src/submit_batch.py upload --file "$BATCH_FILE")
    echo "$FILE_OUTPUT"
    
    # Extract file ID from output
    FILE_ID=$(echo "$FILE_OUTPUT" | grep "File ID:" | awk '{print $3}')
    
    if [ -z "$FILE_ID" ]; then
        echo "❌ Error: Could not extract file ID from upload output"
        exit 1
    fi
    
    echo ""
    echo "✓ File uploaded successfully!"
    echo "  File ID: $FILE_ID"
    echo ""
    
    # Step 3: Create batch job
    echo "Step 3: Creating batch job..."
    echo ""
    
    BATCH_OUTPUT=$(python src/submit_batch.py create --file-id "$FILE_ID" --description "Historical allocation decisions 1996-2016 (n=$N_SAMPLES)")
    echo "$BATCH_OUTPUT"
    
    # Extract batch ID from output
    BATCH_ID=$(echo "$BATCH_OUTPUT" | grep "Batch ID:" | awk '{print $3}')
    
    if [ -z "$BATCH_ID" ]; then
        echo "❌ Error: Could not extract batch ID from create output"
        exit 1
    fi
    
    echo ""
    echo "✓ Batch job created successfully!"
    echo "  Batch ID: $BATCH_ID"
    echo ""
    
    # Save batch ID to file for later reference
    echo "$BATCH_ID" > ./output/latest_batch_id.txt
    echo "  Batch ID saved to: ./output/latest_batch_id.txt"
    echo ""
fi

# Step 4: Monitor batch progress
echo "==================================================================="
echo "Step 4: Monitoring batch progress..."
echo "==================================================================="
echo ""
echo "Batch ID: $BATCH_ID"
echo ""
echo "This will check status every 5 minutes and auto-download when complete."
echo "Press Ctrl+C to stop monitoring (batch will continue processing)."
echo ""

# Monitor the batch with auto-download
python src/monitor_batch.py --batch-id "$BATCH_ID" --output-dir ./output

# Check if monitoring completed successfully
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Monitoring was interrupted or batch has not completed yet."
    echo ""
    echo "To resume monitoring later, run:"
    echo "  ./run_batch_workflow.sh --batch-id $BATCH_ID"
    echo ""
    echo "To check status manually:"
    echo "  python src/submit_batch.py status --batch-id $BATCH_ID"
    exit 0
fi

# Step 5: Process results
echo ""
echo "==================================================================="
echo "Step 5: Processing batch results..."
echo "==================================================================="
echo ""

RESULTS_FILE="./output/batch_results_${BATCH_ID}.jsonl"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "❌ Error: Results file not found: $RESULTS_FILE"
    echo ""
    echo "Download results manually with:"
    echo "  python src/submit_batch.py download --batch-id $BATCH_ID --output-dir ./output"
    exit 1
fi

python src/process_batch_results.py \
    --results "$RESULTS_FILE" \
    --metadata "$METADATA_FILE" \
    --output "$OUTPUT_FILE"

echo ""
echo "==================================================================="
echo "✓ Workflow Complete!"
echo "==================================================================="
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "  - Analyze allocation decisions"
echo "  - Run reservoir simulation with LLM decisions"
echo "  - Compare to benchmark methods (DP, MLP)"
echo ""
if [ "$N_SAMPLES" -gt 1 ]; then
    echo "Multi-sample analysis tips:"
    echo "  - Calculate statistics: df.groupby('date')['allocation_percent'].agg(['mean', 'std'])"
    echo "  - Find high variability: dates where std > 10%"
    echo "  - Use ensemble: median or filtered mean for robust decisions"
    echo ""
fi
echo "Batch ID: $BATCH_ID"
echo ""
