#!/bin/bash
# Complete end-to-end workflow for ablation batch processing
# This script creates, submits, monitors, and processes ablation batch requests
#
# Usage:
#   ./run_ablation_workflow.sh --ablation-type TYPE --month N [OPTIONS]
#
# Required:
#   --ablation-type TYPE   Type of ablation (current_storage, forecasts, etc.)
#   --month N              Month of water year (1-12)
#
# Options:
#   --skip-create          Skip batch creation step (use existing batch file)
#   --batch-id ID          Monitor existing batch instead of creating new one
#   --model-prefix PREFIX  Model prefix for decision_output files (default: o4-mini-2025-04-16)
#   --help                 Show this help message

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ABLATION_TYPE=""
MONTH=""
SKIP_CREATE=false
BATCH_ID=""
MONITOR_ONLY=false
MODEL_PREFIX="o4-mini-2025-04-16"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ablation-type)
            ABLATION_TYPE="$2"
            shift 2
            ;;
        --month)
            MONTH="$2"
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
        --model-prefix)
            MODEL_PREFIX="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_ablation_workflow.sh --ablation-type TYPE --month N [OPTIONS]"
            echo ""
            echo "Required Arguments:"
            echo "  --ablation-type TYPE   Ablation type to run:"
            echo "                         - current_storage"
            echo "                         - forecasts"
            echo "                         - previous_allocation"
            echo "                         - remaining_demand"
            echo "                         - cumulative_inflow"
            echo "                         - storage_and_inflow"
            echo "                         - next_year_demand"
            echo "                         - no_system"
            echo "  --month N              Month of water year (1-12)"
            echo ""
            echo "Options:"
            echo "  --skip-create          Skip batch creation, use existing file"
            echo "  --batch-id ID          Monitor existing batch ID"
            echo "  --model-prefix PREFIX  Model prefix (default: o4-mini-2025-04-16)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Create and run current_storage ablation for month 9"
            echo "  ./run_ablation_workflow.sh --ablation-type current_storage --month 9"
            echo ""
            echo "  # Monitor existing batch"
            echo "  ./run_ablation_workflow.sh --ablation-type forecasts --month 9 --batch-id batch_xyz"
            echo ""
            echo "  # Skip creation, use existing files"
            echo "  ./run_ablation_workflow.sh --ablation-type remaining_demand --month 9 --skip-create"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$ABLATION_TYPE" ]]; then
    echo -e "${RED}Error: --ablation-type is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

if [[ -z "$MONTH" ]]; then
    echo -e "${RED}Error: --month is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Define file paths based on ablation type and month
REQUESTS_FILE="./output/ablation_${ABLATION_TYPE}_month${MONTH}_requests.jsonl"
METADATA_FILE="./output/ablation_${ABLATION_TYPE}_month${MONTH}_metadata.csv"
RESULTS_FILE="./output/ablation_${ABLATION_TYPE}_month${MONTH}_results.csv"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Ablation Batch Workflow${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "Ablation Type: ${GREEN}${ABLATION_TYPE}${NC}"
echo -e "Month: ${GREEN}${MONTH}${NC}"
echo -e "Model Prefix: ${GREEN}${MODEL_PREFIX}${NC}"
echo ""

# Step 1: Create ablation batch requests (unless skipping or monitoring only)
if [[ "$MONITOR_ONLY" == false && "$SKIP_CREATE" == false ]]; then
    echo -e "${YELLOW}Step 1: Creating ablation batch requests...${NC}"
    python ./src/create_ablation_batch_requests.py \
        --month "$MONTH" \
        --ablation-type "$ABLATION_TYPE" \
        --model-prefix "$MODEL_PREFIX"
    
    echo -e "${GREEN}✓ Batch requests created${NC}"
    echo -e "  Requests file: $REQUESTS_FILE"
    echo -e "  Metadata file: $METADATA_FILE"
    echo ""
elif [[ "$SKIP_CREATE" == true ]]; then
    echo -e "${YELLOW}Step 1: Skipping batch creation (using existing files)${NC}"
    if [[ ! -f "$REQUESTS_FILE" ]]; then
        echo -e "${RED}Error: Requests file not found: $REQUESTS_FILE${NC}"
        exit 1
    fi
    if [[ ! -f "$METADATA_FILE" ]]; then
        echo -e "${RED}Error: Metadata file not found: $METADATA_FILE${NC}"
        exit 1
    fi
    echo ""
fi

# Step 2: Upload and create batch (unless monitoring existing)
if [[ "$MONITOR_ONLY" == false ]]; then
    echo -e "${YELLOW}Step 2: Uploading batch file to OpenAI...${NC}"
    
    # Upload file and capture the file ID
    UPLOAD_OUTPUT=$(python ./src/submit_batch.py upload --file "$REQUESTS_FILE" 2>&1)
    echo "$UPLOAD_OUTPUT"
    FILE_ID=$(echo "$UPLOAD_OUTPUT" | grep "File ID:" | awk '{print $3}')
    
    if [[ -z "$FILE_ID" ]]; then
        echo -e "${RED}Error: Failed to extract file ID from upload${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ File uploaded: $FILE_ID${NC}"
    echo ""
    
    echo -e "${YELLOW}Step 3: Creating batch job...${NC}"
    
    # Create batch and capture the batch ID
    BATCH_OUTPUT=$(python ./src/submit_batch.py create --file-id "$FILE_ID" 2>&1)
    echo "$BATCH_OUTPUT"
    BATCH_ID=$(echo "$BATCH_OUTPUT" | grep "Batch ID:" | awk '{print $3}')
    
    if [[ -z "$BATCH_ID" ]]; then
        echo -e "${RED}Error: Failed to extract batch ID from creation${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Batch created: $BATCH_ID${NC}"
    echo ""
else
    echo -e "${YELLOW}Steps 1-3: Skipping creation (monitoring existing batch)${NC}"
    echo -e "Batch ID: ${GREEN}$BATCH_ID${NC}"
    echo ""
fi

# Step 4: Monitor batch until completion
echo -e "${YELLOW}Step 4: Monitoring batch progress...${NC}"
echo -e "Batch ID: ${GREEN}$BATCH_ID${NC}"
echo ""

POLL_INTERVAL=60  # seconds
MAX_WAIT_TIME=86400  # 24 hours in seconds
ELAPSED_TIME=0

while true; do
    # Check batch status
    STATUS_OUTPUT=$(python ./src/submit_batch.py status --batch-id "$BATCH_ID" 2>&1)
    
    # Extract status
    STATUS=$(echo "$STATUS_OUTPUT" | grep "^Status:" | awk '{print $2}')
    COMPLETED=$(echo "$STATUS_OUTPUT" | grep "Completed:" | tail -1 | awk '{print $2}')
    TOTAL=$(echo "$STATUS_OUTPUT" | grep "Total:" | tail -1 | awk '{print $2}')
    
    # Print status update
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[$TIMESTAMP]${NC} Status: $STATUS | Progress: $COMPLETED/$TOTAL"
    
    # Check if completed
    if [[ "$STATUS" == "completed" ]]; then
        echo ""
        echo -e "${GREEN}✓ Batch completed successfully!${NC}"
        echo "$STATUS_OUTPUT"
        break
    elif [[ "$STATUS" == "failed" || "$STATUS" == "expired" || "$STATUS" == "cancelled" ]]; then
        echo ""
        echo -e "${RED}✗ Batch ended with status: $STATUS${NC}"
        echo "$STATUS_OUTPUT"
        exit 1
    fi
    
    # Check timeout
    if [[ $ELAPSED_TIME -ge $MAX_WAIT_TIME ]]; then
        echo ""
        echo -e "${RED}✗ Timeout: Batch did not complete within 24 hours${NC}"
        exit 1
    fi
    
    # Wait before next check
    sleep $POLL_INTERVAL
    ELAPSED_TIME=$((ELAPSED_TIME + POLL_INTERVAL))
done

echo ""

# Step 5: Download results
echo -e "${YELLOW}Step 5: Downloading batch results...${NC}"

DOWNLOAD_OUTPUT=$(python ./src/submit_batch.py download --batch-id "$BATCH_ID" 2>&1)
echo "$DOWNLOAD_OUTPUT"

# Extract results file path
RESULTS_JSONL=$(echo "$DOWNLOAD_OUTPUT" | grep "Results saved to:" | sed 's/Results saved to: //')

if [[ -z "$RESULTS_JSONL" || ! -f "$RESULTS_JSONL" ]]; then
    echo -e "${RED}Error: Failed to download results${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Results downloaded: $RESULTS_JSONL${NC}"
echo ""

# Step 6: Process results
echo -e "${YELLOW}Step 6: Processing batch results...${NC}"

python ./src/process_batch_results.py \
    --results "$RESULTS_JSONL" \
    --metadata "$METADATA_FILE" \
    --output "$RESULTS_FILE"

if [[ -f "$RESULTS_FILE" ]]; then
    echo -e "${GREEN}✓ Results processed and saved${NC}"
    echo -e "  Output file: $RESULTS_FILE"
    
    # Show summary statistics
    echo ""
    echo -e "${BLUE}Summary Statistics:${NC}"
    NUM_RESULTS=$(tail -n +2 "$RESULTS_FILE" | wc -l | xargs)
    echo -e "  Total observations: ${GREEN}$NUM_RESULTS${NC}"
    
    # Count errors if any
    NUM_ERRORS=$(tail -n +2 "$RESULTS_FILE" | awk -F',' '{print $8}' | grep -v '^$' | wc -l | xargs)
    if [[ $NUM_ERRORS -gt 0 ]]; then
        echo -e "  Errors: ${RED}$NUM_ERRORS${NC}"
    else
        echo -e "  Errors: ${GREEN}0${NC}"
    fi
else
    echo -e "${RED}✗ Error: Results file not created${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Ablation workflow completed successfully!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "Files created:"
echo -e "  1. Requests: ${BLUE}$REQUESTS_FILE${NC}"
echo -e "  2. Metadata: ${BLUE}$METADATA_FILE${NC}"
echo -e "  3. Results:  ${BLUE}$RESULTS_FILE${NC}"
echo ""
echo -e "Batch ID: ${BLUE}$BATCH_ID${NC}"
echo ""
echo -e "Next steps:"
echo -e "  - Copy results to analysis folder:"
echo -e "    ${BLUE}cp $RESULTS_FILE ../../analysis/output/resllm/batch/ablation/${NC}"
echo -e "  - Run analysis script:"
echo -e "    ${BLUE}cd ../../analysis && python _4_ablation.py${NC}"
echo ""
