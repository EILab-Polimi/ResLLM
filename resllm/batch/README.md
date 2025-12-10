# Batch Processing for Reservoir Decisions

Generate allocation decisions using OpenAI Batch API (1996-2016).

## Prerequisites

```bash
# .env file
OPENAI_API_KEY=sk-...

# Install
pip install openai pandas numpy pyyaml python-dotenv
```

## Standard Workflow

```bash
# Automated (recommended)
./run_batch_workflow.sh --n-samples 10

# Manual steps
python src/create_batch_requests.py --n-samples 10
python src/submit_batch.py upload --file ./output/batch_requests_1996_2016.jsonl
python src/submit_batch.py create --file-id file-xxxxx
python src/monitor_batch.py --batch-id batch_xxxxx
python src/process_batch_results.py --results ... --metadata ... --output ...
```

## Ablation Studies

Test impact of removing specific information:

```bash
# Automated (recommended)
./run_ablation_workflow.sh --ablation-type bare_minimal --month 5
./run_ablation_workflow.sh --ablation-type minimal --month 5
./run_ablation_workflow.sh --ablation-type default --month 8
./run_ablation_workflow.sh --ablation-type no_system --month 8
./run_ablation_workflow.sh --ablation-type previous_allocation --month 8
./run_ablation_workflow.sh --ablation-type demand --month 8
./run_ablation_workflow.sh --ablation-type cumulative_inflow --month 8
./run_ablation_workflow.sh --ablation-type current_storage --month 8
./run_ablation_workflow.sh --ablation-type storage_and_inflow --month 8
./run_ablation_workflow.sh --ablation-type forecasts --month 8
./run_ablation_workflow.sh --ablation-type current_month --month 8
./run_ablation_workflow.sh --ablation-type forecast_p10--month 8
./run_ablation_workflow.sh --ablation-type forecast_p90 --month 8
./run_ablation_workflow.sh --ablation-type forecast_mean --month 8

  ✗ forecast_p10 month 3: batch not found
  ✗ forecast_mean month 9: batch not found
  ✗ forecast_mean month 10: batch not found
  ✗ forecast_p90 month 10: batch not found
  ✗ forecast_p90 month 11: batch not found

./run_ablation_workflow.sh --ablation-type forecast_mean --month 9
./run_ablation_workflow.sh --ablation-type forecast_mean --month 10
./run_ablation_workflow.sh --ablation-type forecast_p10 --month 3
./run_ablation_workflow.sh --ablation-type forecast_p90 --month 10
./run_ablation_workflow.sh --ablation-type forecast_p90 --month 11

# Manual
python src/create_ablation_batch_requests.py --month 1 --ablation-type forecast_p10
# Then upload/submit/monitor/process as above

python src/process_batch_results.py --results ./output/batch_6935fbaed3408190aec59fff19b7e9aa.jsonl --metadata ./output/ablation_forecast_mean_month9_metadata.csv --output ./output/ablation_forecast_mean_month9_results.csv

```

### Ablation Types

- `current_storage` - Remove current reservoir storage
- `forecasts` - Remove probabilistic inflow forecasts
- `previous_allocation` - Remove previous allocation decision
- `remaining_demand` - Remove remaining water year demand
- `cumulative_inflow` - Remove cumulative inflow to date
- `storage_and_inflow` - Remove both storage and inflow
- `next_year_demand` - Remove next year demand notes
- `no_system` - Remove entire system prompt
- `minimal` - Only allocation request, no context
- `bare_minimal` - Generic "provide percent" prompt

### Output

- Requests: `ablation_{type}_month{month}_requests.jsonl`
- Metadata: `ablation_{type}_month{month}_metadata.csv`
- Results: `ablation_{type}_month{month}_results.csv`

Uses 210 observations from n0-n9 decision outputs (21 dates × 10 samples).

