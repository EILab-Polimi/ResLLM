# MLP Neural Network Benchmark

This benchmark trains a compact MLP to predict monthly reservoir allocation decisions and then replays those decisions in a simulation.

Model: 5 inputs → 32 ReLU → 16 ReLU → 1 output (allocation in 0–1). Training uses WY 1961–1995; testing uses WY 1996–2016.

Inputs: storage (TAF), month (sin/cos), previous allocation (30‑day rolling mean of release/demand), inflow (120‑day moving average). Target: 30‑day forward allocation.

Run:
```bash
python mlp_train.py

# Historical simulation (default WY 1996–2016)
python mlp_simulate.py

# Fixed TOCS option
python mlp_simulate.py --fix-tocs
```

Outputs are written to ./output/:
- Model: mlp_allocation_model.pkl, mlp_allocation_scalers.pkl, mlp_allocation_metadata.json
- Training: mlp_allocation_results.png, mlp_allocation_train_predictions.csv, mlp_allocation_test_predictions.csv
- Simulation: mlp_simulation_output_{config}_{tocs}{start}_{end}.csv, mlp_decision_output_{config}_{tocs}{start}_{end}.csv
