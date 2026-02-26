# ResLLM

ResLLM is a Python library for simulating water reservoir management with large language models (LLMs).

## Installation

```bash
git clone https://github.com/wyattarnold/ResLLM.git
cd ResLLM

# Conda (recommended)
conda env create -f environment.yml
conda activate llm

# For cloud APIs, create a .env file
cp .env.example .env
# Add provider keys as needed: OPENAI_API_KEY, GOOGLE_API_KEY, XAI_API_KEY, MISTRAL_API_KEY
```

## Quick Start
Run a 21-year simulation with OpenAI o4-mini:
```bash
cd resllm
python simulate.py \
  --model-server OpenAI \
  --model o4-mini-2025-04-16 \
  --config folsom.yml \
  --start-year 1996 \
  --end-year 2016 \
  --starting-storage 466.1 \
  --tocs historical \
  --wy-forecast-file FOLC1_wy_hindcast.csv
```


For the full CLI reference, model configuration, output formats, and architecture details, see [resllm/README.md](resllm/README.md).

## Project Structure

```
ResLLM/
├── resllm/                  # Core simulation library (run from here)
│   ├── simulate.py          # Main entry point
│   ├── configs/             # Reservoir configuration YAML files
│   ├── output/              # Simulation results
│   ├── batch/               # OpenAI Batch API tools for ablation studies
│   └── src/                 # Reservoir model, LLM operators, prompts, utilities
├── data/                    # Inflow, demand, and forecast input files
├── benchmarks/              # DP and MLP baselines
└── arnold_et_al_2026/       # Publication figure scripts
```

---

## How It Works

ResLLM couples a **physical reservoir simulation** with an **LLM-based decision agent**. The simulation runs at a daily time step, but the LLM makes **monthly allocation decisions** that determine what fraction of downstream water demand to release.

### Simulation Flow

```
Daily Loop (t = 1 to T)
│
├─> Read Inflow Q(t) from data
│
├─> Compute Reservoir Mass Balance:  S(t) = S(t-1) + Q(t) - R(t)
│
├─> If first day of month:
│   │
│   ├─> Query LLM Decision Agent
│   │   │
│   │   Inputs (current snapshot only):
│   │   • Current storage S(t)
│   │   • Cumulative inflow to date
│   │   • Remaining demand for water year
│   │   • Probabilistic forecasts (if provided)
│   │   • Operational constraints (from config)
│   │
│   │   Outputs:
│   │   • Allocation % (0-100)
│   │   • Justification text
│   │   • Concept importance rankings
│   │
│   └─> Set allocation for the month
│
├─> Compute Release:  R(t) = demand(t) × allocation%
│   (constrained by TOCS, max safe release, and capacity)
│
└─> Advance to next day: t = t + 1
    (loop back to top)

Note: The LLM agent is stateless — it receives only the current state
and forecast, not previous decisions or historical simulation outputs.
```

### The Allocation Decision

Each month, the LLM receives a prompt containing:

1. **System context** — Role as a reservoir operator, goal to minimize shortages, climate characteristics
2. **Operational constraints** — Max/min storage, average seasonal inflow and demand patterns (all derived from config and input data)
3. **Current observations** — Storage level, cumulative inflow, remaining demand
4. **Forecasts** (optional) — Probabilistic water year inflow projections (mean, 10th, 90th percentiles)
5. **Red herring** (optional) — Irrelevant information to test model focus

The agent always sees only the current state snapshot, not its previous decisions or the simulation's historical outputs.

The LLM responds with a structured JSON containing:
- **allocation_percent** — Fraction of demand to release (0–100%)
- **allocation_reasoning** — Natural language justification
- **allocation_concept_importance** — Rankings of which inputs influenced the decision

### Physics-Based Reservoir Model

The `Reservoir` class handles daily mass balance:

```
Storage(t) = Storage(t-1) + Inflow(t) - Release(t)
```

Release is constrained by:
- **Top of Conservation Storage (TOCS)** — Flood control curve that forces releases when storage exceeds seasonal limits
- **Max safe release** — Physical outlet capacity based on storage-elevation relationship
- **Spill** — Uncontrolled overflow when storage exceeds capacity

---

## License

See [LICENSE](LICENSE).

