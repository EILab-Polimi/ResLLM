# ResLLM

ResLLM is a Python library for simulating water reservoir management with large language models (LLMs).

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

By default, the agent sees only the current state snapshot, not its previous decisions or the simulation's historical outputs.

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

## Project Structure

```
ResLLM/
├── resllm/
│   ├── simulate.py          # Main entry point
│   ├── configs/             # Reservoir configuration YAML files
│   ├── output/              # Simulation results
│   ├── batch/               # OpenAI Batch API tools for ablation studies
│   └── src/
│       ├── reservoir.py     # Reservoir class (mass balance, TOCS, constraints)
│       ├── operator.py      # ReservoirAllocationOperator (LLM agent)
│       └── utils.py         # Unit conversions, date utilities
├── data/
│   ├── demand.txt           # 365-day demand series (TAF)
│   ├── folsom_daily.csv     # Historical inflow (date, inflow)
│   └── FOLC1_wy_hindcast.csv# Probabilistic forecasts (optional)
└── benchmarks/
    ├── dp/                  # Dynamic programming (DDP/SDP) baselines
    └── nn/                  # MLP neural network baseline
```

---

## Installation

```bash
git clone https://github.com/wyattarnold/ResLLM.git
cd ResLLM
pip install -r requirements.txt

# For cloud APIs, create a .env file
cp .env.example .env
# Add your keys: OPENAI_API_KEY, OLLAMA_API_KEY
```

---

## Usage

### Command Structure

```bash
python simulate.py \
  --model-server <SERVER> \
  --model <MODEL_NAME> \
  --config <CONFIG_FILE> \
  --start-year <YYYY> \
  --end-year <YYYY> \
  --starting-storage <TAF> \
  [optional flags]
```
| Argument | Description |
|----------|-------------|
| `--model-server` | LLM provider (e.g., `Ollama`, `OpenAI`, `Google`) |
| `--model` | Model identifier (e.g., `kimi-k2-thinking:cloud`, `o4-mini-2025-04-16`) |
| `--config` | Configuration YAML in `configs/` (e.g., `folsom.yml`) |
| `--start-year` | First water year (October–September) |
| `--end-year` | Last water year |
| `--starting-storage` | Initial storage (TAF) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--nsample` | `1` | Number of replicate simulations |
| `--temperature` | `1.0` | Sampling temperature |
| `--tocs` | `fixed` | TOCS mode: `fixed` (seasonal curve) or `historical` (max of curve and observed) |
| `--wy-forecast-file` | `None` | Probabilistic forecast file (enables forecast context) |
| `--reasoning-effort` | `high` | Reasoning level for supported models |
| `--include-double-check` | `False` | Ask model to verify its decision |
| `--include-num-history` | `0` | Number of past decisions to include in context |
| `--include-red-herring` | `True` | Include irrelevant text to test focus |
| `--debug-response` | `False` | Save raw model responses for inspection |


**Full historical period with forecasts (OpenAI):**
```bash
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

---

## Reasoning traces (CoT) handling

`model_reasoning` is populated differently per provider in [resllm/src/operator.py](resllm/src/operator.py):

- Ollama: Uses the native Ollama chat stream (when `think` is enabled) to capture the model’s thinking text. This stream is stored as `model_reasoning`.
- OpenAI: Uses the provider’s structured response; `model_reasoning` is set from the SDK’s `reasoning_content` when available.
- Google (Gemini): Uses thought summaries (not signatures). `includeThoughts` is enabled in [resllm/simulate.py](resllm/simulate.py), and `model_reasoning` is extracted from response `parts` where `thought=True`.

---

## Supported LLM Providers

| Provider | Server | Model | Notes |
|----------|-------------|---------------|-------|
| **Ollama** | `Ollama` | `kimi-k2-thinking:cloud` | Local or cloud; supports thinking trace capture |
| **OpenAI** | `OpenAI` | `o4-mini-2025-04-16` | Requires `OPENAI_API_KEY` |
| **Google** | `Google` | `gemini-2.5-pro` | Requires `GOOGLE_API_KEY` |
| **xAI** | `xAI` | `grok-4-0709` | Requires `XAI_API_KEY` |
| **Mistral** | `Mistral` | `mistral-large-2512` | Requires `MISTRAL_API_KEY` |

---

## Output Files

Simulations write to [resllm/output](resllm/output):

### Simulation Output (`<model>_simulation_output_n<N>.csv`)

Daily time series:

| Column | Description |
|--------|-------------|
| `date` | Calendar date |
| `wy` | Water year |
| `mowy` | Month of water year (1–12) |
| `dowy` | Day of water year (1–365) |
| `qt` | Inflow (TAF) |
| `st` | End-of-day storage (TAF) |
| `rt` | Release (TAF) |
| `dt` | Downstream demand (TAF) |
| `uu` | Target release = demand × allocation% |

### Decision Output (`<model>_decision_output_n<N>.csv`)

Monthly decisions:

| Column | Description |
|--------|-------------|
| `date`, `wy`, `mowy`, `dowy` | When the decision was made |
| `qwyaccum` | Cumulative water year inflow (TAF) |
| `d_wy_rem` | Remaining demand for the water year (TAF) |
| `st_1` | Storage at decision time (TAF) |
| `allocation_percent` | Decision (0–100%) |
| `allocation_justification` | LLM's reasoning |
| `model_reasoning` | Extended thinking trace (if available) |
| `observation` | Full prompt sent to LLM |
| Concept importance columns | Rankings for each input factor |

---

## Configuration Files

Configuration YAML files in [resllm/configs](resllm/configs) define reservoir characteristics. The config determines all operational constraints that the LLM sees in its prompt:

```yaml
config_name: "my_reservoir"

folsom_reservoir:  # key name used by the code
  operable_storage_max: 975   # TAF — upper storage limit
  operable_storage_min: 90    # TAF — dead pool / min operating level
  max_safe_release: 130000    # cfs — outlet capacity
  
  # Storage (TAF) to elevation (ft) — for level-based constraints
  sp_to_ep: [[storage_points], [elevation_points]]
  
  # Day of water year to TOCS (TAF) — flood control curve
  tp_to_tocs: [[day_points], [tocs_values]]
  
  # Storage (TAF) to max release (cfs) — release capacity curve
  sp_to_rp: [[storage_points], [release_points]]
```

To simulate a different reservoir, create a new config file and provide matching inflow/demand/forecast data.

---

## Input Data Format

Input data files live in the `data/` directory. The simulation reads these based on CLI flags (`--inflow-file`, `--demand-file`, `--wy-forecast-file`).

**Inflow file** (daily inflows in TAF):
```csv
date,inflow
1995-10-01,0.5
1995-10-02,0.6
...
```

**Demand file** (365 daily values in TAF, starting October 1):
```
2.5
2.5
2.6
...
```

**Forecast file** (optional, probabilistic water year inflow):
```csv
date,QCYFHM,QCYFH1,QCYFH9
1996-01-01,500,300,700
...
```
- `QCYFHM`: Mean forecast
- `QCYFH1`: 10th percentile
- `QCYFH9`: 90th percentile

The example data uses California's Folsom Reservoir, but you can substitute any reservoir by providing appropriate config and data files.

---

## License

See [LICENSE](LICENSE).

