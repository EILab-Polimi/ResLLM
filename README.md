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
│       ├── operator.py      # LLM operators (native provider APIs + logprobs)
│       ├── prompts.py       # All prompt templates and builder functions
│       ├── model_config.py  # Centralized provider config resolver
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

# Conda (recommended)
conda env create -f environment.yml
conda activate llm

# For cloud APIs, create a .env file
cp .env.example .env
# Add provider keys as needed: OPENAI_API_KEY, GOOGLE_API_KEY, XAI_API_KEY, MISTRAL_API_KEY
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

### Required Arguments

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
| `--temperature` | `None` | Sampling temperature override |
| `--tocs` | `fixed` | TOCS mode: `fixed` (seasonal curve) or `historical` (max of curve and observed) |
| `--wy-forecast-file` | `None` | Probabilistic forecast file (enables forecast context) |
| `--reasoning-effort` | `high` | Reasoning level for supported models |
| `--include-logprobs` | `None` | Request top-N token logprobs (`OpenAI`: 0–5, `Ollama`: 0–20) |
| `--include-red-herring` | `True` | Include irrelevant text to test focus |
| `--debug-response` | `False` | Save raw model responses for inspection |

### Example Commands

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

## Model Configuration

Provider-specific settings are resolved centrally in [resllm/src/model_config.py](resllm/src/model_config.py). CLI arguments (`--model-server`, `--model`, `--reasoning-effort`, `--temperature`, `--include-logprobs`) are captured as a `RunIntent` and resolved into a `ResolvedModelConfig` with validated kwargs, capability flags, and warnings.

Key behaviors:
- **Reasoning effort** accepts `none`, `minimal`, `low`, `medium`, `high`. The value is normalized per provider; `minimal` is mapped to `low` where unsupported. Providers that lack reasoning support (xAI, Mistral) emit a warning and ignore the flag.
- **Ollama cloud vs local**: Models ending in `-cloud` or `:cloud` receive effort strings (`low`/`medium`/`high`) for the `think` parameter; local models receive a boolean (`none` → `False`, all others including default → `True`).
- **Logprobs** (`--include-logprobs N`): Ollama local supports 0–20, OpenAI supports 0–5. Cloud Ollama models and all other providers ignore the flag with a warning. **OpenAI logprobs are mutually exclusive with reasoning** — logprobs use a separate Chat Completions operator that does not pass reasoning params, so reasoning models fall back to non-reasoning mode.

## Reasoning Traces

Reasoning traces (`model_reasoning`) capture the model's chain-of-thought when available:

| Provider | Method | Notes |
|----------|--------|-------|
| Ollama | Native `think` parameter | Streams thinking text; compatible with logprobs |
| OpenAI | Responses API summaries | Reasoning models only; non-reasoning prefixes (`gpt-4.1`, `gpt-4o`, `gpt-4-`) use Chat Completions instead |
| Google | `ThinkingConfig` | Extracted from response parts where `thought=True` |
| Baseten | Chat Completions `reasoning_effort` | Enables thinking for compatible models |
| xAI / Mistral | Not supported | — |

## Token Logprobs

Token-level log probabilities for the `allocation_percent` value are requested via `--include-logprobs N` and written to `<model>_r-<effort>_logprobs_output_n<N>.csv`.

| Provider | Range | Notes |
|----------|-------|-------|
| Ollama (local) | 0–20 | Compatible with reasoning; logprobs and thinking coexist in the same call |
| OpenAI | 0–5 | Uses a dedicated non-reasoning operator (see Logprobs note above) |

For Ollama local models, numeric values are often tokenized as individual digits (e.g., `85` → `["8", "5"]`). The output includes `n_value_tokens`, `value_tokens`, `joint_logprob`, and `joint_prob` columns that aggregate across all constituent tokens. The per-candidate columns (`top1_*`, …) report raw first-digit token probabilities only.

---

## Output Files

Simulations write to [resllm/output](resllm/output). Filenames encode the model name and reasoning effort:

```
<model>_r-<effort>_simulation_output_n<N>.csv
<model>_r-<effort>_decision_output_n<N>.csv
<model>_r-<effort>_logprobs_output_n<N>.csv     # only when --include-logprobs is set
```

Where `<model>` is the sanitized model ID (colons → hyphens, slashes → underscores) and `<effort>` is the `--reasoning-effort` value (default `high`). For example, `kimi-k2-thinking-cloud_r-high_simulation_output_n0.csv`.

### Simulation Output

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

### Decision Output

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

## Architecture Notes

### Operator Classes

- **`BaseReservoirOperator`** — Shared observation-setting, decision-recording, and `pop_logprobs_record()` logic.
- **`ReservoirAllocationOperator`** — Multi-provider operator using native APIs (OpenAI, Google, Ollama, xAI, Mistral). Supports reasoning traces and Ollama logprobs.
- **`OpenAIReservoirOperator`** — OpenAI Chat Completions operator specifically for logprobs extraction (OpenAI's Chat Completions API supports `top_logprobs` up to 5).
- **`build_operator()`** — Factory that selects the right operator class based on `ResolvedModelConfig`.

### Prompt Construction

All prompt text lives in [resllm/src/prompts.py](resllm/src/prompts.py) as template constants. Builder functions (`build_system_message`, `build_instructions`, `build_observation`) compose the final prompt from reservoir state and config. Ollama models receive an additional JSON instruction suffix since they use JSON mode rather than structured output schemas.

---

## License

See [LICENSE](LICENSE).

