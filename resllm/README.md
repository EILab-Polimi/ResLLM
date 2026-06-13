# ResLLM — Technical Reference

Full CLI reference, model configuration, output formats, and architecture details for [ResLLM](../README.md).

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
| `--model-server` | LLM provider: `Ollama` or `OpenAI` |
| `--model` | Model identifier (e.g., `gpt-oss:120b-cloud`, `o4-mini-2025-04-16`) |
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

Provider-specific settings are resolved centrally in [src/model_config.py](src/model_config.py). CLI arguments (`--model-server`, `--model`, `--reasoning-effort`, `--temperature`) are captured as a `RunIntent` and resolved into a `ResolvedModelConfig` with validated kwargs and warnings.

Key behaviors:
- **Reasoning effort** accepts `none`, `minimal`, `low`, `medium`, `high`. The value is normalized per provider; `minimal` is mapped to `low` where unsupported. OpenAI non-reasoning model families (`gpt-4.1`, `gpt-4o`, `gpt-4-`) emit a warning and run via Chat Completions without reasoning.
- **Ollama cloud vs local**: Models ending in `-cloud` or `:cloud` receive effort strings (`low`/`medium`/`high`) for the `think` parameter; local models receive a boolean (`none` → `False`, all others including default → `True`).

---

## Reasoning Traces

Reasoning traces (`model_reasoning`) capture the model's chain-of-thought when available:

| Provider | Method | Notes |
|----------|--------|-------|
| Ollama | Native `think` parameter | Streams thinking text chunk-by-chunk |
| OpenAI | Responses API summaries | Reasoning models only; non-reasoning prefixes (`gpt-4.1`, `gpt-4o`, `gpt-4-`) use Chat Completions instead |

---

## Output Files

Simulations write to [output/](output/). Filenames encode the model name and reasoning effort:

```
<model>_r-<effort>_simulation_output_n<N>.csv
<model>_r-<effort>_decision_output_n<N>.csv
```

Where `<model>` is the sanitized model ID (colons → hyphens, slashes → underscores) and `<effort>` is the `--reasoning-effort` value (default `high`). For example, `gpt-oss-120b-cloud_r-high_simulation_output_n0.csv`.

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

Configuration YAML files in [configs/](configs/) define reservoir characteristics. The config determines all operational constraints that the LLM sees in its prompt:

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

Input data files live in the `data/` directory (repo root). The simulation reads these based on CLI flags (`--inflow-file`, `--demand-file`, `--wy-forecast-file`).

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

## Architecture

### Source Files

```
src/
├── reservoir.py     # Reservoir class (mass balance, TOCS, constraints)
├── operator.py      # LLM operators (OpenAI + Ollama native APIs)
├── prompts.py       # All prompt templates and builder functions
├── model_config.py  # Centralized provider config resolver
└── utils.py         # Unit conversions, date utilities
```

### Operator Classes

- **`BaseReservoirOperator`** — Shared observation-setting and decision-recording logic.
- **`ReservoirAllocationOperator`** — Operator using native provider APIs (OpenAI and Ollama). Supports reasoning traces. OpenAI reasoning models use the Responses API; non-reasoning models use Chat Completions; Ollama uses the streaming `generate` endpoint.
- **`build_operator()`** — Factory that constructs the operator from a `ResolvedModelConfig`.

### Prompt Construction

All prompt text lives in [src/prompts.py](src/prompts.py) as template constants. Builder functions (`build_system_message`, `build_instructions`, `build_observation`) compose the final prompt from reservoir state and config. Ollama models receive an additional JSON instruction suffix since they use JSON mode rather than structured output schemas.
