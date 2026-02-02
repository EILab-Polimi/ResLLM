# Dynamic Programming (DP) Benchmark

This benchmark runs deterministic and stochastic dynamic programming (DDP/SDP) reservoir operating policies and replays them in a simulation. The core logic lives in [benchmarks/dp/dp](dp/), with [benchmarks/dp/main_dp.py](main_dp.py) wiring data, parameters, optimization, and simulation.

## Overview of the implementation

### main_dp.py (entry point)
`main_dp.py` builds a `sys_param` dictionary with two sections:
- `sys_param["simulation"]`: daily inflow series, daily demand, time step, and initial storage.
- `sys_param["algorithm"]`: grids, penalties, and Bellman settings.

Key steps:
1. **Load data**
   - Reads inflow CSV (`--inflow-file`) and demand from `demand.txt` in the same directory.
   - Builds the simulation date range from `--sim-start-date` to `--sim-end-date`.
   - Drops Feb 29 and replaces any non-positive inflows with the previous day's inflow.
2. **Initialize grids & parameters**
   - Storage grid `discr_s` (0–1005 TAF in 5 TAF steps).
   - Release grid `discr_u` (fine 0–10 TAF/day + coarse up to max safe release).
   - Inflow grid `discr_q` (fine 0–50, then coarse 55–155).
   - Deficit penalty exponent `beta` (default 2.0), discount `gamma` (1.0).
3. **Min/max release constraints**
   - Loads `minmaxRel.npz` (or recomputes with `--construct-new-rel-matrices`).
   - These matrices enforce operational min/max release limits by storage and inflow.
4. **SDP inflow statistics**
   - Loads lognormal parameters from `lognormal_qstats.npz` for the daily inflow distribution used by SDP.
   - For AR(1) mode, requires 3 columns: `[mu, sigma, rho]` where `rho` is the lag-1 autocorrelation.
5. **AR(1) autocorrelation settings** (optional, `--use-ar1`)
   - Expands the state space to include previous inflow: (s_t, log q_{t-1}).
   - Discretizes `log(q_prev)` with a non-uniform grid (finer at low flows, coarser at high flows).
   - Grid bounds are data-driven: covers mu +/- 3*sigma for 99.7% of the distribution.
6. **Optimization (optional)**
   - `--optimize` computes and saves a Bellman value function:
     - `BellmanDDP__beta{B}.txt` or `BellmanSDP__beta{B}.txt` in `./output/`.
     - For AR(1): `BellmanSDP_AR1__beta{B}.npy` (3D array saved as NumPy binary).
7. **Simulation (optional)**
   - `--simulate` loads a saved Bellman function and simulates daily operations using `sim_lake`.

### dp/ddp.py (deterministic DP)
DDP assumes a **known inflow trajectory**. For each day (backward in time) and each storage state, it solves a one-step Bellman update:
- `bellman_ddp` computes immediate deficit cost G = max(w_t - r, 0)^beta and adds the interpolated next cost.
- `opt_ddp` performs backward recursion over the full inflow sequence, producing a value matrix H(s, t).

### dp/sdp.py (stochastic DP)
SDP assumes **stochastic inflow** with a day-of-year lognormal distribution. Two modes are available:

#### Standard SDP (independent inflows)
- State space: (s_t) — storage only.
- `bellman_sdp` computes the expected cost over inflow bins using lognormal CDF weights.
- `opt_sdp` iterates until the value function converges (tolerance `tol`, max iterations `max_iter`).

#### SDP with AR(1) autocorrelation (`--use-ar1`)
- State space: (s_t, log q_{t-1}) — storage and previous log-inflow.
- Models inflow persistence using a lag-1 autoregressive model in log-space:
  ```
  log(q_t) | q_{t-1} ~ N(mu_t + rho_t * (sigma_t/sigma_{t-1}) * (log(q_{t-1}) - mu_{t-1}), sigma_t^2 * (1 - rho_t^2))
  ```
- `bellman_sdp_ar1_jit` — Numba JIT-compiled Bellman update with parallel execution over storage states.
- `opt_sdp_ar1` iterates until convergence, using the JIT-compiled function for speedup.
- Value function shape: `(n_s, n_log_q, T)` instead of `(n_s, T)`.

### dp/sim.py (simulation and utilities)
Core utilities and the simulation loop:
- Unit conversions (`taf_to_cfs`, `cfs_to_taf`), interpolation, and min/max release logic.
- `construct_rel_matrices` precomputes min/max releases across storage, inflow, and day.
- `mass_balance` integrates hourly within each day to update storage and actual release.
- `sim_lake` replays a policy using one-step Bellman updates and a tie-breaker that minimizes irrigation deficit.
  - For AR(1) mode, tracks previous inflow and uses `bellman_sdp_ar1` for decision extraction.

## CLI usage
```bash
# Build min/max release matrices (only needed if minmaxRel.npz is missing)
python main_dp.py --construct-new-rel-matrices

# Optimize a policy (DDP default)
python main_dp.py --optimize --algorithm ddp

# Optimize with SDP (independent inflows)
python main_dp.py --optimize --algorithm sdp

# Optimize with SDP-AR1 (autocorrelated inflows)
python main_dp.py --optimize --algorithm sdp --use-ar1

# Simulate a saved policy
python main_dp.py --simulate --algorithm ddp

# Full optimization and simulation with AR(1)
python main_dp.py --optimize --simulate --algorithm sdp --use-ar1 \
  --sim-start-date 1995-10-01 --sim-end-date 2016-09-30 \
  --deficit-penalty-beta 2.0 --inflow-file ../../data/folsom_daily.csv \
  --lognormal-qstat q_stat_obs_test --starting-storage 466.1

# Simulate using a policy directory (e.g., precomputed SDP policy)
python main_dp.py --simulate --algorithm sdp --sdp_policy_dir /path/to/policies
```

## CLI arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--algorithm` | `ddp` | Algorithm: `ddp` or `sdp` |
| `--optimize` | `False` | Compute and save Bellman value function |
| `--simulate` | `False` | Simulate saved policy |
| `--use-ar1` | `False` | Use AR(1) autocorrelated inflows (SDP only) |
| `--deficit-penalty-beta` | `2.0` | Deficit penalty exponent beta |
| `--sim-start-date` | `1995-10-01` | Simulation start date |
| `--sim-end-date` | `2016-09-30` | Simulation end date |
| `--inflow-file` | `folsom_daily-w2016.csv` | Inflow data CSV |
| `--lognormal-qstat` | `q_stat_obs_test` | Key in `lognormal_qstats.npz` for inflow stats |
| `--starting-storage` | `450` | Initial reservoir storage (TAF) |
| `--sdp_policy_dir` | `None` | Directory with pre-trained SDP policy |
| `--construct-new-rel-matrices` | `False` | Recompute min/max release matrices |

## Key inputs
- `lognormal_qstats.npz`: lognormal parameters for daily inflow distributions (SDP only).
  - Standard SDP: 2 columns `[mu, sigma]` per day.
  - AR(1) SDP: 3 columns `[mu, sigma, rho]` per day (required for `--use-ar1`).
- `minmaxRel.npz`: precomputed min/max release constraints.
- `demand.txt`: daily demand series (aligned to 365-day year; leap day removed in code).
- Inflow CSV (default `folsom_daily-w2016.csv`): requires `date` and `inflow` columns.

## Outputs
All outputs are written to [benchmarks/dp/output](output/):
- `BellmanDDP__beta{B}.txt` — DDP value function (2D: n_s x N+1)
- `BellmanSDP__beta{B}.txt` — SDP value function (2D: n_s x T)
- `BellmanSDP_AR1__beta{B}.npy` — SDP-AR1 value function (3D: n_s x n_log_q x T)
- `{DDP|SDP}_beta{B}_sim.csv` with columns: `s,u,r,G`

## Dependencies
- `numpy`, `pandas`, `scipy`
- `numba` (for JIT-compiled AR(1) optimization)

Install numba if using AR(1):
```bash
pip install numba
# or
mamba install numba
```

## Notes
- The objective is a **deficit penalty**: G_t = max(d_t - r_t, 0)^beta averaged over time.
- Both algorithms enforce operational min/max release constraints from `minmaxRel.npz`.
- The simulation removes Feb 29; inputs must align to a 365-day water year.
- AR(1) expands the state space significantly; optimization is slower than standard SDP even with JIT compilation, but captures inflow persistence which can improve policy performance during droughts.
