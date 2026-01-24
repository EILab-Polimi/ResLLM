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
  - Drops Feb 29 and replaces any non‑positive inflows with the previous day’s inflow.
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
5. **Optimization (optional)**
  - `--optimize` computes and saves a Bellman value function:
    - `BellmanDDP__beta{B}.txt` or `BellmanSDP__beta{B}.txt` in `./output/`.
6. **Simulation (optional)**
  - `--simulate` loads a saved Bellman function and simulates daily operations using `sim_lake`.

### dp/ddp.py (deterministic DP)
DDP assumes a **known inflow trajectory**. For each day (backward in time) and each storage state, it solves a one‑step Bellman update:
- `bellman_ddp` computes immediate deficit cost $G = \max(w_t - r, 0)^{\beta}$ and adds the interpolated next cost.
- `opt_ddp` performs backward recursion over the full inflow sequence, producing a value matrix $H(s, t)$.

### dp/sdp.py (stochastic DP)
SDP assumes **stochastic inflow** with a day‑of‑year lognormal distribution. For each day and storage:
- `bellman_sdp` computes the expected cost over inflow bins using lognormal CDF weights.
- `opt_sdp` iterates until the value function converges (tolerance `tol`, max iterations `max_iter`).

### dp/sim.py (simulation and utilities)
Core utilities and the simulation loop:
- Unit conversions (`taf_to_cfs`, `cfs_to_taf`), interpolation, and min/max release logic.
- `construct_rel_matrices` precomputes min/max releases across storage, inflow, and day.
- `mass_balance` integrates hourly within each day to update storage and actual release.
- `sim_lake` replays a policy using one‑step Bellman updates and a tie‑breaker that minimizes irrigation deficit.

## CLI usage
```bash
# Build min/max release matrices (only needed if minmaxRel.npz is missing)
python main_dp.py --construct-new-rel-matrices

# Optimize a policy (DDP default)
python main_dp.py --optimize --algorithm ddp

# Optimize with SDP
python main_dp.py --optimize --algorithm sdp

# Simulate a saved policy
python main_dp.py --simulate --algorithm ddp

# Simulate using a policy directory (e.g., precomputed SDP policy)
python main_dp.py --simulate --algorithm sdp --sdp_policy_dir /path/to/policies
```

## Key inputs
- `lognormal_qstats.npz`: lognormal parameters for daily inflow distributions (SDP only).
- `minmaxRel.npz`: precomputed min/max release constraints.
- `demand.txt`: daily demand series (aligned to 365‑day year; leap day removed in code).
- Inflow CSV (default `folsom_daily-w2016.csv`): requires `date` and `inflow` columns.

## Outputs
All outputs are written to [benchmarks/dp/output](output/):
- `BellmanDDP__beta{B}.txt` or `BellmanSDP__beta{B}.txt`
- `{DDP|SDP}_beta{B}_sim.csv` with columns: `s,u,r,G`

## Notes
- The objective is a **deficit penalty**: $G_t = \max(d_t - r_t, 0)^{\beta}$ averaged over time.
- Both algorithms enforce operational min/max release constraints from `minmaxRel.npz`.
- The simulation removes Feb 29; inputs must align to a 365‑day water year.
