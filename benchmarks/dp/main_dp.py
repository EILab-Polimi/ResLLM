# %%
import os
import numpy as np
import pandas as pd
from dp.ddp import opt_ddp
from dp.sdp import opt_sdp
from dp.sim import sim_lake, cfs_to_taf, construct_rel_matrices
import argparse

# %%— parameters from CLI
def build_parser():
    parser = argparse.ArgumentParser(description="Run DDP/SDP lake simulation")

    parser.add_argument(
        "--construct-new-rel-matrices",
        action="store_true",
        default=False,
        help="Recompute and save new relation matrices",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=False,
        help="Compute Bellman function and save to file",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Simulate saved Bellman function",
    )
    parser.add_argument(
        "--sdp_policy_dir",
        type=str,
        default=None,
        help="Directory with trained SDP policy",
    )
    parser.add_argument(
        "--algorithm",
        choices=["ddp", "sdp"],
        default="ddp",
        help="Algorithm to run: ddp or sdp",
    )
    parser.add_argument(
        "--sim-start-date",
        default="1995-10-01",
        help="Simulation start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--sim-end-date",
        default="2016-09-30",
        help="Simulation end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--deficit-penalty-beta",
        type=float,
        default=2.0,
        help="Deficit penalty for the algorithm",
    )
    parser.add_argument(
        "--inflow-file",
        type=str,
        default="folsom_daily-w2016.csv",
        help="Inflow data file",
    )
    parser.add_argument(
        "--lognormal-qstat",
        type=str,
        default="q_stat_obs_test",
        help="qstat name for lognormal distribution (SDP only)",
    )
    parser.add_argument(
        "--starting-storage",
        type=float,
        default=450,
        help="Initial storage in the reservoir",
    )

    return parser


# %% main_ddp.py
def main(args):
    # %%— parse arguments
    construct_new_rel_matrices = args.construct_new_rel_matrices
    optimize = args.optimize
    simulate = args.simulate
    deficit_penalty_beta = args.deficit_penalty_beta
    algorithm = args.algorithm
    sim_start_date = args.sim_start_date
    sim_end_date = args.sim_end_date
    inflow_file = args.inflow_file
    starting_storage = args.starting_storage
    lognormal_qstat = args.lognormal_qstat
    sdp_policy_dir = args.sdp_policy_dir

    # %%— paths
    inflow_file = os.path.abspath(inflow_file)
    data_dir = os.path.dirname(inflow_file)
    out_dir = "./output/"
    os.makedirs(out_dir, exist_ok=True)

    # %%— load raw series
    data = pd.read_csv(inflow_file)
    data["date"] = pd.to_datetime(data["date"])
    qq = data["inflow"].values
    w = np.loadtxt(os.path.join(data_dir, "demand.txt"))

    # %%— build date index & remove Feb‑29
    dn_sim = pd.date_range(sim_start_date, sim_end_date, freq="D")
    # sub‐series of qq and ev
    mask_dn = (data["date"] >= dn_sim[0]) & (data["date"] <= dn_sim[-1])
    qqs = qq[mask_dn]
    # drop Feb‑29
    feb29 = (dn_sim.month == 2) & (dn_sim.day == 29)
    q_clean = qqs[~feb29]
    # avoid zeros by replacing with previous value
    while np.where(q_clean <= 0)[0].size > 0:
        idx_lt0 = np.where(q_clean <= 0)[0]
        q_clean[idx_lt0] = q_clean[idx_lt0 - 1]

    # %%— fill sys_param.simulation
    sys_param = {"simulation": {}, "algorithm": {}}
    sim = sys_param["simulation"]
    sim["q"] = q_clean
    # initial storage
    sim["s_in"] = starting_storage  # sobs[data["date"] == dn_sim[0]][0]
    # remove row 152 (leap day) from w
    w_clean = np.hstack([w[:151], w[152:]])
    sim["w"] = w_clean
    sim["EnvF"] = 0.0
    sim["delta"] = 1.0  # daily step

    # %%— load DDP grids & min/max tables
    algo = sys_param["algorithm"]
    algo["name"] = algorithm
    algo["Hend"] = 0
    algo["T"] = 365
    algo["gamma"] = 1.0  # discount factor
    algo["beta"] = deficit_penalty_beta  # deficit penalty
    algo["discr_s"] = np.arange(0, 1006, 5)
    algo["discr_u"] = np.hstack(
        [
            np.arange(0, 7.01, 0.01),
            np.arange(7, 9.4, 0.01),
            np.arange(10, cfs_to_taf(115000), 20),
        ]
    )
    algo["discr_q"] = np.hstack([np.arange(0, 50.1, 0.1), np.arange(55, 155.1, 10)])
    if construct_new_rel_matrices == True:
        vv, VV = construct_rel_matrices(sys_param)
        np.savez("minmaxRel.npz", vv=vv, VV=VV)
    else:
        npzfile = np.load("minmaxRel.npz")
        vv, VV = npzfile["vv"], npzfile["VV"]
    algo["min_rel"] = vv
    algo["max_rel"] = VV

    # %%- estimate cyclostationary pdf assuming log-normal distribution
    # T = sys_param["algorithm"]["T"]
    # q = np.asarray(sys_param["simulation"]["q"])
    # Ny = q.size // T
    # Q = q.reshape(Ny, T).T
    # q_stat = np.empty((T, 2))

    # for i in range(T):
    #     qi = Q[i, :]
    #     log_qi = np.log(qi)
    #     q_mean = np.mean(log_qi)
    #     q_std = np.std(log_qi)
    #     q_stat[i, :] = [q_mean, q_std]

    npzfile = np.load("lognormal_qstats.npz")
    sys_param["algorithm"]["q_stat"] = npzfile[lognormal_qstat]

    # %%— run algorithm
    if optimize:
        print(f"Running {algorithm.upper()} optimization...")
        if algorithm == "sdp":
            # run SDP
            Hopt = opt_sdp(tol=0.1, max_iter=15, sys_param=sys_param)
            np.savetxt(
                os.path.join(
                    out_dir, f"BellmanSDP__beta{int(deficit_penalty_beta)}.txt"
                ),
                Hopt,
            )
        elif algorithm == "ddp":
            # run DDP
            Hopt = opt_ddp(disturbance=sim["q"], sys_param=sys_param)
            np.savetxt(
                os.path.join(
                    out_dir, f"BellmanDDP__beta{int(deficit_penalty_beta)}.txt"
                ),
                Hopt,
            )

    # %%— load back & simulate
    if simulate:
        print(f"Running {algorithm.upper()} simulation...")
        policy = {
            "H": np.loadtxt(
                os.path.join(
                    sdp_policy_dir if sdp_policy_dir else out_dir,
                    "Bellman{}__beta{}.txt".format(
                        algorithm.upper(), int(deficit_penalty_beta)
                    ),
                )
            )
        }
        J, s, u, r, G = sim_lake(sim["q"], sim["s_in"], policy, sys_param)
        print(J)

        # %%— save simulation
        np.savetxt(
            os.path.join(
                out_dir,
                "{}_beta{}_sim.csv".format(
                    algorithm.upper(), int(deficit_penalty_beta)
                ),
            ),
            pd.DataFrame(
                {"s": s, "u": u, "r": r, "G": np.concat((np.zeros(1), G), axis=0)}
            ),
            delimiter=",",
            header="s,u,r,G",
            fmt="%.16f",
        )


# %% main
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
