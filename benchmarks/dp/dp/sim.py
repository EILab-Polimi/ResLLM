import numpy as np


def taf_to_cfs(x):
    """
    x in TAF/day  →  y in m3/s
    """
    return x * 1000.0 / 86400.0 * 43560.0


def cfs_to_taf(x):
    """
    x in m3/s  →  y in TAF/day
    """
    return x * 2.29568411e-5 * 86400.0 / 1000.0


def storage_to_level(s, sys_param):
    """
    s [m3] → h [cm]
    """
    A = sys_param["simulation"]["A"]
    h0 = sys_param["simulation"]["h0"]
    return s / A + h0


def level_to_storage(h, sys_param):
    """
    h [cm] → s [m3]
    """
    A = sys_param["simulation"]["A"]
    h0 = sys_param["simulation"]["h0"]
    return A * (h - h0)


def interp_lin_scalar(X, Y, x):
    """
    linear interpolation of points (X, Y) at x (scalar)
    """
    if x <= X[0]:
        return Y[0]
    if x >= X[-1]:
        return Y[-1]
    return np.interp(x, X, Y)


def flood_buffer(d):
    tp = np.array([1, 50, 151, 200, 243, 365], dtype=float)
    sp = np.array([975, 400, 400, 750, 975, 975], dtype=float)
    return interp_lin_scalar(tp, sp, d)


def immediate_costs(ht, rt, sys_param):
    """
    ht [cm], rt [m3/s] → (g_flo [cm], g_irr [m3/s])
    """
    hFLO = sys_param["simulation"]["hFLO"]
    w = sys_param["simulation"]["w"]
    g_flo = max(ht - hFLO, 0.0) * 100.0
    g_irr = max(w - rt, 0.0)
    return g_flo, g_irr


def min_release(s, q, dt, sys_param):
    """
    time‐varying minimal release
    """
    sF = flood_buffer(dt)
    wt = sys_param["simulation"]["w"][dt]
    if s < 90.0:
        return 0.0
    elif s < sF:
        return sys_param["simulation"]["EnvF"]
    elif s < 975.0:
        return max(0.2 * (q + s - sF), 0.5 * wt)
    else:
        return max_release(s)


def max_release(s):
    xs = np.array([0, 90, 100, 400, 600, 1000], dtype=float)
    xr = cfs_to_taf(np.array([0, 0, 35000, 40000, 115000, 135000], dtype=float))
    return interp_lin_scalar(xs, xr, s)


def mass_balance(s, u, q, dt, sys_param):
    """
    hourly integration of storage‐release dynamics
    returns (s1, r1)
    """
    HH = 24
    delta = sys_param["simulation"]["delta"] / HH
    s_ = np.empty(HH + 1)
    r_ = np.empty(HH + 1)
    s_[0] = s
    for i in range(HH):
        qm = min_release(s_[i], q, dt, sys_param)
        qM = max_release(s_[i])
        # actual release capped between qm and qM
        r_[i + 1] = np.clip(u, qm, qM)
        s_[i + 1] = s_[i] + delta * (q - r_[i + 1])
    s1 = s_[-1]
    r1 = np.mean(r_[1:])
    return s1, r1


def extractor_ref(idx_U, discr_u, w):
    """
    pick single decision u from candidates idx_U
    """
    u_vals = discr_u[idx_U]
    defs = u_vals - w
    best = np.argmin(np.abs(defs))
    return u_vals[best], idx_U[0][best]


def construct_rel_matrices(sys_param):
    """
    builds vv(min) and VV(max) release matrices
    """
    discr_s = sys_param["algorithm"]["discr_s"]
    discr_q = sys_param["algorithm"]["discr_q"]
    T = 365

    # build a grid of (s_i, q_j, t)
    sv, qv, tv = np.meshgrid(discr_s, discr_q, np.arange(0, T), indexing="ij")
    # vectorize the mass_balance call for the min‐release (u=0)
    vec_min = np.vectorize(
        lambda si, qj, dt: mass_balance(si, 0.0, qj, dt, sys_param)[1], otypes=[float]
    )
    vv = vec_min(sv, qv, tv)

    # build a grid of (s_i, q_j) for the max‐release (u = discr_q[-1], at final dt=T)
    sv2, qv2 = np.meshgrid(discr_s, discr_q, indexing="ij")
    vec_max = np.vectorize(
        lambda si, qj: mass_balance(si, discr_q[-1], qj, T - 1, sys_param)[1],
        otypes=[float],
    )
    VV = vec_max(sv2, qv2)

    return vv, VV


def sim_lake(q, s0, policy, sys_param):
    """
    translate simLake.m
    q    : inflow vector [m3/s]
    s0   : initial storage [m3]
    policy: has H matrix and name, etc.
    returns J (performance), time series s, u, r, G
    """
    Hlen = len(q)
    q_sim = np.concatenate(([np.nan], q))
    H = Hlen
    T = sys_param["algorithm"]["T"]

    h = np.full(H + 1, np.nan)
    s = np.full(H + 1, np.nan)
    r = np.full(H + 1, np.nan)
    u = np.full(H + 1, np.nan)

    s[0] = s0
    for t in range(H):
        doy = t % T
        name = sys_param["algorithm"]["name"]

        if name == "rand":
            u_t = np.random.uniform(
                sys_param["simulation"]["r_min"], sys_param["simulation"]["r_max"]
            )

        elif name == "ddp":
            # -- load DDP algorithm
            from dp.ddp import bellman_ddp

            # grab discretizations & precomputed min/max tables
            discr_s = sys_param["algorithm"]["discr_s"]  # (n_s,)
            discr_q = sys_param["algorithm"]["discr_q"]  # (n_q,)
            discr_u = sys_param["algorithm"]["discr_u"]  # (n_u,)
            min_rel = sys_param["algorithm"]["min_rel"]  # (n_s, n_q, T)
            max_rel = sys_param["algorithm"]["max_rel"]  # (n_s, n_q)

            # current flood‑deficit weight
            wt = sys_param["simulation"]["w"][doy]
            sys_param["simulation"]["wt"] = wt

            # find index of nearest inflow bin
            idx_q = np.argmin(np.abs(discr_q - q_sim[t + 1]))

            # interpolate min/max release for this storage & inflow
            vv_vals = min_rel[:, idx_q, doy]
            VV_vals = max_rel[:, idx_q]
            sys_param["simulation"]["vv"] = interp_lin_scalar(discr_s, vv_vals, s[t])
            sys_param["simulation"]["VV"] = interp_lin_scalar(discr_s, VV_vals, s[t])

            # one‐step Bellman to get optimal decision indices
            H_next = policy["H"][:, t + 1]
            H_new, idx_u_list = bellman_ddp(
                H_next, np.array([s[t]]), q_sim[t + 1], sys_param
            )
            # print(f"t: {t}")
            # print(f"idx_u_list: {idx_u_list}")
            # print(f"_: {_}")

            # pick a single u via irrigation‐deficit tie‐breaker
            u_t, _ = extractor_ref(idx_u_list, discr_u, wt)

        elif name == "sdp":
            # -- load SDP algorithm
            from dp.sdp import bellman_sdp

            # grab discretizations & precomputed min/max tables
            discr_s = sys_param["algorithm"]["discr_s"]  # (n_s,)
            discr_q = sys_param["algorithm"]["discr_q"]  # (n_q,)
            discr_u = sys_param["algorithm"]["discr_u"]  # (n_u,)
            min_rel = sys_param["algorithm"]["min_rel"]  # (n_s, n_q, T)
            max_rel = sys_param["algorithm"]["max_rel"]  # (n_s, n_q)

            # current demand
            wt = sys_param["simulation"]["w"][doy]
            sys_param["simulation"]["wt"] = wt

            # set disturbance
            sys_param["algorithm"]["stat_t"] = sys_param["algorithm"]["q_stat"][doy, :]

            # find index of nearest inflow bin
            idx_q = np.argmin(np.abs(discr_q - q_sim[t + 1]))
            # interpolate min/max release for this storage & inflow
            vv_vals = min_rel[:, idx_q, doy]
            VV_vals = max_rel[:, idx_q]
            sys_param["simulation"]["vv"] = interp_lin_scalar(discr_s, vv_vals, s[t])
            sys_param["simulation"]["VV"] = interp_lin_scalar(discr_s, VV_vals, s[t])

            # one‐step Bellman to get optimal decision indices
            H_next = policy["H"][:, doy]
            H_new, idx_u_list = bellman_sdp(H_next, s[t], sys_param)

            # pick a single u via irrigation‐deficit tie‐breaker
            u_t, _ = extractor_ref(idx_u_list, discr_u, wt)

        else:
            u_t = np.nan

        u[t] = u_t
        s[t + 1], r[t + 1] = mass_balance(s[t], u[t], q_sim[t + 1], doy, sys_param)
        # h[t+1] = storage_to_level(s[t+1])

    # compute objective
    Ny = H // T
    D = np.tile(sys_param["simulation"]["w"], Ny)
    G = np.maximum(D - r[1:], 0.0) ** sys_param["algorithm"]["beta"]
    J = np.mean(G)
    return J, s, u, r, G
