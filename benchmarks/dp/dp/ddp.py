import numpy as np


def bellman_ddp(H_prev, s_curr, q_curr, sys_param):
    """
    Python version of Bellman_ddp.m

    Inputs:
      H_prev : np.ndarray, shape (n_states,)
      s_curr : np.ndarray, shape (n_states,) // discrete state grid
      q_curr : float                         // current disturbance

    Returns:
      H_new  : np.ndarray, shape (n_states,) // updated cost‐to‐go
      idx_u  : list of np.ndarray            // optimal control indices per state
    """
    # fetch grids and parameters
    discr_s = sys_param["algorithm"]["discr_s"]  # shape (n_states,)
    discr_u = sys_param["algorithm"]["discr_u"]  # shape (n_u,)
    delta = sys_param["simulation"]["delta"]
    wt = sys_param["simulation"]["wt"]

    n_u = len(discr_u)

    VV = np.tile(sys_param["simulation"]["VV"], n_u)
    vv = np.tile(sys_param["simulation"]["vv"], n_u)

    # build release matrix R[s,i] = min( VV[s], max(vv[s], discr_u[i]) )
    R = np.minimum(VV, np.maximum(vv, discr_u))

    # next state for each (s,i)
    s_next = s_curr + delta * (q_curr - R)

    # immediate cost G[s,i]
    G = np.maximum(wt - R, 0.0) ** sys_param["algorithm"]["beta"]

    # interpolate previous cost‐to‐go at each s_next
    H_interp = np.interp(s_next, discr_s, H_prev)

    # total Q‐value
    Q = G + H_interp

    # Bellman update
    H_new = np.min(Q)

    # indices of optimal u per state (within machine‐eps tolerance)
    eps = np.finfo(float).eps
    idx_u = np.where(Q <= (H_new + eps))

    return H_new, idx_u


def opt_ddp(disturbance, sys_param):
    """
    Python port of opt_ddp.m (Deterministic DP over a known disturbance path).

    disturbance : array-like, length N

    Returns:
      H : np.ndarray, shape (n_s, N+1)
    """
    # -- Load interpolation function --
    from dp.sim import interp_lin_scalar

    # pull params
    algo = sys_param["algorithm"]
    discr_s = algo["discr_s"]  # (n_s,)
    discr_q = algo["discr_q"]  # (n_q,)
    min_rel = algo["min_rel"]  # (n_s, n_q, T)
    max_rel = algo["max_rel"]  # (n_s, n_q)
    Hend = algo["Hend"]  # (n_s,)
    T = algo["T"]
    N = len(disturbance)
    n_s = len(discr_s)

    # initialize Bellman value matrix
    H = np.zeros((n_s, N + 1))
    H[:, N] = Hend

    # backward recursion
    for t in range(N - 1, -1, -1):
        d = disturbance[t]
        doy = t % T  # 0..T-1
        sys_param["simulation"]["wt"] = sys_param["simulation"]["w"][doy]

        H_next = H[:, t + 1]
        H_cur = np.empty(n_s)

        for i in range(n_s):
            # update min/max release for state i at day doy
            vv_i = interp_lin_scalar(discr_q, min_rel[i, :, doy], d)
            VV_i = interp_lin_scalar(discr_q, max_rel[i, :], d)
            sys_param["simulation"]["vv"] = vv_i
            sys_param["simulation"]["VV"] = VV_i

            # run one‐state Bellman update
            H_i, _ = bellman_ddp(H_next, discr_s[i : i + 1], d, sys_param)
            H_cur[i] = H_i

        H[:, t] = H_cur

    return H
