import numpy as np
from scipy.stats import lognorm


def bellman_sdp(H_prev, s_curr, sys_param):
    """
    Python version of Bellman_ddp.m

    Inputs:
      H_prev : np.ndarray, shape (n_states,)
      s_curr : np.ndarray, shape (n_states,) // discrete state grid

    Returns:
      H_new  : np.ndarray, shape (n_states,) // updated cost‐to‐go
      idx_u  : list of np.ndarray            // optimal control indices per state
    """
    # fetch grids and parameters
    discr_s = sys_param["algorithm"]["discr_s"]  # shape (n_states,)
    discr_u = sys_param["algorithm"]["discr_u"]  # shape (n_u,)
    discr_q = sys_param["algorithm"]["discr_q"]  # shape (n_q,)
    gamma = sys_param["algorithm"]["gamma"]
    delta = sys_param["simulation"]["delta"]
    wt = sys_param["simulation"]["wt"]

    vv_vec = np.atleast_1d(sys_param["simulation"]["vv"])  # (n_q,)
    VV_vec = np.atleast_1d(sys_param["simulation"]["VV"])  # (n_q,)

    n_u = len(discr_u)
    n_q = len(discr_q)

    vv_mat = np.broadcast_to(vv_vec[:, None], (n_q, n_u))
    VV_mat = np.broadcast_to(VV_vec[:, None], (n_q, n_u))
    uu_mat = np.broadcast_to(discr_u[None, :], (n_q, n_u))

    # calculate actual release constrained by min/max release
    R = np.minimum(VV_mat, np.maximum(vv_mat, uu_mat))

    # next state for each (s, q)
    qq_mat = np.broadcast_to(discr_q[:, None], (n_q, n_u))
    s_next = s_curr + delta * (qq_mat - R)  # (n_q, n_u)

    # immediate cost G[s,i]
    G = np.maximum(wt - R, 0.0) ** sys_param["algorithm"]["beta"]

    # interpolate previous cost‐to‐go at each s_next
    H_interp = np.interp(s_next.ravel(), discr_s, H_prev)
    H_future = np.reshape(H_interp, (n_q, n_u))

    # mean & std of log(disturbance)
    mi_q, sigma_q = sys_param["algorithm"]["stat_t"]
    # compute resolution of Bellman value function --
    # compute the probability of occourence of inflow that falls within the
    # each bin of descritized inflow level
    cdf_q = lognorm(s=sigma_q, scale=np.exp(mi_q)).cdf(discr_q)
    p_q = np.diff(cdf_q)  # length n_q-1
    p0 = 1.0 - p_q.sum()
    p_diff = np.concatenate(([p0], p_q))  # (n_q,)

    # expected cost Q(u) = sum_q [ G + γ H_future ] * p_diff
    Q_mat = G + gamma * H_future  # (n_q, n_u)
    Q_exp = (Q_mat.T * p_diff).sum(axis=1)  # (n_u,)

    # Bellman: pick minimal expectation
    H_new = Q_exp.min()
    eps = np.finfo(float).eps
    idx_u = np.where(Q_exp <= (H_new + eps))

    return H_new, idx_u


def opt_sdp(tol, max_iter, sys_param):
    """
    Python port of opt_sdp.m (Stochastic DP)

    Args:
        tol (float): tolerance for convergence
        max_iter (int): maximum number of Bellman iterations

    Returns:
        H (np.ndarray): optimal value function, shape (n_s, T)
    """

    # pull params
    algo = sys_param["algorithm"]
    discr_s = algo["discr_s"]  # (n_s,)
    min_rel = algo["min_rel"]  # (n_s, n_q, T)
    max_rel = algo["max_rel"]  # (n_s, n_q)
    q_stat = algo["q_stat"]  # (T,)
    T = algo["T"]
    n_s = len(discr_s)

    # initialize Bellman value matrix
    H = np.zeros((n_s, T))
    diff_H = np.inf
    count = 0

    # backward recursion
    while diff_H >= tol and count < max_iter:

        # copy previous value function
        H_old = H.copy()

        # backward recursion
        for t in range(T - 1, -1, -1):

            # set demand and disturbance
            sys_param["simulation"]["wt"] = sys_param["simulation"]["w"][t]
            sys_param["algorithm"]["stat_t"] = q_stat[t, :]

            # next‐stage values (wrap around)
            H_next = H[:, (t + 1) % T]

            # update each state
            for i in range(n_s):

                # set min/max release distributions for state i at time t
                sys_param["simulation"]["vv"] = min_rel[i, :, t]
                sys_param["simulation"]["VV"] = max_rel[i, :]

                # Bellman update for state i
                H[i, t], _ = bellman_sdp(H_next, discr_s[i], sys_param)

        # check convergence
        diff_H = np.max(np.abs(H - H_old))
        count += 1
        print(f"Iteration {count}: max diff = {diff_H}")

    return H
