import os
# Suppress OpenMP deprecation warning
os.environ.setdefault("KMP_WARNINGS", "0")

import numpy as np
from scipy.stats import lognorm, norm
from scipy.interpolate import RegularGridInterpolator
from numba import njit, prange
import math


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
    # q_stat may have 2 or 3 columns; only need mu and sigma here
    stat_t = sys_param["algorithm"]["stat_t"]
    mi_q, sigma_q = stat_t[0], stat_t[1]
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
    diff_H_prev = np.inf
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
        diff_H_prev = diff_H
        diff_H = np.max(np.abs(H - H_old))
        count += 1
        print(f"Iteration {count}: max diff = {diff_H}")
        
        # stop if difference hasn't changed (stuck at numerical precision limit)
        if count > 1 and abs(diff_H - diff_H_prev) < 1e-10:
            print(f"  Convergence stalled (diff unchanged). Stopping.")
            break

    return H


def conditional_lognormal_params(log_q_prev, mu_t, sigma_t, mu_t_prev, sigma_t_prev, rho):
    """
    Compute conditional mean and std of log(q_t) given log(q_{t-1})
    using AR(1) model in log-space.

    Returns:
        cond_mean: conditional mean of log(q_t)
        cond_std: conditional std of log(q_t)
    """
    cond_mean = mu_t + rho * (sigma_t / sigma_t_prev) * (log_q_prev - mu_t_prev)
    cond_std = sigma_t * np.sqrt(max(1 - rho**2, 0.01))  # floor to avoid numerical issues
    return cond_mean, cond_std


# ============================================================================
# Numba JIT-compiled functions for SDP-AR1
# ============================================================================

@njit(cache=True)
def norm_cdf(x):
    """Standard normal CDF using error function approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(cache=True)
def norm_cdf_scaled(x, loc, scale):
    """Normal CDF with location and scale."""
    return norm_cdf((x - loc) / scale)


@njit(cache=True)
def bilinear_interp(x, y, xgrid, ygrid, Z):
    """
    Bilinear interpolation on a regular grid.
    
    x, y: query point
    xgrid: 1D array of x grid points (sorted)
    ygrid: 1D array of y grid points (sorted)
    Z: 2D array of values, shape (len(xgrid), len(ygrid))
    """
    nx = len(xgrid)
    ny = len(ygrid)
    
    # clamp to grid bounds
    x = max(xgrid[0], min(x, xgrid[-1]))
    y = max(ygrid[0], min(y, ygrid[-1]))
    
    # find indices
    ix = 0
    for i in range(nx - 1):
        if xgrid[i + 1] >= x:
            ix = i
            break
    else:
        ix = nx - 2
    
    iy = 0
    for j in range(ny - 1):
        if ygrid[j + 1] >= y:
            iy = j
            break
    else:
        iy = ny - 2
    
    # interpolation weights
    x0, x1 = xgrid[ix], xgrid[ix + 1]
    y0, y1 = ygrid[iy], ygrid[iy + 1]
    
    if x1 - x0 > 0:
        wx = (x - x0) / (x1 - x0)
    else:
        wx = 0.0
    
    if y1 - y0 > 0:
        wy = (y - y0) / (y1 - y0)
    else:
        wy = 0.0
    
    # bilinear interpolation
    z00 = Z[ix, iy]
    z01 = Z[ix, iy + 1]
    z10 = Z[ix + 1, iy]
    z11 = Z[ix + 1, iy + 1]
    
    return (z00 * (1 - wx) * (1 - wy) +
            z10 * wx * (1 - wy) +
            z01 * (1 - wx) * wy +
            z11 * wx * wy)


@njit(cache=True, parallel=True)
def bellman_sdp_ar1_jit(
    H_prev, discr_s, discr_log_q, discr_u, discr_q, log_discr_q,
    vv_all, VV_all, wt, beta, gamma, delta,
    mu_t, sigma_t, rho_t, mu_t_prev, sigma_t_prev
):
    """
    JIT-compiled Bellman update for all (s, log_q_prev) states.
    
    Returns:
      H_new : np.ndarray, shape (n_s, n_log_q)
    """
    n_s = len(discr_s)
    n_log_q = len(discr_log_q)
    n_u = len(discr_u)
    n_q = len(discr_q)
    
    H_new = np.zeros((n_s, n_log_q))
    
    # conditional std (same for all log_q_prev values, only mean changes)
    rho_sq = rho_t * rho_t
    cond_std = sigma_t * math.sqrt(max(1.0 - rho_sq, 0.01))
    sigma_ratio = sigma_t / sigma_t_prev
    
    # parallel over storage states
    for i in prange(n_s):
        s_curr = discr_s[i]
        
        # precompute R, s_next, G for this storage state
        R = np.zeros((n_q, n_u))
        s_next = np.zeros((n_q, n_u))
        G = np.zeros((n_q, n_u))
        
        for iq in range(n_q):
            vv = vv_all[i, iq]
            VV = VV_all[i, iq]
            q_val = discr_q[iq]
            
            for iu in range(n_u):
                u_val = discr_u[iu]
                # actual release constrained by min/max
                r_val = min(VV, max(vv, u_val))
                R[iq, iu] = r_val
                s_next[iq, iu] = s_curr + delta * (q_val - r_val)
                # immediate cost
                deficit = wt - r_val
                if deficit > 0:
                    G[iq, iu] = deficit ** beta
                else:
                    G[iq, iu] = 0.0
        
        # clip s_next to grid bounds
        s_min = discr_s[0]
        s_max = discr_s[-1]
        for iq in range(n_q):
            for iu in range(n_u):
                if s_next[iq, iu] < s_min:
                    s_next[iq, iu] = s_min
                elif s_next[iq, iu] > s_max:
                    s_next[iq, iu] = s_max
        
        # precompute H_future for all (q, u)
        H_future = np.zeros((n_q, n_u))
        log_q_min = discr_log_q[0]
        log_q_max = discr_log_q[-1]
        
        for iq in range(n_q):
            log_q_curr = log_discr_q[iq]
            # clip log_q to grid
            if log_q_curr < log_q_min:
                log_q_curr = log_q_min
            elif log_q_curr > log_q_max:
                log_q_curr = log_q_max
            
            for iu in range(n_u):
                H_future[iq, iu] = bilinear_interp(
                    s_next[iq, iu], log_q_curr,
                    discr_s, discr_log_q, H_prev
                )
        
        # Q_mat = G + gamma * H_future
        Q_mat = np.zeros((n_q, n_u))
        for iq in range(n_q):
            for iu in range(n_u):
                Q_mat[iq, iu] = G[iq, iu] + gamma * H_future[iq, iu]
        
        # loop over log_q_prev
        for j in range(n_log_q):
            log_q_prev = discr_log_q[j]
            
            # conditional mean
            cond_mean = mu_t + rho_t * sigma_ratio * (log_q_prev - mu_t_prev)
            
            # compute probability mass for each q bin
            p_q = np.zeros(n_q)
            for iq in range(n_q):
                if iq == 0:
                    p_q[iq] = norm_cdf_scaled(log_discr_q[iq], cond_mean, cond_std)
                else:
                    p_q[iq] = (norm_cdf_scaled(log_discr_q[iq], cond_mean, cond_std) -
                               norm_cdf_scaled(log_discr_q[iq - 1], cond_mean, cond_std))
            
            # normalize
            p_sum = 0.0
            for iq in range(n_q):
                p_sum += p_q[iq]
            if p_sum > 0:
                for iq in range(n_q):
                    p_q[iq] /= p_sum
            
            # expected cost Q(u) = sum_q Q_mat[q,u] * p_q[q]
            Q_exp = np.zeros(n_u)
            for iu in range(n_u):
                for iq in range(n_q):
                    Q_exp[iu] += Q_mat[iq, iu] * p_q[iq]
            
            # Bellman: min over u
            H_new[i, j] = Q_exp[0]
            for iu in range(1, n_u):
                if Q_exp[iu] < H_new[i, j]:
                    H_new[i, j] = Q_exp[iu]
    
    return H_new


def bellman_sdp_ar1(H_prev, s_curr, log_q_prev, sys_param):
    """
    Bellman update with AR(1) autocorrelated inflows.

    Inputs:
      H_prev    : np.ndarray, shape (n_states, n_log_q) - value function for next period
      s_curr    : float - current storage state
      log_q_prev: float - log of previous inflow

    Returns:
      H_new  : float - updated cost-to-go
      idx_u  : tuple - optimal control indices
    """
    # fetch grids and parameters
    discr_s = sys_param["algorithm"]["discr_s"]  # (n_s,)
    discr_u = sys_param["algorithm"]["discr_u"]  # (n_u,)
    discr_q = sys_param["algorithm"]["discr_q"]  # (n_q,)
    discr_log_q = sys_param["algorithm"]["discr_log_q"]  # (n_log_q,)
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

    # next state for each (q, u)
    qq_mat = np.broadcast_to(discr_q[:, None], (n_q, n_u))
    s_next = s_curr + delta * (qq_mat - R)  # (n_q, n_u)

    # immediate cost G[q, u]
    G = np.maximum(wt - R, 0.0) ** sys_param["algorithm"]["beta"]

    # get AR(1) parameters
    mu_t, sigma_t, rho_t = sys_param["algorithm"]["stat_t"]
    mu_t_prev, sigma_t_prev, _ = sys_param["algorithm"]["stat_t_prev"]

    # compute conditional distribution of log(q_t) given log(q_{t-1})
    cond_mean, cond_std = conditional_lognormal_params(
        log_q_prev, mu_t, sigma_t, mu_t_prev, sigma_t_prev, rho_t
    )

    # compute probability mass for each discretized q
    # using the conditional normal distribution on log(q)
    log_discr_q_vals = np.log(np.maximum(discr_q, 1e-6))  # log of inflow bins
    cdf_log_q = norm(loc=cond_mean, scale=cond_std).cdf(log_discr_q_vals)
    p_q = np.diff(cdf_log_q)
    p0 = cdf_log_q[0]  # probability of q < discr_q[0]
    p_q = np.concatenate(([p0], p_q))  # (n_q,)
    p_q = p_q / p_q.sum()  # normalize

    # create interpolator for H_prev over (s, log_q)
    interp_H = RegularGridInterpolator(
        (discr_s, discr_log_q), H_prev,
        method="linear", bounds_error=False, fill_value=None
    )

    # vectorized interpolation: flatten (n_q, n_u) grid
    # For next state, log_q dimension is log(q_t) where q_t is the realized inflow
    s_next_clip = np.clip(s_next, discr_s[0], discr_s[-1])
    log_q_next = np.clip(
        np.broadcast_to(log_discr_q_vals[:, None], (n_q, n_u)),
        discr_log_q[0], discr_log_q[-1]
    )
    # stack into (n_q * n_u, 2) for batch interpolation
    pts = np.stack([s_next_clip.ravel(), log_q_next.ravel()], axis=-1)
    H_future = interp_H(pts).reshape(n_q, n_u)

    # expected cost Q(u) = sum_q [ G + γ H_future ] * p_q
    Q_mat = G + gamma * H_future  # (n_q, n_u)
    Q_exp = (Q_mat.T * p_q).sum(axis=1)  # (n_u,)

    # Bellman: pick minimal expectation
    H_new = Q_exp.min()
    eps = np.finfo(float).eps
    idx_u = np.where(Q_exp <= (H_new + eps))

    return H_new, idx_u


def opt_sdp_ar1(tol, max_iter, sys_param):
    """
    Stochastic DP with AR(1) autocorrelated inflows.

    The state space is expanded to (storage, log(q_prev)).
    Uses Numba JIT compilation for speedup.

    Args:
        tol (float): tolerance for convergence
        max_iter (int): maximum number of Bellman iterations

    Returns:
        H (np.ndarray): optimal value function, shape (n_s, n_log_q, T)
    """
    # pull params
    algo = sys_param["algorithm"]
    discr_s = algo["discr_s"].astype(np.float64)  # (n_s,)
    discr_log_q = algo["discr_log_q"].astype(np.float64)  # (n_log_q,)
    discr_u = algo["discr_u"].astype(np.float64)  # (n_u,)
    discr_q = algo["discr_q"].astype(np.float64)  # (n_q,)
    min_rel = algo["min_rel"].astype(np.float64)  # (n_s, n_q, T)
    max_rel = algo["max_rel"].astype(np.float64)  # (n_s, n_q)
    q_stat = algo["q_stat"].astype(np.float64)  # (T, 3) with [mu, sigma, rho]
    T = algo["T"]
    gamma = algo["gamma"]
    beta = algo["beta"]
    delta = sys_param["simulation"]["delta"]
    w = sys_param["simulation"]["w"].astype(np.float64)
    
    n_s = len(discr_s)
    n_log_q = len(discr_log_q)

    # precompute log of discr_q
    log_discr_q = np.log(np.maximum(discr_q, 1e-6))

    # initialize Bellman value matrix (3D now)
    H = np.zeros((n_s, n_log_q, T))
    diff_H = np.inf
    diff_H_prev = np.inf
    count = 0

    print(f"SDP-AR1 (Numba JIT): state space = ({n_s}, {n_log_q}, {T})")
    print("Compiling JIT functions on first iteration...")

    # backward recursion
    while diff_H >= tol and count < max_iter:

        # copy previous value function
        H_old = H.copy()

        # backward recursion
        for t in range(T - 1, -1, -1):
            t_prev = (t - 1) % T
            t_next = (t + 1) % T

            # get parameters for this time step
            wt = w[t]
            mu_t, sigma_t, rho_t = q_stat[t, :]
            mu_t_prev, sigma_t_prev, _ = q_stat[t_prev, :]

            # next-stage values
            H_next = np.ascontiguousarray(H[:, :, t_next])

            # min/max release for this time step
            vv_all = np.ascontiguousarray(min_rel[:, :, t])  # (n_s, n_q)
            VV_all = np.ascontiguousarray(max_rel[:, :])  # (n_s, n_q)

            # JIT-compiled Bellman update
            H[:, :, t] = bellman_sdp_ar1_jit(
                H_next, discr_s, discr_log_q, discr_u, discr_q, log_discr_q,
                vv_all, VV_all, wt, beta, gamma, delta,
                mu_t, sigma_t, rho_t, mu_t_prev, sigma_t_prev
            )

            if t % 50 == 0:
                print(f"  Day {t} done")

        # check convergence
        diff_H_prev = diff_H
        diff_H = np.max(np.abs(H - H_old))
        count += 1
        print(f"Iteration {count}: max diff = {diff_H:.4f}")
        
        # stop if difference hasn't changed (stuck at numerical precision limit)
        if count > 1 and abs(diff_H - diff_H_prev) < 1e-10:
            print(f"  Convergence stalled (diff unchanged). Stopping.")
            break

    return H
