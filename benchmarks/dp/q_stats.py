"""Compute and plot Folsom inflow rolling stats and daily log-normal fits."""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300

DATA_PATH = "../../data/folsom_daily.csv"
ROLLING_MEAN_WINDOW = 10
ROLLING_STD_WINDOW = 30
WATER_YEAR_DAYS = 365


def plot_rolling_stats(df_obs_year: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(
        df_obs_year.index.year,
        df_obs_year["rolling_mean"],
        label="Observed",
        color="grey",
    )
    plt.title("Folsom Reservoir Inflow (Rolling 10 Year Mean)")
    plt.xlabel("Year")
    plt.ylabel("Inflow (taf)")
    plt.legend()

    plt.figure(figsize=(10, 5))
    plt.plot(
        df_obs_year.index.year,
        df_obs_year["rolling_std"],
        label="Observed",
        color="grey",
    )
    plt.title("Folsom Reservoir Inflow (Rolling 30 Year Std.Dev)")
    plt.xlabel("Year")
    plt.ylabel("Inflow (taf)")
    plt.legend()

# %% log-normal fit for test period
def compute_lognormal_stats(
    df,
    start_date,
    end_date,
    inflow_col="inflow",
    T=WATER_YEAR_DAYS,
    zero_tol=0.00001,
    verbose=False,
):
    """
    Compute daily log-normal mean, std dev, and lag-1 autocorrelation for inflow between start_date and end_date.
    Drops Feb‑29 and replaces zeros (<=0.001) with the previous value.

    Parameters
    ----------
    df : pandas.DataFrame
        Time‑indexed dataframe containing an 'inflow' column.
    start_date : str or datetime-like
        Start of the period (inclusive).
    end_date : str or datetime-like
        End of the period (inclusive).
    inflow_col : str, default 'inflow'
        Name of the inflow column in df.
    T : int, default 365
        Number of days in a water year (excl. Feb‑29).
    zero_tol : float, default 0.001
        Tolerance for zero values. Values <= zero_tol are replaced with the previous value.

    Returns
    -------
    q_stat : ndarray, shape (T, 3)
        Daily [mean(log inflow), std(log inflow), lag-1 autocorrelation] for each day of the water year.
    """
    # build date index and slice the dataframe
    date_range = pd.date_range(start_date, end_date, freq="D")
    df_sel = df.loc[start_date:end_date]
    q = df_sel[inflow_col].to_numpy()

    is_feb29 = (date_range.month == 2) & (date_range.day == 29)
    q_clean = q[~is_feb29]

    # replace non‑positive values with the previous valid value
    while (q_clean <= zero_tol).any():
        idx = np.where(q_clean <= zero_tol)[0]
        q_clean[idx] = q_clean[idx - 1]

    # reshape into years × days and compute stats
    n_days = (q_clean.size // T) * T
    q_clean = q_clean[:n_days]
    Ny = q_clean.size // T
    if verbose:
        print(f"Number of years: {Ny}")
    Q = q_clean.reshape(Ny, T).T
    log_Q = np.log(Q)  # shape (T, Ny)
    
    q_stat = np.empty((T, 3))  # mean, std, lag-1 autocorrelation
    for i in range(T):
        logs = log_Q[i, :]
        q_stat[i, 0] = logs.mean()
        q_stat[i, 1] = logs.std()
        
        # lag-1 autocorrelation: correlation between day i and day i-1
        # For day 0, use the last day of the previous water year (day T-1)
        if i == 0:
            logs_prev = log_Q[T - 1, :-1]  # last day of previous years
            logs_curr = log_Q[i, 1:]       # first day of current years (shifted by 1 year)
        else:
            logs_prev = log_Q[i - 1, :]
            logs_curr = log_Q[i, :]
        
        # compute Pearson correlation coefficient
        if len(logs_prev) > 1 and logs_prev.std() > 0 and logs_curr.std() > 0:
            q_stat[i, 2] = np.corrcoef(logs_prev, logs_curr)[0, 1]
        else:
            q_stat[i, 2] = 0.0

    return q_stat


def main() -> None:
    df_obs = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    df_obs_year = df_obs.resample("YS-OCT").sum()
    df_obs_year["rolling_mean"] = df_obs_year["inflow"].rolling(
        ROLLING_MEAN_WINDOW
    ).mean()
    df_obs_year["rolling_std"] = df_obs_year["inflow"].rolling(
        ROLLING_STD_WINDOW
    ).std()
    plot_rolling_stats(df_obs_year)

    q_stat_pre70 = compute_lognormal_stats(df_obs, "1905-10-01", "1970-09-30")
    q_stat_test = compute_lognormal_stats(df_obs, "1995-10-01", "2016-09-30")
    q_stat_all = compute_lognormal_stats(df_obs, "1905-10-01", "2016-09-30")

    plt.figure(figsize=(10, 5))
    days = np.arange(1, WATER_YEAR_DAYS + 1)
    mu_test, sigma_test = q_stat_test[:, 0], q_stat_test[:, 1]
    mu_pre70, sigma_pre70 = q_stat_pre70[:, 0], q_stat_pre70[:, 1]

    plt.plot(days, mu_pre70, color="red", label="Obs 1905-1970")
    plt.plot(days, mu_test, color="blue", label="Obs 1995-2016")

    k = 1
    alpha = 0.3
    plt.fill_between(
        days,
        mu_test - k * sigma_test,
        mu_test + k * sigma_test,
        color="blue",
        alpha=alpha,
    )
    plt.fill_between(
        days,
        mu_pre70 - k * sigma_pre70,
        mu_pre70 + k * sigma_pre70,
        color="red",
        alpha=alpha,
    )

    plt.xlabel("Day of Water Year")
    plt.ylabel("Log Inflow")
    plt.title("Daily Log‑Normal Fit (Mean ± 1 Std Dev)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    np.savez(
        "lognormal_qstats.npz",
        q_stat_obs_pre70=q_stat_pre70,
        q_stat_obs_test=q_stat_test,
        q_stat_obs_all=q_stat_all,
    )


if __name__ == "__main__":
    main()

# %%
