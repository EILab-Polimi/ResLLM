"""Plot Folsom Reservoir historical storage, outflow, and demand.

Generates time series plots for each water year showing:
- Daily storage vs. rule curve
- Daily outflow with 30-day moving average vs. demand

Computes summary statistics including annual averages for demand, inflow, 
outflow, evaporation, and June storage.
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300

# %% Load data
# Folsom daily data (1996-2016)
df = pd.read_csv('folsom_daily_96_16.csv', parse_dates=['date'])

# Demand (365 values, one per day of water year)
with open('demand.txt') as f:
    demand = np.array([float(line.strip()) for line in f])

# Rule curve (365 values, one per day of water year)
with open('tocs.txt') as f:
    rule_curve = np.array([float(line.strip()) for line in f])

# Helper functions
def get_water_year(date):
    """Assign water year (Oct 1 - Sep 30)."""
    return date.year + 1 if date.month >= 10 else date.year

def assign_month(date):
    """Extract month from date."""
    return date.month

df['water_year'] = df['date'].apply(get_water_year)
df['month'] = df['date'].apply(assign_month)

# %% Plot storage and outflow for each water year
water_years = df['water_year'].unique()
for wy in water_years:
    wy_df = df[df['water_year'] == wy]
    storage = wy_df['storage'].values
    outflow = wy_df['outflow'].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True)
    fig.suptitle(f'Folsom Storage and Outflow - Water Year {wy}', y=0.9)

    # Storage plot
    ax1.plot(np.arange(1, len(storage) + 1), storage, label='Storage')
    ax1.plot(np.arange(1, len(rule_curve) + 1), rule_curve, 'k--', label='Rule Curve')
    ax1.set_ylabel('Storage (TAF)')
    ax1.set_ylim(0, 1000)
    ax1.legend()
    ax1.grid(True)

    # Outflow plot
    ax2.plot(np.arange(1, len(outflow) + 1), outflow, alpha=0.5, color='grey', label='Outflow')
    
    # 30-day centered moving average
    window_size = 30
    if len(outflow) >= window_size:
        outflow_ma = np.convolve(outflow, np.ones(window_size) / window_size, mode='valid')
        ax2.plot(np.arange(window_size // 2 + 1, len(outflow) - window_size // 2 + 1), 
                 outflow_ma[1:], 'blue', alpha=1.0, label='30-day MA')
    
    ax2.plot(np.arange(1, len(demand) + 1), demand, 'r--', label='Demand')
    ax2.set_ylabel('Outflow (TAF)')
    ax2.set_xlabel('Day of Water Year (Oct 1 = day 1)')
    ax2.legend()
    ax2.set_ylim(0, 20)
    ax2.grid(True)

    # Set x-axis ticks to month starts
    month_starts = [1, 32, 62, 93, 124, 152, 183, 213, 244, 274, 305, 336]
    month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    ax2.set_xticks(month_starts)
    ax2.set_xticklabels(month_labels)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
# %% Compute summary statistics
print(f'Average annual demand: {demand.mean() * 365:.1f} TAF')
print(f'Average annual inflow: {df["inflow"].mean() * 365:.1f} TAF')
print(f'Average annual outflow: {df["outflow"].mean() * 365:.1f} TAF')
print(f'Average annual losses: {df["evap"].mean() * 365:.1f} TAF')
print(f'Average June storage: {df["storage"].loc[df["month"] == 6].mean():.1f} TAF')

# %%
