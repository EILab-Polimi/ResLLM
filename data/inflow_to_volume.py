import pandas as pd
from calendar import monthrange
import io

file = "./f_kariba_inflow_monthly.csv"

df = pd.read_csv(file)

# 2. Ensure the date column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# 3. Define a function to calculate seconds in a specific month
def calculate_monthly_volume(row):
    year = row['date'].year
    month = row['date'].month
    
    # Get the number of days in that specific month (handles leap years automatically)
    _, num_days = monthrange(year, month)
    
    # Calculate total seconds: Days * 24 hours * 60 mins * 60 secs
    seconds_in_month = num_days * 86400
    
    # Calculate Volume: Flow Rate (m3/s) * Total Seconds
    return row['inflow'] * seconds_in_month

# 4. Apply the function to create a new column
df['volume_m3'] = df.apply(calculate_monthly_volume, axis=1)

# Optional: Create a "Million Cubic Meters" (MCM) column for easier reading
df['volume_MCM'] = df['volume_m3'] / 1_000_000

new_df = df[['date', 'volume_m3']]
new_df.rename(columns={'volume_m3': 'inflow'}, inplace=True)
new_df['inflow'] = new_df['inflow'].round(0)

new_df.to_csv("f_kariba_inflow_monthly_volume.csv", index=False)