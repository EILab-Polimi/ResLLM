import pandas as pd
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime

# ==============================================================================
# CONFIGURATION AND DATA PATHS
# ==============================================================================

# Base repo directory (two levels up from benchmarks/nn/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "benchmarks", "nn", "output")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load demand data (366 values, one per day of water year)
# Water year starts October 1 (day 1 of WY) and ends September 30 (day 365/366)
demand_path = os.path.join(DATA_DIR, "demand.txt")
demand_data = np.loadtxt(demand_path)
print(f"Demand data loaded: {len(demand_data)} values")

# Load reservoir data
data_path = os.path.join(DATA_DIR, "folsom_daily.csv")
df = pd.read_csv(data_path, parse_dates=["date"])


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_water_year_day(date):
    """Calculate day of water year (1-366). Oct 1 = day 1, Sep 30 = day 365/366"""
    if date.month >= 10:
        oct_1 = pd.Timestamp(year=date.year, month=10, day=1)
    else:
        oct_1 = pd.Timestamp(year=date.year - 1, month=10, day=1)
    return (date - oct_1).days + 1


# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================

df["water_year_day"] = df["date"].apply(get_water_year_day)

# Map demand to each day using water year day (handle leap year day 366)
df["demand"] = df["water_year_day"].apply(
    lambda wd: demand_data[min(wd - 1, len(demand_data) - 1)]
)

# Calculate water year
df["water_year"] = df["date"].apply(lambda x: x.year if x.month < 10 else x.year + 1)

# Apply demand reduction for training periods to simulate historical lower demand
demand_reduction_mask_1 = (df["water_year"] >= 1961) & (df["water_year"] <= 1980)
demand_reduction_mask_2 = (df["water_year"] > 1980) & (df["water_year"] < 1996)

df.loc[demand_reduction_mask_1, "demand"] *= 0.9
df.loc[demand_reduction_mask_2, "demand"] *= 0.95

print(f"\nDemand adjustments applied:")
print(f"  1961-1980: 10% reduction ({demand_reduction_mask_1.sum()} days)")
print(f"  1981-1995: 5% reduction ({demand_reduction_mask_2.sum()} days)")
print(f"  1996-2016: No adjustment (test period)")

# Create water year month (Oct = 1, Nov = 2, ..., Sep = 12)
df["water_year_month"] = df["date"].apply(
    lambda x: x.month - 9 if x.month >= 10 else x.month + 3
)

# Create cyclic encoding of water year month
df["month_sin"] = np.sin(2 * np.pi * df["water_year_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["water_year_month"] / 12)

# Calculate 25-day centered moving average of release (capped at 10 TAF/day)
df["release_ma"] = (
    df["outflow"].clip(upper=10).rolling(window=25, min_periods=1).mean().shift(-13)
)
df["allocation"] = (df["release_ma"] / df["demand"]).clip(upper=1.0)

# Calculate previous and future 30-day rolling average allocation
df["allocation_prev"] = df["allocation"].rolling(window=30, min_periods=1).mean().shift(1)
df["allocation_future"] = df["allocation"].rolling(window=30, min_periods=1).mean().shift(-29)

# Calculate 120-day average inflow
df["inflow_ma"] = df["inflow"].rolling(window=120, min_periods=1).mean()

# Extract first day of each month for training data
df["is_first_of_month"] = df["date"].dt.day == 1
monthly_data = df[df["is_first_of_month"]].copy()

print(f"\nTotal daily records: {len(df)}")
print(f"Monthly records (first of month): {len(monthly_data)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# ==============================================================================
# TRAIN/TEST SPLIT AND FEATURE PREPARATION
# ==============================================================================

# Remove rows with missing values
monthly_data = monthly_data.dropna(
    subset=["storage", "allocation_future", "allocation_prev", "inflow_ma"]
)

print(f"\nMonthly records after dropping NaN: {len(monthly_data)}")
print(f"\nTarget (future allocation) statistics:")
print(f"  Mean: {monthly_data['allocation_future'].mean():.4f}")
print(f"  Std:  {monthly_data['allocation_future'].std():.4f}")
print(f"  Min:  {monthly_data['allocation_future'].min():.4f}")
print(f"  Max:  {monthly_data['allocation_future'].max():.4f}")

# Split data by water year
train_mask = (monthly_data["water_year"] >= 1961) & (monthly_data["water_year"] < 1996)
test_mask = (monthly_data["water_year"] >= 1996) & (monthly_data["water_year"] <= 2016)

train_data = monthly_data[train_mask]
test_data = monthly_data[test_mask]

# Prepare features and target
feature_cols = ["storage", "month_sin", "month_cos", "allocation_prev", "inflow_ma"]
X_train = train_data[feature_cols].values
y_train = train_data["allocation_future"].values

X_test = test_data[feature_cols].values
y_test = test_data["allocation_future"].values

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {feature_cols}")
print(f"Training period: WY {train_data['water_year'].min()}-{train_data['water_year'].max()}")
print(f"Test period: WY {test_data['water_year'].min()}-{test_data['water_year'].max()}")

# ==============================================================================
# STANDARDIZATION
# ==============================================================================

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

print(f"\nFeature standardization:")
print(f"  Means: {scaler_X.mean_}")
print(f"  Stds:  {scaler_X.scale_}")
print(f"\nTarget standardization:")
print(f"  Mean: {scaler_y.mean_[0]:.4f}")
print(f"  Std:  {scaler_y.scale_[0]:.4f}")

# ==============================================================================
# MODEL TRAINING
# ==============================================================================

mlp = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="lbfgs",
    max_iter=10000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.28,
    verbose=True,
    alpha=2.5,
    learning_rate_init=0.0001,
    batch_size=32,
    n_iter_no_change=12,
)

print("\nTraining MLP model for allocation prediction...")
mlp.fit(X_train_scaled, y_train_scaled)

# ==============================================================================
# PREDICTIONS AND EVALUATION
# ==============================================================================

# Make predictions
y_train_pred_scaled = mlp.predict(X_train_scaled)
y_test_pred_scaled = mlp.predict(X_test_scaled)

# Inverse transform to original scale
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

# Cap predicted allocations at 1.0 (100%)
y_train_pred = np.clip(y_train_pred, None, 1.0)
y_test_pred = np.clip(y_test_pred, None, 1.0)

# Calculate overall metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Calculate shortage-only metrics (allocation < 1.0)
train_mask_shortage = y_train < 1.0
test_mask_shortage = y_test < 1.0

if train_mask_shortage.sum() > 0:
    train_shortage_mse = mean_squared_error(
        y_train[train_mask_shortage], y_train_pred[train_mask_shortage]
    )
    train_shortage_rmse = np.sqrt(train_shortage_mse)
    train_shortage_r2 = r2_score(
        y_train[train_mask_shortage], y_train_pred[train_mask_shortage]
    )
else:
    train_shortage_mse = train_shortage_rmse = train_shortage_r2 = np.nan

if test_mask_shortage.sum() > 0:
    test_shortage_mse = mean_squared_error(
        y_test[test_mask_shortage], y_test_pred[test_mask_shortage]
    )
    test_shortage_rmse = np.sqrt(test_shortage_mse)
    test_shortage_r2 = r2_score(
        y_test[test_mask_shortage], y_test_pred[test_mask_shortage]
    )
else:
    test_shortage_mse = test_shortage_rmse = test_shortage_r2 = np.nan

# Print performance summary
print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)
print(f"\nArchitecture: {mlp.hidden_layer_sizes} neurons, alpha={mlp.alpha}")
print(f"Features: {len(feature_cols)}")
print(f"\nAll Data:")
print(f"  Training   - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
print(f"  Test       - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

print(f"\nShortage Conditions Only (allocation < 1.0):")
print(f"  Training (n={train_mask_shortage.sum()}):", end="")
if train_mask_shortage.sum() > 0:
    print(f" R²: {train_shortage_r2:.4f}, RMSE: {train_shortage_rmse:.4f}")
else:
    print(" No shortage conditions")

print(f"  Test (n={test_mask_shortage.sum()}):", end="")
if test_mask_shortage.sum() > 0:
    print(f" R²: {test_shortage_r2:.4f}, RMSE: {test_shortage_rmse:.4f}")
else:
    print(" No shortage conditions")

print(f"\nPrediction Ranges:")
print(f"  Training - Actual: [{y_train.min():.4f}, {y_train.max():.4f}], "
      f"Predicted: [{y_train_pred.min():.4f}, {y_train_pred.max():.4f}]")
print(f"  Test     - Actual: [{y_test.min():.4f}, {y_test.max():.4f}], "
      f"Predicted: [{y_test_pred.min():.4f}, {y_test_pred.max():.4f}]")
print("=" * 80)

# ==============================================================================
# VISUALIZATION
# ==============================================================================

fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Training predictions vs actual
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_train, y_train_pred, alpha=0.5, s=20)
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2)
ax1.axhline(y=1.0, color="green", linestyle=":", lw=1, label="Full allocation")
ax1.axvline(x=1.0, color="green", linestyle=":", lw=1)
ax1.set_xlabel("Actual Allocation")
ax1.set_ylabel("Predicted Allocation")
ax1.set_title(f"Training Set (R²={train_r2:.3f})")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Test predictions vs actual
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, y_test_pred, alpha=0.5, s=20)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
ax2.axhline(y=1.0, color="green", linestyle=":", lw=1, label="Full allocation")
ax2.axvline(x=1.0, color="green", linestyle=":", lw=1)
ax2.set_xlabel("Actual Allocation")
ax2.set_ylabel("Predicted Allocation")
ax2.set_title(f"Test Set (R²={test_r2:.3f})")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Time series plot for test set
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(test_data["date"], y_test, label="Actual", alpha=0.7, linewidth=2, 
         marker="o", markersize=3)
ax3.plot(test_data["date"], y_test_pred, label="Predicted", alpha=0.7, linewidth=2, 
         marker="s", markersize=3)
ax3.axhline(y=1.0, color="green", linestyle="--", lw=1, label="Full allocation")
ax3.set_xlabel("Date")
ax3.set_ylabel("Allocation (Release/Demand)")
ax3.set_title("Test Set Time Series (Monthly)")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis="x", rotation=45)

# Monthly averages - Training
train_wy_month_actual = train_data.groupby("water_year_month")["allocation_future"].mean()
train_data_with_pred = train_data.copy()
train_data_with_pred["allocation_pred"] = y_train_pred
train_wy_month_pred = train_data_with_pred.groupby("water_year_month")["allocation_pred"].mean()

# Monthly averages - Test
test_wy_month_actual = test_data.groupby("water_year_month")["allocation_future"].mean()
test_data_with_pred = test_data.copy()
test_data_with_pred["allocation_pred"] = y_test_pred
test_wy_month_pred = test_data_with_pred.groupby("water_year_month")["allocation_pred"].mean()

month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
x_pos = np.arange(1, 13)

# Training monthly plot
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(x_pos, train_wy_month_actual, "o-", label="Actual", linewidth=2, markersize=8)
ax4.plot(x_pos, train_wy_month_pred, "s-", label="Predicted", linewidth=2, markersize=8)
ax4.axhline(y=1.0, color="green", linestyle="--", lw=1, label="Full allocation")
ax4.set_xlabel("Water Year Month")
ax4.set_ylabel("Mean Allocation")
ax4.set_title("Average Allocation by Month (Training Set)")
ax4.set_xticks(x_pos)
ax4.set_xticklabels(month_labels, rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Test monthly plot
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(x_pos, test_wy_month_actual, "o-", label="Actual", linewidth=2, markersize=8)
ax5.plot(x_pos, test_wy_month_pred, "s-", label="Predicted", linewidth=2, markersize=8)
ax5.axhline(y=1.0, color="green", linestyle="--", lw=1, label="Full allocation")
ax5.set_xlabel("Water Year Month")
ax5.set_ylabel("Mean Allocation")
ax5.set_title("Average Allocation by Month (Test Set)")
ax5.set_xticks(x_pos)
ax5.set_xticklabels(month_labels, rotation=45)
ax5.legend()
ax5.grid(True, alpha=0.3)

output_fig_path = os.path.join(OUTPUT_DIR, "mlp_allocation_results.png")
plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
print(f"\nFigure saved: {output_fig_path}")
plt.close()

# ==============================================================================
# SAVE MODEL, SCALERS, AND METADATA
# ==============================================================================

print("\n" + "=" * 80)
print("SAVING MODEL ARTIFACTS")
print("=" * 80)

# Create model metadata
model_metadata = {
    "model_type": "MLPRegressor",
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "features": feature_cols,
    "num_features": len(feature_cols),
    "architecture": {
        "hidden_layers": list(mlp.hidden_layer_sizes),
        "activation": mlp.activation,
        "solver": mlp.solver,
        "alpha": float(mlp.alpha),
        "learning_rate_init": float(mlp.learning_rate_init),
        "batch_size": mlp.batch_size,
    },
    "training_period": f"WY {train_data['water_year'].min()}-{train_data['water_year'].max()}",
    "test_period": f"WY {test_data['water_year'].min()}-{test_data['water_year'].max()}",
    "performance": {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "test_shortage_r2": float(test_shortage_r2) if not np.isnan(test_shortage_r2) else None,
        "test_shortage_rmse": float(test_shortage_rmse) if not np.isnan(test_shortage_rmse) else None,
    },
    "scaler_means": scaler_X.mean_.tolist(),
    "scaler_stds": scaler_X.scale_.tolist(),
    "target_scaler_mean": float(scaler_y.mean_[0]),
    "target_scaler_std": float(scaler_y.scale_[0]),
}

# Save model, scalers, and metadata
model_path = os.path.join(OUTPUT_DIR, "mlp_allocation_model.pkl")
scalers_path = os.path.join(OUTPUT_DIR, "mlp_allocation_scalers.pkl")
metadata_path = os.path.join(OUTPUT_DIR, "mlp_allocation_metadata.json")

joblib.dump(mlp, model_path)
print(f"✓ Model saved: {model_path}")

scalers = {"scaler_X": scaler_X, "scaler_y": scaler_y}
joblib.dump(scalers, scalers_path)
print(f"✓ Scalers saved: {scalers_path}")

with open(metadata_path, "w") as f:
    json.dump(model_metadata, f, indent=2)
print(f"✓ Metadata saved: {metadata_path}")

# ==============================================================================
# SAVE TRAINING AND TEST PREDICTIONS
# ==============================================================================

# Create training data DataFrame
train_output = pd.DataFrame({
    "date": train_data["date"].values,
    "water_year": train_data["water_year"].values,
    "water_year_month": train_data["water_year_month"].values,
    "storage": X_train[:, 0],
    "month_sin": X_train[:, 1],
    "month_cos": X_train[:, 2],
    "allocation_prev": X_train[:, 3],
    "inflow_ma": X_train[:, 4],
    "actual_allocation": y_train,
    "predicted_allocation": y_train_pred,
    "error": y_train - y_train_pred,
    "abs_error": np.abs(y_train - y_train_pred),
    "is_shortage": y_train < 1.0,
})

# Create test data DataFrame
test_output = pd.DataFrame({
    "date": test_data["date"].values,
    "water_year": test_data["water_year"].values,
    "water_year_month": test_data["water_year_month"].values,
    "storage": X_test[:, 0],
    "month_sin": X_test[:, 1],
    "month_cos": X_test[:, 2],
    "allocation_prev": X_test[:, 3],
    "inflow_ma": X_test[:, 4],
    "actual_allocation": y_test,
    "predicted_allocation": y_test_pred,
    "error": y_test - y_test_pred,
    "abs_error": np.abs(y_test - y_test_pred),
    "is_shortage": y_test < 1.0,
})

# Save to CSV
train_output_path = os.path.join(OUTPUT_DIR, "mlp_allocation_train_predictions.csv")
test_output_path = os.path.join(OUTPUT_DIR, "mlp_allocation_test_predictions.csv")

train_output.to_csv(train_output_path, index=False)
print(f"✓ Training predictions saved: {train_output_path} ({train_output.shape[0]} rows)")

test_output.to_csv(test_output_path, index=False)
print(f"✓ Test predictions saved: {test_output_path} ({test_output.shape[0]} rows)")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
