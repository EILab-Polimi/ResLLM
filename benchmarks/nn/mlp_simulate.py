"""
Simulation script using the trained MLP allocation model to make decisions.

This script simulates reservoir operations using the MLP model to make monthly 
allocation decisions, following the same logic as simulate.py.

Supports both historical (1905-2016) and paleo (1900-1999) simulations.

Usage:
    # Historical simulation (default):
    python mlp_simulate.py
    
    # Paleo simulation:
    python mlp_simulate.py --config folsom_paleo --start-year 1900 --end-year 1999
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
import yaml
import argparse

# Add resllm to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'resllm'))
from src.reservoir import Reservoir
import src.utils as utils

def load_model():
    """Load the trained model, scalers, and metadata."""
    # Load model
    mlp = joblib.load('./output/mlp_allocation_model.pkl')
    
    # Load scalers
    scalers = joblib.load('./output/mlp_allocation_scalers.pkl')
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']
    
    # Load metadata
    with open('./output/mlp_allocation_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return mlp, scaler_X, scaler_y, metadata

def prepare_features(storage, water_year_month, allocation_prev, inflow_ma):
    """
    Prepare features for prediction.
    
    Parameters:
    -----------
    storage : float or array-like
        Current reservoir storage (TAF)
    water_year_month : int or array-like
        Water year month (1=Oct, 2=Nov, ..., 12=Sep)
    allocation_prev : float or array-like
        Previous month's allocation (0.0 to 1.0)
        Calculated as: (30-day rolling avg release) / (30-day rolling avg demand)
    inflow_ma : float or array-like
        120-day moving average inflow
    
    Returns:
    --------
    X : numpy array
        Feature matrix with shape (n_samples, 5)
        Features: [storage, month_sin, month_cos, allocation_prev, inflow_ma]
    """
    # Convert to numpy arrays if single values
    storage = np.atleast_1d(storage)
    water_year_month = np.atleast_1d(water_year_month)
    allocation_prev = np.atleast_1d(allocation_prev)
    inflow_ma = np.atleast_1d(inflow_ma)
    
    # Calculate cyclic encoding for month
    month_sin = np.sin(2 * np.pi * water_year_month / 12)
    month_cos = np.cos(2 * np.pi * water_year_month / 12)
    
    # Combine features in the correct order
    X = np.column_stack([storage, month_sin, month_cos, allocation_prev, inflow_ma])
    
    return X


def predict_allocation(mlp, scaler_X, scaler_y, X):
    """
    Make allocation predictions.
    
    Parameters:
    -----------
    mlp : MLPRegressor
        Trained model
    scaler_X : StandardScaler
        Feature scaler
    scaler_y : StandardScaler
        Target scaler
    X : numpy array
        Feature matrix
    
    Returns:
    --------
    predictions : numpy array
        Predicted allocations (0.0 to 1.0)
    """
    # Scale features
    X_scaled = scaler_X.transform(X)
    
    # Make predictions
    y_pred_scaled = mlp.predict(X_scaled)
    
    # Inverse transform to original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Clip to valid range [0, 1]
    y_pred = np.clip(y_pred, 0.0, 1.0)
    
    return y_pred


def run_simulation(start_year=1996, end_year=2016, starting_storage=466.1, config_name="folsom_hist", fix_tocs=False):
    """
    Run reservoir simulation using MLP model for allocation decisions.
    
    Parameters:
    -----------
    start_year : int
        Starting water year
    end_year : int
        Ending water year
    starting_storage : float
        Initial storage in TAF
    config_name : str
        Name of the configuration file (without .yml extension)
        Options: 'folsom_hist' or 'folsom_paleo'
    
    Returns:
    --------
    simulation_df : pd.DataFrame
        Daily simulation results
    decision_df : pd.DataFrame
        Monthly allocation decisions
    """
    print("="*60)
    print("MLP-Based Reservoir Simulation")
    print("="*60)
    
    # Load MLP model
    print("\nLoading MLP model...")
    mlp, scaler_X, scaler_y, metadata = load_model()
    print(f"✓ Model loaded (Test R²: {metadata['performance']['test_r2']:.4f})")
    
    # Load reservoir configuration
    config_path = f"../../resllm/configs/{config_name}.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"✓ Configuration loaded: {config['config_name']}")
    
    # Setup reservoir
    data_dir = "../../data"
    R1_characteristics = {
        "tocs": 'fixed' if fix_tocs else config["folsom_reservoir"]["tocs"],
        "demand_file": os.path.join(data_dir, config["folsom_reservoir"]["demand_file"]),
        "inflow_file": os.path.join(data_dir, config["folsom_reservoir"]["inflow_file"]),
        "wy_forecast_file": False,
        "operable_storage_max": config["folsom_reservoir"]["operable_storage_max"],
        "operable_storage_min": config["folsom_reservoir"]["operable_storage_min"],
        "max_safe_release": utils.cfs_to_taf(config["folsom_reservoir"]["max_safe_release"]),
        "sp_to_ep": config["folsom_reservoir"]["sp_to_ep"],
        "tp_to_tocs": config["folsom_reservoir"]["tp_to_tocs"],
        "sp_to_rp": config["folsom_reservoir"]["sp_to_rp"],
    }
    
    R1 = Reservoir(characteristics=R1_characteristics)
    print("✓ Reservoir initialized")
    
    # Validate water year range
    available_years = sorted(R1.inflows['water_year'].unique())
    min_wy = available_years[0]
    max_wy = available_years[-1]
    
    if start_year < min_wy or end_year > max_wy:
        raise ValueError(
            f"\nRequested water years {start_year}-{end_year} are outside available range.\n"
            f"Available data: water years {min_wy}-{max_wy}\n"
            f"(Date range: {R1.inflows['date'].iloc[0]} to {R1.inflows['date'].iloc[-1]})"
        )
    
    print(f"✓ Data validation passed (WY {min_wy}-{max_wy} available)")
    
    # Initialize simulation
    ny = end_year - start_year + 1
    R1.record = pd.DataFrame()
    decision_record = []
    
    # Rolling window tracking for allocation_prev calculation
    release_history = []
    demand_history = []
    inflow_history = []
    
    # Initialize
    allocation_percent = 100.0
    allocation_prev = 1.0
    t = 0
    
    print(f"\nSimulating water years {start_year} to {end_year}...")
    print(f"Starting storage: {starting_storage:.1f} TAF")
    
    # Simulation loop
    for wy in np.arange(start_year, end_year + 1):
        print(f"  Water year {wy}...")
        
        # Date range for the water year
        date_range = pd.date_range(start=f"{wy-1}-10-01", end=f"{wy}-09-30", freq="D")
        
        # Remove leap day
        if len(date_range) == 366:
            leap_day = (date_range.month == 2) & (date_range.day == 29)
            date_range = date_range[~leap_day]
        
        # Loop through days
        for ty, d in enumerate(date_range):
            # Get water year month
            mowy = d.month - 9 if d.month > 9 else d.month + 3
            
            # Get previous storage
            st_1 = starting_storage if t == 0 else R1.record.loc[t - 1, "st"]
            
            # MLP Decision at start of each month
            if d.day == 1:
                # Calculate inflow_ma (120-day moving average)
                if len(inflow_history) >= 120:
                    inflow_ma = np.mean(inflow_history[-120:])
                elif len(inflow_history) > 0:
                    inflow_ma = np.mean(inflow_history)
                else:
                    inflow_ma = 7.0  # Default value
                
                # Prepare features for MLP
                X = prepare_features(
                    storage=st_1,
                    water_year_month=mowy,
                    allocation_prev=allocation_prev,
                    inflow_ma=inflow_ma
                )
                
                # Make prediction
                allocation_predicted = predict_allocation(mlp, scaler_X, scaler_y, X)[0]
                allocation_percent = allocation_predicted * 100.0
                
                # Record decision
                decision_record.append({
                    'date': d,
                    'wy': wy,
                    'mowy': mowy,
                    'dowy': ty + 1,
                    'storage': st_1,
                    'allocation_prev': allocation_prev,
                    'inflow_ma': inflow_ma,
                    'allocation_decision': allocation_predicted,
                    'allocation_percent': allocation_percent
                })
            
            # Current downstream demand
            dt = R1.demand[ty]
            # Set target demand from allocation decision
            uu = dt * allocation_percent / 100.0
            
            # Inflow
            qt = R1.inflows.loc[
                (R1.inflows["water_year"] == wy)
                & (R1.inflows["month"] == d.month)
                & (R1.inflows["day"] == d.day),
                "inflow",
            ].values[0]
            
            # TOCS and evaluate
            tocs = R1.compute_tocs(dowy=ty + 1, date=d.strftime("%Y-%m-%d"))
            rt, st = R1.evaluate(st_1=st_1, qt=qt, uu=uu, tocs=tocs)
            
            # Record timestep
            R1.record_timestep(
                idx=t, date=d, wy=wy, mowy=mowy, dowy=ty + 1, 
                qt=qt, st=st, rt=rt, dt=dt, uu=uu
            )
            
            # Update rolling histories
            release_history.append(rt)
            demand_history.append(dt)
            inflow_history.append(qt)
            
            # Calculate allocation_prev (30-day rolling average)
            if len(release_history) >= 30:
                allocation_prev = np.array(release_history[-30:]).clip(max=10) / np.array(demand_history[-30:])
                allocation_prev = np.mean(np.clip(allocation_prev, 0.0, 1.0))
            
            # Increment timestep
            t += 1
    
    print(f"✓ Simulation complete ({t} timesteps)")
    
    # Convert decision record to DataFrame
    decision_df = pd.DataFrame(decision_record)
    
    return R1.record, decision_df


def main():
    """Run the MLP-based simulation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run MLP-based reservoir simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Run historical simulation (1996-2016):
  python mlp_simulate.py
  # Run historical simulation with fixed TOCS:
  python mlp_simulate.py --fix-tocs 
  
  # Run paleo simulation (1900-1999):
  python mlp_simulate.py --config folsom_paleo --start-year 1900 --end-year 1999
  
Note: Paleo data covers water years 1900-1999 (starts Oct 1, 1899)
      Historical data covers water years 1905-2016 (starts Oct 1, 1904)
        """)
    
    parser.add_argument('--config', type=str, default='folsom_hist',
                        choices=['folsom_hist', 'folsom_paleo'],
                        help='Configuration to use (default: folsom_hist)')
    parser.add_argument('--start-year', type=int, default=1996,
                        help='Starting water year (default: 1996)')
    parser.add_argument('--end-year', type=int, default=2016,
                        help='Ending water year (default: 2016)')
    parser.add_argument('--starting-storage', type=float, default=466.1,
                        help='Initial storage in TAF (default: 466.1)')
    parser.add_argument('--fix-tocs', action='store_true', default=False,
                        help='Fix TOCS to static 365 dowy (default: False)')
    
    args = parser.parse_args()
    
    # Run simulation
    simulation_df, decision_df = run_simulation(
        start_year=args.start_year,
        end_year=args.end_year,
        starting_storage=args.starting_storage,
        fix_tocs=args.fix_tocs,
        config_name=args.config
    )
    
    # Save outputs
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    start_year = args.start_year
    end_year = args.end_year
    config_suffix = args.config.replace('folsom_', '')
    
    tocs_suffix = 'fixtocs_' if args.fix_tocs else ''

    simulation_file = f'./output/mlp_simulation_output_{config_suffix}_{tocs_suffix}{start_year}_{end_year}.csv'
    decision_file = f'./output/mlp_decision_output_{config_suffix}_{tocs_suffix}{start_year}_{end_year}.csv'

    simulation_df.to_csv(simulation_file, index=False)
    print(f"✓ Simulation output saved to: {simulation_file}")
    print(f"  Shape: {simulation_df.shape}")
    
    decision_df.to_csv(decision_file, index=False)
    print(f"✓ Decision output saved to: {decision_file}")
    print(f"  Shape: {decision_df.shape}")
    
    # Calculate and display summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    # Calculate actual allocation (release / demand)
    simulation_df['allocation'] = simulation_df['rt'] / simulation_df['dt']
    simulation_df['allocation'] = simulation_df['allocation'].clip(upper=1.0)
    
    # Overall statistics
    print(f"\nOverall Performance:")
    print(f"  Mean allocation: {simulation_df['allocation'].mean():.3f}")
    print(f"  Min allocation: {simulation_df['allocation'].min():.3f}")
    print(f"  Max allocation: {simulation_df['allocation'].max():.3f}")
    print(f"  Shortage days (allocation < 1.0): {(simulation_df['allocation'] < 1.0).sum()}")
    print(f"  Full allocation days: {(simulation_df['allocation'] >= 0.99).sum()}")
    
    # Storage statistics
    print(f"\nStorage Performance:")
    print(f"  Mean storage: {simulation_df['st'].mean():.1f} TAF")
    print(f"  Min storage: {simulation_df['st'].min():.1f} TAF")
    print(f"  Max storage: {simulation_df['st'].max():.1f} TAF")
    print(f"  Final storage: {simulation_df['st'].iloc[-1]:.1f} TAF")
    
    # Monthly decision statistics
    print(f"\nMonthly Decisions:")
    print(f"  Mean allocation decision: {decision_df['allocation_decision'].mean():.3f}")
    print(f"  Min allocation decision: {decision_df['allocation_decision'].min():.3f}")
    print(f"  Max allocation decision: {decision_df['allocation_decision'].max():.3f}")
    print(f"  Shortage months (< 100%): {(decision_df['allocation_decision'] < 1.0).sum()}")
    
    print("\n" + "="*60)
    print("Simulation completed successfully!")
    print("="*60)



if __name__ == "__main__":
    main()
