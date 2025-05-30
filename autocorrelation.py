# %%
'''Imports for Autocorrelation Analysis'''
import polars as pl
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import signal
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# %%
'''Load and Prepare Data'''
def load_sensor_data(parquet_path):
    """Load parquet file and prepare for time series analysis"""
    df = pl.read_parquet(parquet_path)
    
    # Sort by time to ensure proper ordering
    df = df.sort(['session', 'config', 'job', 'layer', 't'])
    
    # Create grouping columns for analysis
    df = df.with_columns([
        pl.concat_str([
            pl.col('session'), 
            pl.col('config'), 
            pl.col('job')
        ], separator='_').alias('group_id'),
        pl.concat_str([
            pl.col('session'), 
            pl.col('config'), 
            pl.col('job'),
            pl.col('layer')
        ], separator='_').alias('layer_group_id')
    ])
    
    return df

# %%
'''Autocorrelation Functions'''
def compute_autocorrelation(series, max_lag=None):
    """Compute autocorrelation function for a time series"""
    series = np.array(series)
    
    # Remove NaN values
    series = series[~np.isnan(series)]
    
    if len(series) < 10:  # Need minimum data points
        return np.array([]), np.array([])
    
    # Set max_lag if not provided
    if max_lag is None:
        max_lag = min(len(series) // 4, 50)  # Max 50 lags or 1/4 of series
    
    max_lag = min(max_lag, len(series) - 1)
    
    # Standardize the series
    series_mean = np.mean(series)
    series_std = np.std(series)
    
    if series_std == 0:
        return np.arange(max_lag + 1), np.zeros(max_lag + 1)
    
    series_norm = (series - series_mean) / series_std
    
    # Compute autocorrelation using numpy correlate
    autocorr = np.correlate(series_norm, series_norm, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
    autocorr = autocorr[:max_lag + 1]  # Trim to max_lag
    autocorr = autocorr / autocorr[0]  # Normalize
    
    lags = np.arange(len(autocorr))
    
    return lags, autocorr

def compute_partial_autocorr(series, max_lag=20):
    """Compute partial autocorrelation using Yule-Walker equations"""
    series = np.array(series)
    series = series[~np.isnan(series)]
    
    if len(series) < max_lag + 5:
        return np.array([]), np.array([])
    
    # Standardize
    series = (series - np.mean(series)) / np.std(series)
    
    pacf_values = []
    
    for k in range(1, max_lag + 1):
        if k == 1:
            # Lag-1 is just regular correlation
            pacf_k = np.corrcoef(series[:-1], series[1:])[0, 1]
        else:
            # Solve Yule-Walker equations
            r = [np.corrcoef(series[:-i], series[i:])[0, 1] for i in range(1, k + 1)]
            R = np.array([[r[abs(i-j)] if abs(i-j) < len(r) else 0 
                          for j in range(k)] for i in range(k)])
            
            try:
                phi = np.linalg.solve(R, r)
                pacf_k = phi[-1]
            except np.linalg.LinAlgError:
                pacf_k = 0
        
        pacf_values.append(pacf_k if not np.isnan(pacf_k) else 0)
    
    return np.arange(1, max_lag + 1), np.array(pacf_values)

# %%
'''Group-wise Analysis Functions'''
def analyze_group_autocorrelation(df, group_col='group_id', sensor_cols=['sensor0', 'sensor1'], 
                                time_col='t', max_lag=30):
    """Analyze autocorrelation for each group"""
    results = []
    
    groups = df.select(group_col).unique().to_pandas()[group_col].tolist()
    
    for group in groups:
        group_data = df.filter(pl.col(group_col) == group).sort(time_col)
        
        # Skip if too few data points
        if len(group_data) < 20:
            continue
            
        group_result = {
            'group_id': group,
            'n_points': len(group_data),
            'time_span': group_data.select(pl.col(time_col).max() - pl.col(time_col).min()).item(),
        }
        
        # Analyze each sensor
        for sensor in sensor_cols:
            if sensor in group_data.columns:
                sensor_data = group_data.select(sensor).to_numpy().flatten()
                
                # Basic statistics
                group_result[f'{sensor}_mean'] = np.nanmean(sensor_data)
                group_result[f'{sensor}_std'] = np.nanstd(sensor_data)
                group_result[f'{sensor}_range'] = np.nanmax(sensor_data) - np.nanmin(sensor_data)
                
                # Autocorrelation analysis
                lags, autocorr = compute_autocorrelation(sensor_data, max_lag)
                
                if len(autocorr) > 1:
                    group_result[f'{sensor}_lag1_autocorr'] = autocorr[1] if len(autocorr) > 1 else 0
                    group_result[f'{sensor}_max_autocorr'] = np.max(autocorr[1:]) if len(autocorr) > 1 else 0
                    group_result[f'{sensor}_max_autocorr_lag'] = lags[1:][np.argmax(autocorr[1:])] if len(autocorr) > 1 else 0
                    
                    # Significant autocorrelation test (rough approximation)
                    n = len(sensor_data)
                    confidence_bound = 1.96 / np.sqrt(n)  # 95% confidence for white noise
                    significant_lags = np.sum(np.abs(autocorr[1:]) > confidence_bound)
                    group_result[f'{sensor}_significant_lags'] = significant_lags
                    group_result[f'{sensor}_is_structured'] = autocorr[1] > confidence_bound
                else:
                    group_result[f'{sensor}_lag1_autocorr'] = 0
                    group_result[f'{sensor}_max_autocorr'] = 0
                    group_result[f'{sensor}_max_autocorr_lag'] = 0
                    group_result[f'{sensor}_significant_lags'] = 0
                    group_result[f'{sensor}_is_structured'] = False
        
        results.append(group_result)
    
    return pl.DataFrame(results)

# %%
'''Visualization Functions'''
def plot_autocorrelation_summary(results_df, sensor_cols=['sensor0', 'sensor1']):
    """Create summary plots of autocorrelation analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sensor Autocorrelation Analysis Summary', fontsize=16)
    
    for i, sensor in enumerate(sensor_cols):
        if f'{sensor}_lag1_autocorr' not in results_df.columns:
            continue
            
        # Convert to pandas for easier plotting
        results_pd = results_df.to_pandas()
        
        # Lag-1 autocorrelation distribution
        ax = axes[i, 0]
        lag1_autocorr = results_pd[f'{sensor}_lag1_autocorr'].dropna()
        ax.hist(lag1_autocorr, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(np.mean(lag1_autocorr), color='green', linestyle='-', linewidth=2)
        ax.set_xlabel('Lag-1 Autocorrelation')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{sensor}: Lag-1 Autocorrelation Distribution\nMean: {np.mean(lag1_autocorr):.3f}')
        
        # Maximum autocorrelation vs lag
        ax = axes[i, 1]
        max_autocorr = results_pd[f'{sensor}_max_autocorr'].dropna()
        max_lag = results_pd[f'{sensor}_max_autocorr_lag'].dropna()
        scatter = ax.scatter(max_lag, max_autocorr, alpha=0.6, c=results_pd['n_points'], 
                           cmap='viridis', s=30)
        ax.set_xlabel('Lag of Maximum Autocorrelation')
        ax.set_ylabel('Maximum Autocorrelation')
        ax.set_title(f'{sensor}: Max Autocorr vs Lag')
        plt.colorbar(scatter, ax=ax, label='N Points')
        
        # Structure detection summary
        ax = axes[i, 2]
        structured = results_pd[f'{sensor}_is_structured'].sum()
        total = len(results_pd)
        unstructured = total - structured
        
        ax.pie([structured, unstructured], 
               labels=[f'Structured ({structured})', f'Unstructured ({unstructured})'],
               autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{sensor}: Structure Detection\n({structured}/{total} groups show structure)')
    
    plt.tight_layout()
    return fig

def plot_individual_autocorrelations(df, group_col='group_id', sensor_cols=['sensor0', 'sensor1'], 
                                   time_col='t', n_examples=6, max_lag=30):
    """Plot autocorrelation functions for example groups"""
    groups = df.select(group_col).unique().to_pandas()[group_col].tolist()[:n_examples]
    
    fig, axes = plt.subplots(len(groups), len(sensor_cols), 
                            figsize=(6*len(sensor_cols), 4*len(groups)))
    
    if len(groups) == 1:
        axes = axes.reshape(1, -1)
    if len(sensor_cols) == 1:
        axes = axes.reshape(-1, 1)
        
    fig.suptitle('Individual Autocorrelation Functions', fontsize=16)
    
    for i, group in enumerate(groups):
        group_data = df.filter(pl.col(group_col) == group).sort(time_col)
        
        for j, sensor in enumerate(sensor_cols):
            ax = axes[i, j]
            
            if sensor in group_data.columns:
                sensor_data = group_data.select(sensor).to_numpy().flatten()
                lags, autocorr = compute_autocorrelation(sensor_data, max_lag)
                
                if len(autocorr) > 0:
                    ax.plot(lags, autocorr, 'b-', linewidth=2, label='Autocorrelation')
                    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    
                    # Add confidence bounds
                    n = len(sensor_data)
                    confidence_bound = 1.96 / np.sqrt(n)
                    ax.axhline(y=confidence_bound, color='r', linestyle=':', alpha=0.7, label='95% Confidence')
                    ax.axhline(y=-confidence_bound, color='r', linestyle=':', alpha=0.7)
                    
                    ax.set_xlabel('Lag')
                    ax.set_ylabel('Autocorrelation')
                    ax.set_title(f'Group: {group[:20]}...\n{sensor} (N={len(sensor_data)})')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, 'Insufficient Data', transform=ax.transAxes, 
                           ha='center', va='center')
                    ax.set_title(f'Group: {group[:20]}...\n{sensor}')
    
    plt.tight_layout()
    return fig

# %%
'''Main Analysis Function'''
def run_autocorrelation_analysis(parquet_path, output_dir=None):
    """Main function to run complete autocorrelation analysis"""
    
    print("Loading data...")
    df = load_sensor_data(parquet_path)
    print(f"Loaded {len(df)} data points from {df.select('group_id').unique().height} groups")
    
    print("Computing autocorrelations...")
    results = analyze_group_autocorrelation(df, max_lag=30)
    print(f"Analyzed {len(results)} groups")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("AUTOCORRELATION ANALYSIS SUMMARY")
    print("="*50)
    
    for sensor in ['sensor0', 'sensor1']:
        if f'{sensor}_lag1_autocorr' in results.columns:
            lag1_mean = results.select(f'{sensor}_lag1_autocorr').mean().item()
            structured_count = results.select(f'{sensor}_is_structured').sum().item()
            total_groups = len(results)
            
            print(f"\n{sensor.upper()}:")
            print(f"  Mean lag-1 autocorrelation: {lag1_mean:.4f}")
            print(f"  Groups with significant structure: {structured_count}/{total_groups} ({100*structured_count/total_groups:.1f}%)")
            
            if lag1_mean > 0.3:
                print(f"  → Strong temporal structure detected")
            elif lag1_mean > 0.1:
                print(f"  → Moderate temporal structure detected")
            else:
                print(f"  → Weak temporal structure (likely noisy)")
    
    print("\n" + "="*50)
    
    # Create visualizations
    print("Creating visualizations...")
    fig1 = plot_autocorrelation_summary(results)
    fig2 = plot_individual_autocorrelations(df)
    
    # Save results if output directory provided
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results.write_csv(Path(output_dir) / "autocorrelation_results.csv")
        fig1.savefig(Path(output_dir) / "autocorrelation_summary.png", dpi=300, bbox_inches='tight')
        fig2.savefig(Path(output_dir) / "individual_autocorrelations.png", dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}")
    
    plt.show()
    
    return results, df

# %%
'''Usage Example'''
if __name__ == "__main__":
    # Example usage - modify path as needed
    parquet_directory = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_parquet/"
    parquet_paths = [parquet_directory+file for file in os.listdir(parquet_directory)]
    output_dir = os.getcwd()+"/data/pyrometer/"
    
    # Run analysis
    results, data = run_autocorrelation_analysis(parquet_paths[0], output_dir)
    
    # Additional analysis examples:
    
    # 1. Find groups with strongest autocorrelation
    print("\nTop 5 groups with strongest sensor0 autocorrelation:")
    top_groups = results.sort('sensor0_lag1_autocorr', descending=True).head(5)
    print(top_groups.select(['group_id', 'sensor0_lag1_autocorr', 'n_points']))
    
    # 2. Compare sensor0 vs sensor1 autocorrelation
    if 'sensor1_lag1_autocorr' in results.columns:
        correlation = results.select([
            pl.corr('sensor0_lag1_autocorr', 'sensor1_lag1_autocorr').alias('sensor_correlation')
        ]).item()
        print(f"\nCorrelation between sensor0 and sensor1 autocorrelations: {correlation:.3f}")
    
    # 3. Analyze by group characteristics
    print("\nAutocorrelation vs Group Size:")
    size_groups = results.with_columns([
        pl.when(pl.col('n_points') < 100).then(pl.lit('Small'))
        .when(pl.col('n_points') < 500).then(pl.lit('Medium'))
        .otherwise(pl.lit('Large')).alias('size_category')
    ]).group_by('size_category').agg([
        pl.col('sensor0_lag1_autocorr').mean().alias('mean_autocorr'),
        pl.len().alias('count')
    ])
    print(size_groups)