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
'''Load and Prepare Data with Error Handling'''
def validate_parquet_file(parquet_path):
    """Validate if a parquet file is readable and not corrupted"""
    try:
        # Check file size first
        file_size = os.path.getsize(parquet_path)
        if file_size < 12:  # Minimum size for parquet header/footer
            return False, f"File too small: {file_size} bytes"
        
        # Try to read just the schema without loading data
        try:
            schema = pl.scan_parquet(parquet_path).schema
            return True, "Valid"
        except Exception as e:
            return False, f"Schema read error: {str(e)[:100]}"
            
    except Exception as e:
        return False, f"File access error: {str(e)[:100]}"

def load_sensor_data(parquet_path):
    """Load parquet file and prepare for time series analysis with error handling"""
    try:
        # Validate file first
        is_valid, message = validate_parquet_file(parquet_path)
        if not is_valid:
            raise ValueError(f"Invalid parquet file: {message}")
        
        print(f"Loading: {os.path.basename(parquet_path)}")
        df = pl.read_parquet(parquet_path)
        
        if len(df) == 0:
            raise ValueError("Empty dataframe")
        
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Check for required columns
        required_cols = ['session', 'config', 'job', 'layer', 't']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  Warning: Missing expected columns: {missing_cols}")
            # Create dummy columns if missing
            for col in missing_cols:
                if col == 't':
                    df = df.with_row_count('t')  # Use row index as time
                else:
                    df = df.with_columns(pl.lit(f"unknown_{col}").alias(col))
        
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
        
    except Exception as e:
        print(f"Error loading {parquet_path}: {str(e)}")
        return None

def get_sensor_columns(df):
    """Identify sensor columns in the dataframe"""
    # Look for columns that might be sensors
    potential_sensors = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['sensor', 'pyrometer', 'temp', 'signal']):
            potential_sensors.append(col)
        elif col.startswith(('sensor', 'pyro', 'temp')):
            potential_sensors.append(col)
    
    # If no obvious sensor columns, look for numeric columns (excluding metadata)
    if not potential_sensors:
        metadata_cols = {'session', 'config', 'job', 'layer', 't', 'group_id', 'layer_group_id'}
        for col in df.columns:
            if col not in metadata_cols and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                potential_sensors.append(col)
    
    return potential_sensors[:10]  # Limit to first 10 sensor columns

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

def compute_autocorrelation_fast(series, max_lag=None):
    """Optimized autocorrelation computation for large datasets"""
    series = np.array(series, dtype=np.float64)
    
    # Remove NaN values
    valid_mask = ~np.isnan(series)
    series = series[valid_mask]
    
    if len(series) < 10:
        return np.array([]), np.array([])
    
    # Set max_lag more conservatively for large datasets
    if max_lag is None:
        max_lag = min(len(series) // 10, 100)  # Max 100 lags or 1/10 of series
    
    max_lag = min(max_lag, len(series) - 1, 200)  # Hard limit at 200 lags
    
    # Standardize the series
    series_mean = np.mean(series)
    series_std = np.std(series)
    
    if series_std == 0:
        return np.arange(max_lag + 1), np.zeros(max_lag + 1)
    
    series_norm = (series - series_mean) / series_std
    
    # Use scipy's correlate for better performance on large arrays
    from scipy.signal import correlate
    
    # Compute full correlation
    full_corr = correlate(series_norm, series_norm, mode='full')
    
    # Extract autocorrelations for positive lags
    mid_point = len(full_corr) // 2
    autocorr = full_corr[mid_point:mid_point + max_lag + 1]
    autocorr = autocorr / autocorr[0]  # Normalize by lag-0
    
    lags = np.arange(len(autocorr))
    
    return lags, autocorr

# %%
'''Group-wise Analysis Functions'''
def analyze_group_autocorrelation(df, group_col='group_id', sensor_cols=None, 
                                time_col='t', max_lag=30, sample_size=100000):
    """Analyze autocorrelation for each group with optimizations for large datasets"""
    if sensor_cols is None:
        sensor_cols = get_sensor_columns(df)
        print(f"Auto-detected sensor columns: {sensor_cols}")
    
    results = []
    
    groups = df.select(group_col).unique().to_pandas()[group_col].tolist()
    
    for group in groups:
        print(f"Processing group: {group}")
        group_data = df.filter(pl.col(group_col) == group).sort(time_col)
        
        n_points = len(group_data)
        print(f"  Group size: {n_points:,} points")
        
        # Skip if too few data points
        if n_points < 20:
            continue
        
        # For very large datasets, use sampling
        if n_points > sample_size:
            print(f"  Large dataset detected. Using sampling approach...")
            sample_indices = np.random.choice(n_points, sample_size, replace=False)
            sample_indices = np.sort(sample_indices)  # Keep temporal order
            sampled_data = group_data[sample_indices]
        else:
            sampled_data = group_data
            
        group_result = {
            'group_id': group,
            'n_points': n_points,
            'time_span': group_data.select(pl.col(time_col).max() - pl.col(time_col).min()).item(),
            'analysis_method': 'sampled' if n_points > sample_size else 'full'
        }
        
        # Analyze each sensor
        for sensor in sensor_cols:
            if sensor in group_data.columns:
                print(f"    Analyzing {sensor}...")
                
                # Get sensor data (sampled if needed)
                if n_points > sample_size:
                    sensor_data = sampled_data.select(sensor).to_numpy().flatten()
                else:
                    sensor_data = group_data.select(sensor).to_numpy().flatten()
                
                # Basic statistics on full dataset
                full_sensor_data = group_data.select(sensor).to_numpy().flatten()
                group_result[f'{sensor}_mean'] = np.nanmean(full_sensor_data)
                group_result[f'{sensor}_std'] = np.nanstd(full_sensor_data)
                group_result[f'{sensor}_range'] = np.nanmax(full_sensor_data) - np.nanmin(full_sensor_data)
                
                # Fast autocorrelation on sampled data
                print(f"      Computing autocorrelation on {len(sensor_data):,} points...")
                lags, autocorr = compute_autocorrelation_fast(sensor_data, max_lag)
                
                if len(autocorr) > 1:
                    group_result[f'{sensor}_lag1_autocorr'] = autocorr[1]
                    group_result[f'{sensor}_max_autocorr'] = np.max(autocorr[1:])
                    group_result[f'{sensor}_max_autocorr_lag'] = lags[1:][np.argmax(autocorr[1:])]
                    
                    # Significance test
                    n_sample = len(sensor_data)
                    confidence_bound = 1.96 / np.sqrt(n_sample)
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
        print(f"  Completed group analysis\n")
    
    return pl.DataFrame(results)

# %%
'''File Processing Functions'''
def get_valid_parquet_files(directory):
    """Get list of valid parquet files from directory"""
    parquet_files = []
    invalid_files = []
    
    print(f"Scanning directory: {directory}")
    
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            filepath = os.path.join(directory, filename)
            is_valid, message = validate_parquet_file(filepath)
            
            if is_valid:
                parquet_files.append(filepath)
                print(f"✓ Valid: {filename}")
            else:
                invalid_files.append((filepath, message))
                print(f"✗ Invalid: {filename} - {message}")
    
    print(f"\nFound {len(parquet_files)} valid files, {len(invalid_files)} invalid files")
    
    if invalid_files:
        print("\nInvalid files:")
        for filepath, reason in invalid_files:
            print(f"  {os.path.basename(filepath)}: {reason}")
    
    return parquet_files

# %%
'''Main Analysis Function'''
def run_autocorrelation_analysis(parquet_path, output_dir=None):
    """Main function to run complete autocorrelation analysis"""
    
    print("Loading data...")
    df = load_sensor_data(parquet_path)
    
    if df is None:
        print(f"Failed to load data from {parquet_path}")
        return None, None
    
    print(f"Loaded {len(df)} data points from {df.select('group_id').unique().height} groups")
    
    # Get sensor columns
    sensor_cols = get_sensor_columns(df)
    if not sensor_cols:
        print("No sensor columns found!")
        return None, None
    
    print(f"Analyzing sensors: {sensor_cols}")
    
    print("Computing autocorrelations...")
    results = analyze_group_autocorrelation(df, sensor_cols=sensor_cols, max_lag=50, sample_size=100000)
    print(f"Analyzed {len(results)} groups")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("AUTOCORRELATION ANALYSIS SUMMARY")
    print("="*50)
    
    for sensor in sensor_cols:
        if f'{sensor}_lag1_autocorr' in results.columns:
            lag1_values = results.select(f'{sensor}_lag1_autocorr').to_numpy().flatten()
            lag1_values = lag1_values[~np.isnan(lag1_values)]  # Remove NaN values
            
            if len(lag1_values) > 0:
                lag1_mean = np.mean(lag1_values)
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
    
    # Save results if output directory provided
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.splitext(os.path.basename(parquet_path))[0]
        results.write_csv(Path(output_dir) / f"autocorrelation_results_{filename}.csv")
        print(f"Results saved to {output_dir}")
    
    return results, df

# %%
'''Usage Example'''
if __name__ == "__main__":
    # Example usage - modify path as needed
    parquet_directory = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_parquet/"
    output_dir = os.getcwd() + "/data/pyrometer/"
    
    # Get valid parquet files
    parquet_paths = get_valid_parquet_files(parquet_directory)
    
    if not parquet_paths:
        print("No valid parquet files found!")
        exit(1)
    
    # Run analysis on each valid file
    all_results = []
    
    for path in parquet_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(path)}")
        print(f"{'='*60}")
        
        try:
            results, data = run_autocorrelation_analysis(path, output_dir)
            
            if results is not None:
                # Add filename to results
                results = results.with_columns(
                    pl.lit(os.path.basename(path)).alias('source_file')
                )
                all_results.append(results)
                
                # Print some quick stats for this file
                print(f"\nQuick stats for {os.path.basename(path)}:")
                sensor_cols = get_sensor_columns(data) if data is not None else []
                for sensor in sensor_cols[:2]:  # Just show first 2 sensors
                    if f'{sensor}_lag1_autocorr' in results.columns:
                        autocorr_values = results.select(f'{sensor}_lag1_autocorr').to_numpy().flatten()
                        autocorr_values = autocorr_values[~np.isnan(autocorr_values)]
                        if len(autocorr_values) > 0:
                            print(f"  {sensor}: mean autocorr = {np.mean(autocorr_values):.3f}")
            else:
                print(f"Failed to analyze {path}")
                
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    
    # Combine all results
    if all_results:
        print(f"\n{'='*60}")
        print("COMBINED ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        combined_results = pl.concat(all_results)
        combined_results.write_csv(Path(output_dir) / "combined_autocorrelation_results.csv")
        
        print(f"Combined results from {len(all_results)} files")
        print(f"Total groups analyzed: {len(combined_results)}")
        print(f"Results saved to: {output_dir}/combined_autocorrelation_results.csv")
    else:
        print("No results to combine - all files failed to process")