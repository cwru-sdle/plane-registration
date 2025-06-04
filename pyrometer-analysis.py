# %%
import multiprocessing
import pyarrow.parquet as pq
import os
import polars as pl
from tqdm import tqdm
import shared # Realtive import
import numpy as np
import scipy
import sklearn

def autocovariance(signal: np.ndarray, max_lag: int = None) -> np.ndarray:
    n = len(signal)
    if max_lag is None or max_lag >= n:
        max_lag = n - 1

    mean = np.mean(signal)
    autocov = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        # Covariance between signal[:-lag] and signal[lag:]
        autocov[lag] = np.sum((signal[:n - lag] - mean) * (signal[lag:] - mean)) / n
    return autocov

def expoential_fit(signal: np.ndarray) -> tuple[float, float, float]:
    initial_guess = [np.mean(signal), -.1]

    def fit_func(x,a,b):
        x = np.asarray(x)
        return a - np.exp(x / b)

    time_steps = np.arange(len(signal)) * 10

    params_opt, _ = scipy.optimize.curve_fit(fit_func, time_steps, signal, p0=initial_guess,bounds=[[0,-1000],[np.inf,-.000001]])
    x_fit, y_fit = params_opt
    y_pred = fit_func(time_steps, x_fit, y_fit)

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains NaN or Inf")

    mse = sklearn.metrics.mean_squared_error(signal, y_pred)
    return mse, x_fit, y_fit

def compute_segment_features(signal: np.ndarray) -> dict:
    assert not np.isnan(signal).any()
    assert not np.isinf(signal).any()
    assert len(signal) > 2
    
    cov = autocovariance(signal, max_lag=10)
    mse, x_intercept, y_intercept = expoential_fit(signal)
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'cv': scipy.stats.variation(signal),
        'slope': np.polyfit(np.arange(len(signal)), signal, deg=1)[0],
        'dominant_freq': np.argmax(np.abs(scipy.fft.rfft(signal))),
        'autoco_1': cov[1],
        'autoco_2': cov[2],
        'autoco_3': cov[3],
        'autoco_4': cov[4],
        'autoco_5': cov[5],
        'autoco_6': cov[6],
        'autoco_7': cov[7],
        'autoco_8': cov[8],
        'autoco_9': cov[9],
        'mse': mse,
        "x_intercept":x_intercept,
        "y_intercept":y_intercept
    }
# %%
def worker_process(args) -> dict:
    file, row_group = args
    df = pl.from_arrow(pq.ParquetFile(file).read_row_group(row_group))
    segment_list = shared.extract_segments(df,as_numpy=False,grouping="0-1")

    features = []
    for segment in segment_list:
        for sensor in segment.iter_columns():
            seg_np = sensor.to_numpy()
            feats = compute_segment_features(seg_np)
            feats['segment_length'] = len(seg_np)
            features.append(feats)

    return {
        'file': file,
        'row_group': row_group,
        'segments_count': len(segment_list),
        'segment_features': features
    }

with multiprocessing.Pool(processes=19) as pool:
    results = list(tqdm(
        pool.imap(worker_process, shared.path_and_groups),
        total=len(shared.path_and_groups),
        desc="Processing row groups"
    ))
# %%
flat_features = []

for result in results:
    file = result['file']
    row_group = result['row_group']
    for feat in result['segment_features']:
        flat_features.append({
            'file': file,
            'row_group': row_group,
            **feat
        })

df = pl.DataFrame(flat_features)
df.write_csv(shared.output_directory + "segment_summary-0-1.csv")