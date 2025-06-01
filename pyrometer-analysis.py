# %%
import multiprocessing
import pyarrow.parquet as pq
import os
import polars as pl
from tqdm import tqdm
import shared # Realtive import

import numpy as np
from scipy.stats import variation
from scipy.fft import rfft

def compute_segment_features(signal: np.ndarray) -> dict:
    assert not np.isnan(signal).any()
    assert not np.isinf(signal).any()
    assert len(signal) > 2

    mean = np.mean(signal)
    std = np.std(signal)
    slope = np.polyfit(np.arange(len(signal)), signal, deg=1)[0]
    dominant_freq = np.argmax(np.abs(rfft(signal)))

    return {
        'mean': mean,
        'std': std,
        'cv': variation(signal),
        'slope': slope,
        'dominant_freq': dominant_freq,
    }
# %%
def worker_process(args) -> dict:
    file, row_group = args
    df = pl.from_arrow(pq.ParquetFile(file).read_row_group(row_group))
    segment_list = shared.extract_segments(df,as_numpy=False)

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


with multiprocessing.Pool(processes=29) as pool:
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
df.write_csv(shared.output_directory + "segment_summary.csv")