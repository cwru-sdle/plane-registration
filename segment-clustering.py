# %%
'''Imports'''
import tslearn
import tslearn.preprocessing # Needed
import tslearn.clustering # Needed
import sklearn
import matplotlib
import pyarrow.parquet as pq
import numpy as np
import os
import polars as pl
import warnings
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import argparse

warnings.filterwarnings("ignore", message=".*force_all_finite.*")

parquet_directory = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_parquet/"
parquet_files = [
    f for f in os.listdir(parquet_directory)
    if f.endswith(".parquet")
]
full_paths = [os.path.join(parquet_directory, f) for f in parquet_files]


parser = argparse.ArgumentParser(description="Time series clustering with parquet segments")
parser.add_argument("--n", type=int, default=10000, help="Number of segments to process")
parser.add_argument("--knn_k", type=int, default=5, help="Number of clusters (k)")

args = parser.parse_args()

n = args.n
knn_k = args.knn_k

# %%
'''Create Generator'''
def segment_df(df: pl.DataFrame) -> list[pl.DataFrame]:
    df = df.with_columns([
        pl.col('state0').shift(1).alias('state0_prev'),
    ])
    df = df.with_columns([
        ((pl.col('state0_prev') == 0) & (pl.col('state0') == 1)).alias('is_start'),
        ((pl.col('state0_prev') == 1) & (pl.col('state0') == 0)).alias('is_end'),
    ])

    # Get indices of starts and ends
    starts = df.filter(pl.col('is_start')).select(pl.col('is_start').arg_true()).to_series().to_list()
    ends = df.filter(pl.col('is_end')).select(pl.col('is_end').arg_true()).to_series().to_list()

    # Handle case where segment started but didn’t end in this batch
    segments = []
    for start in starts:
        # Find the first end after this start
        end = next((e for e in ends if e > start), None)
        if end:
            segments.append(df[start:end].select(['sensor0', 'sensor1']))

    return segments

def load_parquet_segments(full_paths):
    for path in full_paths:
        stream_file = pq.ParquetFile(path)
        for group_index in range(stream_file.num_row_groups):
            df = pl.from_arrow(stream_file.read_row_group(group_index))
            for segment in segment_df(df):
                yield segment

# %%

X_train = next(load_parquet_segments(full_paths))
X_train = tslearn.preprocessing.TimeSeriesScalerMeanVariance().fit_transform(X_train)

model = tslearn.clustering.TimeSeriesKMeans(n_clusters=knn_k, metric="dtw", max_iter=10, random_state=0)
model.fit(X_train)
centroids = model.cluster_centers_

# Step 2: Sequentially refine centroids manually
for segment in tqdm(itertools.islice(load_parquet_segments(full_paths), n), total=n):
    X_train = tslearn.preprocessing.TimeSeriesScalerMeanVariance().fit_transform(segment)
    
    # Combine centroids with the new batch
    combined = np.concatenate((centroids, X_train), axis=0)
    
    # Re-cluster using previous centroids as initialization
    model = tslearn.clustering.TimeSeriesKMeans(n_clusters=knn_k, metric="dtw", max_iter=10,
                             init=centroids, n_init=1, random_state=0)
    model.fit(combined)
    centroids = model.cluster_centers_

# %%
labels = model.predict(X_train)

n_clusters = model.n_clusters
fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 3 * n_clusters))

if n_clusters == 1:
    axes = [axes]

for cluster_idx in range(n_clusters):
    ax = axes[cluster_idx]
    
    # Select time series belonging to this cluster
    cluster_ts = X_train[labels == cluster_idx]
    
    # Plot some examples (up to 5)
    for ts in cluster_ts[:5]:
        ax.plot(ts.squeeze(), color='gray', alpha=0.3)
    
    # Plot the cluster centroid
    centroid = model.cluster_centers_[cluster_idx]
    ax.plot(centroid.squeeze(), color='red', linewidth=2, label='Centroid')
    
    ax.set_title(f'Cluster {cluster_idx} (n={len(cluster_ts)})')
    ax.legend()

plt.tight_layout()
plt.show()
