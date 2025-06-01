import os
import pyarrow.parquet as pq
import numpy as np
import polars as pl
import typing

# To allow relative imports
parquet_directory = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_parquet/"
output_directory = os.getcwd()+"/data/pyrometer/"
parquet_paths = [
    os.path.join(parquet_directory, file)
    for file in os.listdir(parquet_directory)
    if not file.endswith(".txt")
]
path_and_groups = []
for path in parquet_paths:
    pf = pq.ParquetFile(path)
    n_groups = pf.num_row_groups
    for group in range(n_groups):
        path_and_groups.append((path, group))

def extract_segments(df: pl.DataFrame, as_numpy: bool = False) -> list[typing.Union[pl.DataFrame, np.ndarray]]:
    """
    Extract segments where state0 transitions from 0 to 1 and back to 0.

    Args:
        df (pl.DataFrame): The input Polars DataFrame.
        as_numpy (bool): If True, return segments as NumPy arrays; otherwise as Polars DataFrames.

    Returns:
        list of segments as either pl.DataFrame or np.ndarray
    """
    df = df.with_columns([
        pl.col('state0').shift(1).alias('state0_prev'),
    ])
    df = df.with_columns([
        ((pl.col('state0_prev') == 0) & (pl.col('state0') == 1)).alias('is_start'),
        ((pl.col('state0_prev') == 1) & (pl.col('state0') == 0)).alias('is_end'),
    ])

    df_with_index = df.with_row_index()

    starts = df_with_index.filter(pl.col('is_start')).select('index').to_series().to_list()
    ends = df_with_index.filter(pl.col('is_end')).select('index').to_series().to_list()

    segments = []
    for start in starts:
        end = next((e for e in ends if e > start), None)
        if end:
            if end-start>=10:
                segment = df[start:end].select(['sensor0', 'sensor1'])
                segments.append(segment.to_numpy(structured=True,order='c') if as_numpy else segment) # list[row_indx][column_row]
    return segments