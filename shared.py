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

def extract_segments(
    df: pl.DataFrame, 
    grouping: typing.Literal["0-1", "1-0", "0-0", "1-1"],
    as_numpy: bool = False
) -> typing.List[typing.Union[pl.DataFrame, np.ndarray]]:
    """
    Extract segments based on state transitions in the 'state0' column.

    Args:
        df (pl.DataFrame): The input Polars DataFrame containing a 'state0' column.
        grouping (Literal["0-1", "1-0", "0-0", "1-1"]): Type of transition to detect:
            - "0-1": Extract segments from 0→1 transition to 1→0 transition
            - "1-0": Extract segments from 1→0 transition to 0→1 transition  
            - "0-0": Extract segments where state0 remains 0
            - "1-1": Extract segments where state0 remains 1
        as_numpy (bool): If True, return segments as NumPy arrays; otherwise as Polars DataFrames.

    Returns:
        List[Union[pl.DataFrame, np.ndarray]]: List of segments as either DataFrames or arrays.
        
    Raises:
        ValueError: If grouping parameter is not one of the valid options.
        KeyError: If 'state0' column is not found in the DataFrame.
    """
    # Validate inputs
    if 'state0' not in df.columns:
        raise KeyError("Column 'state0' not found in DataFrame")
    
    if df.is_empty():
        return []
    
    # Add previous state column for transition detection
    df_with_prev = df.with_columns([
        pl.col('state0').shift(1).alias('state0_prev'),
    ])
    
    # Define transition logic based on grouping type
    if grouping == "0-1":
        # Extract segments from 0→1 transition to next 1→0 transition
        start_condition = (pl.col('state0_prev') == 0) & (pl.col('state0') == 1)
        end_condition = (pl.col('state0_prev') == 1) & (pl.col('state0') == 0)
        
    elif grouping == "1-0":
        # Extract segments from 1→0 transition to next 0→1 transition
        start_condition = (pl.col('state0_prev') == 1) & (pl.col('state0') == 0)
        end_condition = (pl.col('state0_prev') == 0) & (pl.col('state0') == 1)
        
    elif grouping == "0-0":
        # Extract segments where state0 remains 0
        start_condition = (pl.col('state0_prev') != 0) & (pl.col('state0') == 0)
        end_condition = (pl.col('state0') == 0) & (pl.col('state0').shift(-1) != 0)
        
    elif grouping == "1-1":
        # Extract segments where state0 remains 1
        start_condition = (pl.col('state0_prev') != 1) & (pl.col('state0') == 1)
        end_condition = (pl.col('state0') == 1) & (pl.col('state0').shift(-1) != 1)
        
    else:
        raise ValueError(f"Invalid grouping '{grouping}'. Must be one of: '0-1', '1-0', '0-0', '1-1'")
    
    # Mark start and end points
    df_marked = df_with_prev.with_columns([
        start_condition.alias('is_start'),
        end_condition.alias('is_end'),
    ])
    
    # Get indices of start and end points
    starts = df_marked.with_row_index().filter(pl.col('is_start'))['index'].to_list()
    ends = df_marked.with_row_index().filter(pl.col('is_end'))['index'].to_list()
    
    segments = []
    for start in starts:
        end = next((e for e in ends if e > start), None)
        if end:
            if end-start>=10:
                segment = df[start:end].select(['sensor0', 'sensor1'])
                segments.append(segment.to_numpy(structured=True,order='c') if as_numpy else segment) # list[row_indx][column_row]
    return segments