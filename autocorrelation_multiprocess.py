# %%
import multiprocessing
import pyarrow.parquet as pq
import os
import polars as pl
from . import parquet_paths
# %%

def worker_process(file:str,row_group:int)-> float: 
    df = pl.from_arrow(pq.ParquetFile(file).read_row_group(row_group))
    
# %%
all_tasks = []
for path in parquet_paths:
    pf = pq.ParquetFile(path)
    n_groups = pf.num_row_groups
    for group in range(n_groups):
        all_tasks.append((path, group))
        
with Pool(processes=4) as pool:  # Limit to 4 concurrent processes
    pool.starmap(worker_process,all_tasks)