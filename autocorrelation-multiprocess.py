# %%
import multiprocessing
import pyarrow.parquet as pq
import os
import polars as pl
import shared # Realtive import
# %%

def worker_process(file:str,row_group:int)-> float: 
    df = pl.from_arrow(pq.ParquetFile(file).read_row_group(row_group))
    
    
# %%
with multiprocessing.Pool(processes=4) as pool:  # Limit to 4 concurrent processes
    pool.starmap(worker_process,shared.path_and_groups)