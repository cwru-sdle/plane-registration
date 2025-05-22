from pypcd import pypcd
import polars as pl
import pandas as pd
import os

part_1 = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/lpbf-aconity/registeration/session_2025_03_25_16-54-19.812/config_3_67d0704e91000019001f401c/job_1_67e28ebb8f00003c0203e1e7/sensors/ECAMPCDWriterSink__1/layer/"
part_2 = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/lpbf-aconity/registeration/session_2025_03_25_18-52-34.763/config_3_67d0660b8a00007d0127a247/job_8_67e28ebb8f00003c0203e1e7/sensors/ECAMPCDWriterSink__1/layer/"

def get_pcd_paths(directory):
    return sorted([
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.endswith(".pcd")
    ])

paths = get_pcd_paths(part_1) + get_pcd_paths(part_2)

all_polars_dfs = []
all_pandas_dfs = []

for file_path in paths:
    try:
        with open(file_path, "rb") as f:
            layer = int(file_path.split("/")[-1].replace(".pcd","").replace(".",""))
            pc = pypcd.PointCloud.from_fileobj(f)

            # Create Polars DF
            df_polars = pl.DataFrame({field: pc.pc_data[field] for field in pc.fields})
            df_polars = df_polars.with_columns(pl.lit(layer).alias("layer"))
            all_polars_dfs.append(df_polars)

            # Convert Polars DF to Pandas DF
            df_pandas = df_polars.to_pandas()
            all_pandas_dfs.append(df_pandas)

    except AssertionError:
        print(f"Skipping invalid or corrupted file: {file_path}")

# Calculate total memory size for Polars DataFrames
total_polars_bytes = sum(df.estimated_size() for df in all_polars_dfs)

# Calculate total memory size for Pandas DataFrames (in bytes)
total_pandas_bytes = sum(df.memory_usage(deep=True).sum() for df in all_pandas_dfs)

print(f"Total Polars memory size: {total_polars_bytes / (1024**2):.2f} MB")
print(f"Total Pandas memory size: {total_pandas_bytes / (1024**2):.2f} MB")
