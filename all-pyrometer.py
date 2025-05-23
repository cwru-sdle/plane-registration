# %%
'''Imports'''
from pypcd import pypcd
import polars as pl
import os
import re
from multiprocessing import Pool, cpu_count

# %%
'''Get Paths and Compile Regex Pattern'''
log_path = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_log"
output_dir = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_parquet/"
os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(
    r"""
    session_(\d{4}_\d{2}_\d{2}_\d{2}-\d{2}-\d{2}\.\d+)/     # session
    config_(\d+_[a-f0-9]+)/                                 # config
    job_(\d+_[a-f0-9]+)/                                    # job
    sensors/ECAMPCDWriterSink__1/layer/(\d{3}\.\d{3})\.pcd  # layer
    """,
    re.VERBOSE
)

# %%
'''Process a Batch of Files from One Session'''
def process_and_save_session(session, file_args):
    session_safe = session.replace("/", "_")
    output_path = os.path.join(output_dir, f"{session_safe}.parquet")
    
    # Check if output file already exists; skip if yes
    if os.path.exists(output_path):
        print(f"Skipping session {session} as {output_path} already exists.")
        return
    
    dfs = []
    for full_path, _, config, job, layer_str in file_args:
        try:
            with open(full_path, "rb") as f:
                layer = int(layer_str.replace(".", ""))
                pc = pypcd.PointCloud.from_fileobj(f)
                df = pl.DataFrame({field: pc.pc_data[field] for field in pc.fields})
                df = df.with_columns([
                    pl.lit(layer).alias("layer"),
                    pl.lit(session).alias("session"),
                    pl.lit(config).alias("config"),
                    pl.lit(job).alias("job")
                ])
                dfs.append(df)
        except (AssertionError, ValueError) as e:
            print(f"Skipping file due to error ({e.__class__.__name__}): {full_path}")
    
    if dfs:
        full_df = pl.concat(dfs)
        full_df.write_parquet(output_path)
        print(f"Saved: {output_path}")
        del full_df
        del dfs

# %%
'''Scan Files and Dispatch by Session'''
def scan_and_process_sessions(log_path, pattern):
    session_files = {}
    for dirpath, _, filenames in os.walk(log_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            match = pattern.search(full_path)
            if match:
                session, config, job, layer = match.groups()
                session_files.setdefault(session, []).append((full_path, session, config, job, layer))
    
    for session, files in session_files.items():
        process_and_save_session(session, files)

# %%
'''Main Execution'''
if __name__ == "__main__":
    scan_and_process_sessions(log_path, pattern)
