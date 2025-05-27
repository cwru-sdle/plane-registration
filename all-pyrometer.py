# %%
'''Imports'''
from pypcd import pypcd
import polars as pl
import os
import re
import pyarrow.parquet as pq

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
    sensors/ECAMPCDWriterSink__1/(?:([\d]+)/)?                      # optional numeric part_number
    (?:layer/)?(\d{3}\.\d{3})\.pcd                                  # layer
    """,
    re.VERBOSE
)

# %%
'''Process a Batch of Files from One Session'''

def process_and_save_session(session, file_args):
    session_safe = session.replace("/", "_")
    output_path = os.path.join(output_dir, f"{session_safe}.parquet")

    writer = None
    for full_path, _, config, job, subfolder, layer_str in file_args:
        try:
            with open(full_path, "rb") as f:
                layer = int(layer_str.replace(".", ""))
                pc = pypcd.PointCloud.from_fileobj(f)
                df = pl.DataFrame({field: pc.pc_data[field] for field in pc.fields})
                columns = [
                    pl.lit(layer).alias("layer"),
                    pl.lit(session).alias("session"),
                    pl.lit(config).alias("config"),
                    pl.lit(job).alias("job")
                ]
                if subfolder:
                    columns.append(pl.lit(subfolder).cast(pl.Int32).alias("subfolder"))
                df = df.with_columns(columns)

                table = df.to_arrow()

                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
        except (AssertionError, ValueError) as e:
            print(f"Skipping file due to error ({e.__class__.__name__}): {full_path}")

    if writer:
        writer.close()

# %%
'''Scan Files and Dispatch by Session'''
def scan_and_process_sessions(log_path, pattern):
    session_files = {}
    for dirpath, _, filenames in os.walk(log_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            match = pattern.search(full_path)
            if match:
                session, config, job, part_number, layer = match.groups()
                session_files.setdefault(session, []).append(
                    (full_path, session, config, job, part_number, layer)
                )
    for session, files in session_files.items():
        process_and_save_session(session, files)

# %%
'''Main Execution'''
if __name__ == "__main__":
    scan_and_process_sessions(log_path, pattern)
