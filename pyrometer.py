# %%
'''Imports'''
from pypcd import pypcd
import polars as pl
import os
import re
from multiprocessing import Pool, cpu_count
# %%
'''Get Paths and Load DataFrames'''
log_path = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_log/"

pattern = re.compile(
    r"""
    session_(\d{4}_\d{2}_\d{2}_\d{2}-\d{2}-\d{2}\.\d+)/     # session
    config_(\d+_[a-f0-9]+)/                                 # config
    job_(\d+_[a-f0-9]+)/                                    # job
    sensors/ECAMPCDWriterSink__1/layer/(\d{3}\.\d{3})\.pcd  # layer
    """,
    re.VERBOSE
)

# Define this outside so it can be pickled
def process_file(args):
    full_path, session, config, job, layer_str = args
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
            return df
    except (AssertionError, ValueError) as e:
        print(f"Skipping file due to error ({e.__class__.__name__}): {full_path}")
        return None

# Main
def parallel_process(log_path, pattern):
    file_args = []
    for dirpath, dirnames, filenames in os.walk(log_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            match = pattern.search(full_path)
            if match:
                file_args.append((full_path, *match.groups()))

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, file_args)

    # Filter out failed results (None)
    all_dfs = [df for df in results if df is not None]
    return all_dfs
all_dfs = parallel_process(log_path, pattern)

# %%
'''Compute Memory Size'''
sizes = [df.estimated_size() for df in all_dfs]
total_bytes = sum(sizes)
print(f"Memory size of all dataframes combined: {total_bytes / (1024**2):.2f} MB")
# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D plotting

# Example: plotting from one DataFrame
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#filtered_df = df.filter(pl.col("t") % 10 == 0)
filtered_df = df.group_by("layer").map_groups(lambda group: group.sample(fraction=0.3, with_replacement=False))
ax.scatter(filtered_df['x'], filtered_df['y'], filtered_df['z'], c=filtered_df['sensor0'], s=10)  # s = marker size

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Point Cloud")

ax.view_init(elev=0, azim=-90)
plt.show()

if "__name__" == __"__main__:
    pass