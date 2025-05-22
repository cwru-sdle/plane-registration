# %%
'''Imports'''
from pypcd import pypcd
import polars as pl
import os
import re
# %%
part_1 = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/lpbf-aconity/registeration/session_2025_03_25_16-54-19.812/config_3_67d0704e91000019001f401c/job_1_67e28ebb8f00003c0203e1e7/sensors/ECAMPCDWriterSink__1/layer/"
part_2 = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/lpbf-aconity/registeration/session_2025_03_25_18-52-34.763/config_3_67d0660b8a00007d0127a247/job_8_67e28ebb8f00003c0203e1e7/sensors/ECAMPCDWriterSink__1/layer/"

def get_pcd_paths(directory):
    return sorted([
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.endswith(".pcd")
    ])

paths = get_pcd_paths(part_1) + get_pcd_paths(part_2)
# %%
'''Load dfs'''
all_dfs = []
for file_path in paths:
    try:
        with open(file_path, "rb") as f:
            layer = int(file_path.split("/")[-1].replace(".pcd","").replace(".",""))
            pc = pypcd.PointCloud.from_fileobj(f)
            df = pl.DataFrame({field: pc.pc_data[field] for field in pc.fields})
            df = df.with_columns(pl.lit(layer).alias("layer"))
            all_dfs.append(df)
    except AssertionError:
        print(f"Skipping invalid or corrupted file: {file_path}")
sizes = [df.estimated_size() for df in all_dfs]

# Sum total memory usage (in bytes)
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