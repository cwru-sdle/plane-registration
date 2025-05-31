import os
import pyarrow.parquet as pq
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