# %%
import polars as pl
import numpy as np
import torch
import torch_geometric
import sklearn
import pyarrow.parquet as pq

path = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_parquet/2025_03_25_18-52-34.763.parquet"
num_rows_to_load = 1000
knn_k = 5
stream_file = pq.ParquetFile(path)
# %%
'''Creates a Graph For Each Row'''

# For loop loads throguh each row group, which correspon to each layer, and loads them as a seperate graph
for group_index in range(stream_file.num_row_groups): 
    df = pl.from_arrow(stream_file.read_row_group(group_index)) # Loads the row groups (layer) as an eager polars dataframe
    print(df.select(pl.all().n_unique()))
    x_polars = torch.tensor(df_polars.select(["sensor1", "sensor2"]).to_numpy(), dtype=torch.float)
    pos_polars = df_polars.select(["x", "y", "z", "layer"]).to_numpy()

# KNN graph with Polars data
nbrs_polars = sklearn.neighbors.NearestNeighbors(n_neighbors=knn_k + 1).fit(pos_polars)
distances_p, indices_p = nbrs_polars.kneighbors(pos_polars)

edge_index_p = []
for i in range(pos_polars.shape[0]):
    for j in indices_p[i, 1:]:
        edge_index_p.append([i, j])
        edge_index_p.append([j, i])

edge_index_p = torch.tensor(edge_index_p, dtype=torch.long).t().contiguous()

data_polars = torch_geometric.data.Data(x=x_polars, edge_index=edge_index_p)

print(f"Graph from Polars: {data_polars}")
print(f"Number of nodes: {data_polars.num_nodes}, Number of edges: {data_polars.num_edges}")

# %%
