import os
import pyarrow.parquet as pq
import pandas as pd
from sqlalchemy import create_engine, event
import getpass

import pyodbc
import pandas as pd
import pyarrow.parquet as pq
import os


# Directory containing large Parquet files
parquet_dir = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_parquet/"
all_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith('.parquet')]

# Prompt for credentials
dsn = input("Enter your DSN (e.g., CRADLE-3.2-Impala): ")

# Setup connection (NO SQLAlchemy here)
conn = pyodbc.connect(f"DSN={dsn}")
cursor = conn.cursor()

# Example insert function
def insert_dataframe(df, table_name):
    # Construct and execute insert statements manually
    cols = ",".join(df.columns)
    placeholders = ",".join("?" for _ in df.columns)
    sql = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"
    cursor.fast_executemany = True
    cursor.executemany(sql, df.values.tolist())
    conn.commit()

# Replace with your desired SQL table name
table_name = 'aconity_pcd'

# Whether to replace or append to SQL table
first_write = True

for file in all_files:
    print(f"Processing: {file}")
    pf = pq.ParquetFile(file)
    for i in range(pf.num_row_groups):
        print(f"  Row group {i+1}/{pf.num_row_groups}")
        df = pf.read_row_group(i).to_pandas()
        insert_dataframe(df, table_name)