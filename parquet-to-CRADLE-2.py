import pyodbc
import pyarrow.parquet as pq
import os

# Setup: change these as needed
parquet_dir = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_parquet/"
dsn = "CRADLE-3.2-impala"
uid = ""
pwd = ""
table_name = "aconity_pcd"

# Connect via pyodbc directly (no SQLAlchemy)
conn = pyodbc.connect(f"DSN={dsn};UID={uid};PWD={pwd}",autocommit=True)
cursor = conn.cursor()
cursor.fast_executemany = True

# Loop over Parquet files
all_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith(".parquet")]

for file in all_files:
    print(f"Processing: {file}")
    pf = pq.ParquetFile(file)

    for i in range(pf.num_row_groups):
        print(f"  Reading row group {i+1}/{pf.num_row_groups}")
        df = pf.read_row_group(i).to_pandas()
        df = df.astype({
            't': float,
            'x': float,
            'y': float,
            'z': float,
            'sensor0': float,
            'sensor1': float,
            'sensor2': float,
            'sensor3': float,
            'state0': str,
            'state1': str,
            'layer': int,
            'session': str,
            'config': str,
            'job': str,
        })
        # Insert row group into SQL
        if df.empty:
            continue
        cols = ", ".join(df.columns)
        placeholders = ", ".join("?" for _ in df.columns)
        columns = df.columns.tolist()
        column_list = ', '.join(columns)
        placeholders = ', '.join(['?'] * len(columns))  # One ? per column

        insert_sql = f"INSERT INTO aconity_pcd ({column_list}) VALUES ({placeholders})"
        print(f"{i}")
        cursor.executemany(insert_sql, df.itertuples(index=False, name=None))

    conn.execute("COMMIT")