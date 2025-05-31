# %%
from pyspark.sql import SparkSession
import os
parquet_directory = "/mnt/vstor/CSE_MSE_RXF131/lab-staging/mds3/AdvManu/aconity_parquet/"
files = ["file://"+parquet_directory+file for file in os.listdir(parquet_directory)]
spark = SparkSession.builder \
    .config("spark.pyspark.python", "/python") \
    .getOrCreate()

# %%
df = spark.read.parquet(files[10])
unique_layers = df.select("layer").distinct().rdd.flatMap(lambda x: x).collect()
print(unique_layers)