from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ------------------------------------------------------------------
# 1. Create the SparkSession
#    The AQE (Adaptive Query Execution) settings below let Spark
#    re-optimize joins at runtime. They are on by default in recent
#    EMR releases (Spark 3.2+), but we set them explicitly so the
#    behavior is visible and does not depend on the EMR version.
# ------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("broadcast_and_skew")
    .config("spark.sql.adaptive.enabled", "true")                       # turn AQE on
    .config("spark.sql.adaptive.skewJoin.enabled", "true")              # auto-split skewed partitions
    .config("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024)   # auto-broadcast tables under 10 MB
    .getOrCreate()
)

# ------------------------------------------------------------------
# 2. EXTRACT: big table (orders) and small table (customers)
# ------------------------------------------------------------------
orders_df = (
    spark.read
    .schema("order_id INT, customer_id STRING, amount DOUBLE")
    .option("header", True)
    .csv("s3://my-bucket/raw/orders/")
)

customers_df = (
    spark.read
    .schema("customer_id STRING, customer_name STRING, city STRING")
    .option("header", True)
    .csv("s3://my-bucket/raw/customers/")
)

# ------------------------------------------------------------------
# 3. BROADCAST JOIN: copy the small table to every executor, so the
#    big table never has to move across the network (no shuffle of orders).
#    F.broadcast() is a hint that forces this strategy.
# ------------------------------------------------------------------
broadcast_df = orders_df.join(
    F.broadcast(customers_df),          # small side gets broadcast
    on="customer_id",
    how="left"                          # for a LEFT join, only the RIGHT side can be broadcast
)

# ------------------------------------------------------------------
# 4. VERIFY the strategy: explain() prints the physical plan.
#    Look for BroadcastHashJoin (good) vs SortMergeJoin (shuffle on both sides).
# ------------------------------------------------------------------
broadcast_df.explain()

# ------------------------------------------------------------------
# 5. DETECT SKEW: count rows per join key and look at the top keys.
#    One key with far more rows than the rest means one task does most of the work.
# ------------------------------------------------------------------
key_counts_df = (
    orders_df
    .groupBy("customer_id")                     # one row per key
    .agg(F.count("*").alias("row_count"))       # rows per key
    .orderBy(F.col("row_count").desc())         # hottest keys first
)
key_counts_df.show(5)

# ------------------------------------------------------------------
# 6. MANUAL SALTING: for when the "small" table is too big to broadcast
#    AND one key is hot. Idea: split the hot key into N sub-keys.
# ------------------------------------------------------------------
num_salts = 3     # plain Python variable, not a DataFrame (no _df suffix)

# 6a. Big side: give every order a RANDOM salt from 0 to num_salts-1
orders_salted_df = orders_df.withColumn(
    "salt", F.floor(F.rand() * num_salts).cast("int")   # rand() is 0..1, so floor(rand*3) is 0, 1 or 2
)

# 6b. Small side: REPLICATE every customer once per salt value.
#     array(0,1,2) + explode() turns 1 row into 3 rows, one per salt.
customers_salted_df = customers_df.withColumn(
    "salt",
    F.explode(F.array(*[F.lit(i) for i in range(num_salts)]))
)

# 6c. Join on BOTH the real key and the salt, then drop the helper column
salted_join_df = (
    orders_salted_df
    .join(customers_salted_df, on=["customer_id", "salt"], how="inner")
    .drop("salt")
)

# ------------------------------------------------------------------
# 7. LOAD: write the result (this is the ACTION)
# ------------------------------------------------------------------
(
    salted_join_df
    .write
    .mode("overwrite")
    .parquet("s3://my-bucket/curated/orders_enriched/")
)

spark.stop()
