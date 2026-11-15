from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ------------------------------------------------------------------
# 1. Create the SparkSession (entry point to Spark)
#    On EMR we do NOT call .master(...): YARN already provides the
#    cluster manager, and spark-submit passes in the resources.
# ------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("basic_etl")      # name shown in the YARN / Spark UI
    .getOrCreate()             # reuse an existing session if one exists
)

# ------------------------------------------------------------------
# 2. EXTRACT: read raw CSV from S3 into a DataFrame
#    On EMR, data lives in S3 (s3://), not on local disk.
# ------------------------------------------------------------------
orders_df = (
    spark.read
    .option("header", True)        # first line contains column names
    .option("inferSchema", True)   # let Spark guess data types (demo only)
    .csv("s3://my-bucket/raw/orders/")   # reads every CSV file in this folder
)

# ------------------------------------------------------------------
# 3. TRANSFORM
# ------------------------------------------------------------------

# 3a. Keep only completed orders (cancelled/pending are not revenue)
completed_orders_df = orders_df.filter(F.col("status") == "COMPLETED")

# 3b. Derive a date column from the timestamp (drops the time part)
orders_with_date_df = completed_orders_df.withColumn(
    "order_date", F.to_date("order_ts")
)

# 3c. Aggregate: one row per day with order count and total revenue
daily_revenue_df = (
    orders_with_date_df
    .groupBy("order_date")                              # group rows by day
    .agg(
        F.count("*").alias("order_count"),              # number of orders
        F.sum("amount").alias("total_revenue")          # sum of order amounts
    )
)

# ------------------------------------------------------------------
# 4. LOAD: write result to S3 as Parquet (this is the ACTION that
#    triggers the whole plan to execute across the cluster)
# ------------------------------------------------------------------
(
    daily_revenue_df
    .write
    .mode("overwrite")                                  # replace existing output
    .parquet("s3://my-bucket/curated/daily_revenue/")
)

# ------------------------------------------------------------------
# 5. Release cluster resources
# ------------------------------------------------------------------
spark.stop()
