from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ------------------------------------------------------------------
# 1. Create the SparkSession (no .master() on EMR, YARN provides it)
# ------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("join_basics")
    .getOrCreate()
)

# ------------------------------------------------------------------
# 2. EXTRACT: read two datasets from S3
#    .schema("...") accepts a DDL string, a shorter alternative to
#    building a StructType by hand. Same effect: no inference pass.
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
# 3. INNER JOIN: only rows whose customer_id exists in BOTH sides
#    Passing the column NAME as a string ("customer_id") works when both
#    DataFrames use the same column name. Spark keeps ONE customer_id.
# ------------------------------------------------------------------
inner_df = orders_df.join(customers_df, on="customer_id", how="inner")

# ------------------------------------------------------------------
# 4. LEFT JOIN: keep ALL orders, attach customer info where it exists
#    (unmatched orders get nulls in the customer columns)
# ------------------------------------------------------------------
left_df = orders_df.join(customers_df, on="customer_id", how="left")

# ------------------------------------------------------------------
# 5. RIGHT JOIN: keep ALL customers, attach orders where they exist
# ------------------------------------------------------------------
right_df = orders_df.join(customers_df, on="customer_id", how="right")

# ------------------------------------------------------------------
# 6. FULL OUTER JOIN: keep everything from both sides
# ------------------------------------------------------------------
full_df = orders_df.join(customers_df, on="customer_id", how="full")

# ------------------------------------------------------------------
# 7. LEFT SEMI JOIN: a FILTER, not a merge.
#    Keeps orders that HAVE a matching customer, returns ONLY orders columns.
# ------------------------------------------------------------------
semi_df = orders_df.join(customers_df, on="customer_id", how="left_semi")

# ------------------------------------------------------------------
# 8. LEFT ANTI JOIN: the opposite filter.
#    Keeps orders that have NO matching customer (great for data quality
#    checks such as finding orphan records).
# ------------------------------------------------------------------
anti_df = orders_df.join(customers_df, on="customer_id", how="left_anti")

# ------------------------------------------------------------------
# 9. JOIN WITH AN EXPRESSION: the duplicate column problem
#    When the condition is an expression, BOTH customer_id columns survive.
#    We reference each by its own DataFrame (df["col"]), because
#    F.col("customer_id") would be ambiguous here.
# ------------------------------------------------------------------
dup_df = orders_df.join(
    customers_df,
    on=orders_df["customer_id"] == customers_df["customer_id"],
    how="inner"
)

# Selecting "customer_id" on dup_df would raise an AnalysisException
# (ambiguous reference). Fix: drop the right-side copy explicitly.
fixed_df = dup_df.drop(customers_df["customer_id"])

# ------------------------------------------------------------------
# 10. LOAD: write one result (the write is the ACTION)
# ------------------------------------------------------------------
(
    inner_df
    .write
    .mode("overwrite")
    .parquet("s3://my-bucket/curated/orders_enriched/")
)

spark.stop()
