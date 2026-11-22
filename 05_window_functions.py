from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ------------------------------------------------------------------
# 1. Create the SparkSession (no .master() on EMR, YARN provides it)
# ------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("window_functions")
    .getOrCreate()
)

# ------------------------------------------------------------------
# 2. EXTRACT: read orders from S3 with an explicit schema
# ------------------------------------------------------------------
orders_df = (
    spark.read
    .schema("order_id INT, customer_id STRING, order_date DATE, amount DOUBLE")
    .option("header", True)
    .csv("s3://my-bucket/raw/orders/")
)

# ------------------------------------------------------------------
# 3. DEFINE WINDOW SPECIFICATIONS
#    A window spec is a plain Python object (not a DataFrame, so no _df suffix).
#    partitionBy = the "group" each calculation is restricted to
#    orderBy     = the row order inside each group
# ------------------------------------------------------------------

# 3a. Per customer, oldest order first (used for sequence, lag/lead, running total)
w_by_date = Window.partitionBy("customer_id").orderBy(F.col("order_date").asc())

# 3b. Per customer, biggest amount first (used for ranking)
w_by_amount = Window.partitionBy("customer_id").orderBy(F.col("amount").desc())

# 3c. Per customer, newest order first (used for "latest record per key")
#     order_id is a tie-breaker so the result is deterministic across runs
w_latest = (
    Window.partitionBy("customer_id")
    .orderBy(F.col("order_date").desc(), F.col("order_id").desc())
)

# 3d. Running total frame: from the first row of the group up to the CURRENT row.
#     rowsBetween counts physical rows, so ties on order_date cannot distort it.
w_running = (
    Window.partitionBy("customer_id")
    .orderBy(F.col("order_date").asc())
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

# ------------------------------------------------------------------
# 4. ROW_NUMBER: 1,2,3... per customer, no ties, always unique
# ------------------------------------------------------------------
sequenced_df = orders_df.withColumn("order_seq", F.row_number().over(w_by_date))

# ------------------------------------------------------------------
# 5. RANK vs DENSE_RANK: how ties are handled
#    rank()       -> ties share a number, the next number is SKIPPED (1,2,2,4)
#    dense_rank() -> ties share a number, NO gap                     (1,2,2,3)
# ------------------------------------------------------------------
ranked_df = (
    orders_df
    .withColumn("amount_rank", F.rank().over(w_by_amount))
    .withColumn("amount_dense_rank", F.dense_rank().over(w_by_amount))
)

# ------------------------------------------------------------------
# 6. LAG / LEAD: look at the previous / next row inside the group
#    The first row has no previous row, so lag gives null (same for lead on the last row)
# ------------------------------------------------------------------
lag_lead_df = (
    orders_df
    .withColumn("prev_amount", F.lag("amount", 1).over(w_by_date))   # 1 row back
    .withColumn("next_amount", F.lead("amount", 1).over(w_by_date))  # 1 row ahead
)

# ------------------------------------------------------------------
# 7. RUNNING TOTAL: aggregate function + window frame
# ------------------------------------------------------------------
running_df = orders_df.withColumn("running_total", F.sum("amount").over(w_running))

# ------------------------------------------------------------------
# 8. DEDUPLICATION / LATEST RECORD PER KEY (the most common DE use)
#    Number rows newest-first, keep only row 1, drop the helper column.
# ------------------------------------------------------------------
latest_order_df = (
    orders_df
    .withColumn("rn", F.row_number().over(w_latest))   # 1 = newest order of each customer
    .filter(F.col("rn") == 1)                          # keep only the newest
    .drop("rn")                                        # helper column no longer needed
)

# ------------------------------------------------------------------
# 9. LOAD: write one result (this is the ACTION)
# ------------------------------------------------------------------
(
    latest_order_df
    .write
    .mode("overwrite")
    .parquet("s3://my-bucket/curated/latest_order_per_customer/")
)

spark.stop()
