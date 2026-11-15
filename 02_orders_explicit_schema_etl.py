from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, StringType, DoubleType, TimestampType
)

# ------------------------------------------------------------------
# 1. Create the SparkSession (no .master() on EMR, YARN provides it)
# ------------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("orders_explicit_schema_etl")
    .getOrCreate()
)

# ------------------------------------------------------------------
# 2. Define the schema explicitly (this is a plain Python object,
#    not a DataFrame, so no _df suffix)
#    - The last field, _corrupt_record, is a special column where Spark
#      stores the raw line of any row it fails to parse.
#    - It MUST be StringType and must be part of the schema.
# ------------------------------------------------------------------
orders_schema = StructType([
    StructField("order_id",        IntegerType(),   True),   # order identifier
    StructField("customer_id",     StringType(),    True),   # customer identifier
    StructField("status",          StringType(),    True),   # COMPLETED / CANCELLED / PENDING
    StructField("order_ts",        TimestampType(), True),   # when the order was placed
    StructField("amount",          DoubleType(),    True),   # order value
    StructField("_corrupt_record", StringType(),    True),   # raw text of bad rows
])

# ------------------------------------------------------------------
# 3. EXTRACT: read CSV with the explicit schema
#    mode options:
#      PERMISSIVE   (default) keep bad rows, put raw line in _corrupt_record
#      DROPMALFORMED silently drop bad rows (data loss, avoid in prod)
#      FAILFAST     fail the whole job on the first bad row
#    We use PERMISSIVE so we can route bad rows to a quarantine location.
# ------------------------------------------------------------------
raw_orders_df = (
    spark.read
    .schema(orders_schema)                                # no inference pass
    .option("header", True)                               # first line is the header
    .option("mode", "PERMISSIVE")                         # keep malformed rows
    .option("columnNameOfCorruptRecord", "_corrupt_record")
    .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")     # expected timestamp layout
    .csv("s3://my-bucket/raw/orders/")
)

# ------------------------------------------------------------------
# 4. Cache the DataFrame. Spark restricts queries that touch only the
#    _corrupt_record column on raw files, and we will use this DF twice
#    (good rows and bad rows), so caching also avoids re-reading S3.
# ------------------------------------------------------------------
raw_orders_df = raw_orders_df.cache()

# ------------------------------------------------------------------
# 5. SPLIT into good rows and bad rows
# ------------------------------------------------------------------

# 5a. Good rows: _corrupt_record is null, then drop the helper column
valid_orders_df = (
    raw_orders_df
    .filter(F.col("_corrupt_record").isNull())    # parsed successfully
    .drop("_corrupt_record")                      # no longer needed
)

# 5b. Bad rows: keep only the raw text plus when we rejected it
rejected_orders_df = (
    raw_orders_df
    .filter(F.col("_corrupt_record").isNotNull())            # failed to parse
    .select("_corrupt_record")                               # raw original line
    .withColumn("rejected_at", F.current_timestamp())        # audit timestamp
)

# ------------------------------------------------------------------
# 6. LOAD: write both outputs (each write is an ACTION)
# ------------------------------------------------------------------

# 6a. Clean data to the curated zone
(
    valid_orders_df
    .write
    .mode("overwrite")
    .parquet("s3://my-bucket/curated/orders/")
)

# 6b. Bad data to a quarantine zone for investigation / reprocessing
(
    rejected_orders_df
    .write
    .mode("append")                                 # keep history of rejects
    .parquet("s3://my-bucket/quarantine/orders/")
)

# ------------------------------------------------------------------
# 7. Release cached data and cluster resources
# ------------------------------------------------------------------
raw_orders_df.unpersist()
spark.stop()
