import os
import sys
import time
 
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Configuring
INPUT_CLEAN  = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\processed\power_cleaned"
OUTPUT_BATCH = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\batch"
 
SEP = "=" * 72

# Helper function to create Spark session with configurable shuffle partitions
def read_parquet_via_pyarrow(spark: SparkSession, path: str):
    """Reads a local Parquet dataset via pyarrow then convert to Spark DataFrame."""
    import pyarrow.parquet as pq
    
    table = pq.read_table(path)
    pdf   = table.to_pandas()
    df    = spark.createDataFrame(pdf)
    
    if "Datetime" in df.columns:
        df = df.withColumn("Datetime", F.col("Datetime").cast("timestamp"))
    if "Year"  in df.columns:
        df = df.withColumn("Year",  F.col("Year").cast("int"))
    if "Month" in df.columns:
        df = df.withColumn("Month", F.col("Month").cast("int"))
    if "Date"  in df.columns:
        df = df.withColumn("Date",  F.col("Date").cast("date"))
    
    return df   
# Spark session
def create_spark_session(shuffle_partitions: int = 8) -> SparkSession:
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
 
    existing = SparkSession.getActiveSession()
    if existing:
        existing.stop()
 
    spark = (
        SparkSession.builder
        .appName("M2_02_ScalabilityAnalysis")
        .master("local[*]")
        .config("spark.driver.memory",          "4g")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.task.maxFailures",        "4")
        .config("spark.ui.showConsoleProgress",  "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
 

# Benchmark 1: Shuffle partition counts
def benchmark_partition_counts(spark):
    df = read_parquet_via_pyarrow(spark, INPUT_CLEAN)
    results = []
 
    partition_counts = [2, 4, 8, 16]
 
    for p in partition_counts:
        spark.conf.set("spark.sql.shuffle.partitions", str(p))
 
        t0 = time.time()
        count = (
            df
            .withColumn("Date", F.to_date("Datetime"))
            .groupBy("Date")
            .agg(F.sum("Global_active_power").alias("total_power"))
            .count()  # force full evaluation
        )
        elapsed = time.time() - t0
        results.append((p, elapsed, count))
        print(f"  partitions={p:>3}  |  {count:,} daily groups  |  {elapsed:.2f}s")
 
    # Reset to default
    spark.conf.set("spark.sql.shuffle.partitions", "8")

    best = min(results, key=lambda x: x[1])

    print("\n[RESULTS] Partition Count Benchmark:")
    return results

# Benchmark 2: Partition pruning 
def benchmark_partition_pruning(spark):
    df    = read_parquet_via_pyarrow(spark, INPUT_CLEAN)
    total = df.count()
 
    # Full scan — reads all partitions
    t0 = time.time()
    full_count = df.filter(F.col("Month") == 1).count()
    full_elapsed = time.time() - t0
 
    # Pruned scan — reads only Year=*/Month=1 directories
    t0 = time.time()
    pruned_count = df.filter(
        (F.col("Year") == 2008) & (F.col("Month") == 1)
    ).count()
    pruned_elapsed = time.time() - t0
    pct_skipped = ((total - pruned_count) / total) * 100
 
    speedup = full_elapsed / max(pruned_elapsed, 0.001)
    print(f"\n[RESULTS] Partition Pruning Benchmark:")

    return full_elapsed, pruned_elapsed

# Data skew analysis
def analyse_data_skew(spark: SparkSession):
    df = read_parquet_via_pyarrow(spark, INPUT_CLEAN)
 
    skew_df = (
        df.groupBy("Year", "Month")
          .count()
          .withColumnRenamed("count", "rows")
          .orderBy("Year", "Month")
    )
 
    print("  Rows per Year/Month partition:")
    skew_df.show(60, truncate=False)
 
    stats = skew_df.agg(
        F.min("rows").alias("min_rows"),
        F.max("rows").alias("max_rows"),
        F.round(F.avg("rows"), 0).alias("avg_rows"),
        F.round(F.stddev("rows"), 0).alias("stddev_rows"),
        F.count("*").alias("num_partitions"),
    ).collect()[0]
 
    skew_ratio = stats["max_rows"] / max(stats["min_rows"], 1)

    assessment = (
        "LOW SKEW — partitions are well balanced ✓" if skew_ratio < 3
        else "MODERATE SKEW — acceptable for time-series data"
        if skew_ratio < 10
        else "HIGH SKEW — consider salting or finer partitioning"
    )


    print(f"""
    Skew Analysis:
        Number of partitions : {stats['num_partitions']}
        Min partition size   : {stats['min_rows']:,} rows
        Max partition size   : {stats['max_rows']:,} rows
        Avg partition size   : {stats['avg_rows']:,.0f} rows
        Std deviation        : {stats['stddev_rows']:,.0f} rows
        Skew ratio (max/min) : {skew_ratio:.2f}x
 
    Assessment: {assessment}
    """)

# Scalability report
def print_scalability_report(n: int, benchmark_results):
    best = min(benchmark_results, key=lambda x: x[1])
    print(f"""[SCALABILITY REPORT]
  Total rows: {n:,} | Best partition count: {best[0]}  |  Time: {best[1]:.2f}s
  Partition count benchmarks:""")
    for p, elapsed, count in benchmark_results:
        print(f"  - {p:>3} partitions: {elapsed:.2f}s  |  {count:,} daily groups")  


# Main execution
def main():
    t_start = time.time()
    spark   = create_spark_session()
 
    print("\n[LOAD] Reading cleaned power data ...")
    n = read_parquet_via_pyarrow(spark, INPUT_CLEAN).count()
    print(f"  ✓ {n:,} rows")
 
    bench_results = benchmark_partition_counts(spark)
    benchmark_partition_pruning(spark)
    analyse_data_skew(spark)
    print_scalability_report(n, bench_results)
 
    elapsed = time.time() - t_start
    spark.stop()
    print(f"\n[OVERALL] Total execution time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()