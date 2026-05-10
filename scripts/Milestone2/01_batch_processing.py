import os
import sys
import time
 
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    TimestampType, IntegerType, DateType
)

# Configuring
INPUT_CLEAN   = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\processed\power_cleaned"
INPUT_WEATHER = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\raw\weather"
OUTPUT_BATCH  = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\batch"
 
WEATHER_DATA_PATH = r"C:\Users\user\Desktop\Energy Consumption\data\open-meteo-48.82N2.29E43m.csv"
 
SEP = "=" * 72

# Spark session
def create_spark_session() -> SparkSession:
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
 
    spark = (
        SparkSession.builder
        .appName("M2_01_BatchProcessing")
        .master("local[*]")
        .config("spark.driver.memory",          "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.task.maxFailures",        "4")
        .config("spark.ui.showConsoleProgress",  "false")
        # Fault tolerance: speculative execution re-launches slow tasks
        .config("spark.speculation",             "false")  # off for local mode
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

# Loading datasets 
# Loading cleaned power data 
def load_cleaned_power(spark: SparkSession):
    print("\n[LOAD] Cleaned power data from Parquet ...")
    import pyarrow.parquet as pq
    dataset = pq.read_table(INPUT_CLEAN)
    pdf     = dataset.to_pandas()
    df      = spark.createDataFrame(pdf)

    # Ensure correct types after pandas round-trip
    df = df.withColumn("Datetime", F.col("Datetime").cast("timestamp"))
    df = df.withColumn("Year",     F.col("Year").cast("int"))
    df = df.withColumn("Month",    F.col("Month").cast("int"))

    total = df.count()
    print(f"  ✓ Total rows: {total:,}")

    # Partition pruning demo — filter after loading
    print("\n  [PARTITION PRUNING DEMO] Filtering Year=2007 ...")
    t0 = time.time()
    pruned_count = df.filter(F.col("Year") == 2007).count()
    t1 = time.time()
    print(f"  Year=2007 rows : {pruned_count:,}  (elapsed: {t1-t0:.2f}s)")
    print(f"  Rows skipped   : {total - pruned_count:,}")

    return df, total 

# Loading weather data
def load_weather(spark: SparkSession):
    print("\n[LOAD] Weather data ...")
    import pandas as pd

    # Always read from raw CSV — most reliable on Windows
    pdf = pd.read_csv(WEATHER_DATA_PATH, skiprows=3)
    pdf.columns = [
        c.split("(")[0].strip().replace(" ", "_").replace("-", "_").lower()
        for c in pdf.columns
    ]
    pdf = pdf.rename(columns={"time": "Date"})
    pdf["Date"] = pd.to_datetime(pdf["Date"]).dt.date

    df    = spark.createDataFrame(pdf)
    df    = df.withColumn("Date", F.to_date(F.col("Date").cast("string"), "yyyy-MM-dd"))
    count = df.count()
    print(f"{count:,} daily weather records loaded")
    return df

# Aggregation layer A: Hourly patterns
def compute_hourly_patterns(df):
    print("\n[AGGREGATE] A — Hourly consumption patterns ...")
    df_hourly = (
        df
        .withColumn("Hour", F.hour("Datetime"))
        .groupBy("Hour")
        .agg(
            F.round(F.avg("Global_active_power"), 4).alias("avg_kw"),
            F.round(F.max("Global_active_power"), 4).alias("peak_kw"),
            F.round(F.min("Global_active_power"), 4).alias("min_kw"),
            F.round(F.stddev("Global_active_power"), 4).alias("stddev_kw"),
            F.count("*").alias("record_count"),
        )
        .orderBy("Hour")
    )
 
    print("  Hourly averages (kW):")
    df_hourly.show(24, truncate=False)
    return df_hourly

# Aggregation layer B: Daily totals
def compute_daily_totals(df):
    print("\n[AGGREGATE] B — Daily energy totals ...")
 
    df_daily = (
        df
        .withColumn("Date", F.to_date("Datetime"))
        .groupBy("Date", "Year", "Month")
        .agg(
            # kWh = sum of per-minute averages × (1/60)
            F.round(F.sum("Global_active_power") / 60.0, 4).alias("total_kwh"),
            F.round(F.avg("Global_active_power"), 4).alias("avg_kw"),
            F.round(F.max("Global_active_power"), 4).alias("peak_kw"),
            F.count("*").alias("minutes_recorded"),
        )
        .orderBy("Date")
    )
 
    total_days = df_daily.count()
    print(f"{total_days:,} daily records computed")
    df_daily.show(5, truncate=False)
    return df_daily

# Aggregation layer C: Monthly totals
def compute_monthly_summary(df):
    print("\n[AGGREGATE] C — Monthly consumption summary ...")
 
    df_monthly = (
        df
        .groupBy("Year", "Month")
        .agg(
            F.round(F.sum("Global_active_power") / 60.0,  4).alias("total_kwh"),
            F.round(F.avg("Global_active_power"),          4).alias("avg_kw"),
            F.round(F.max("Global_active_power"),          4).alias("peak_kw"),
            F.round(F.stddev("Global_active_power"),       4).alias("stddev_kw"),
            F.round(F.avg("Sub_metering_1"),               4).alias("avg_sm1_kitchen_wh"),
            F.round(F.avg("Sub_metering_2"),               4).alias("avg_sm2_laundry_wh"),
            F.round(F.avg("Sub_metering_3"),               4).alias("avg_sm3_hvac_wh"),
            F.round(F.avg("Sub_metering_remainder"),       4).alias("avg_sm_other_wh"),
            F.count("*").alias("minutes_recorded"),
        )
        .orderBy("Year", "Month")
    )
 
    total_months = df_monthly.count()
    print(f"{total_months:,} monthly records computed")
    df_monthly.show(10, truncate=False)
    return df_monthly

# Aggregation layer D: Sub metering breakdown
def compute_submetering_breakdown(df):
    print("\n[AGGREGATE] D — Sub-metering appliance breakdown ...")
 
    df_sub = (
        df
        .withColumn("Date", F.to_date("Datetime"))
        .groupBy("Date", "Year", "Month")
        .agg(
            # Convert Wh/min totals to kWh: sum(Wh) / 1000
            F.round(F.sum("Sub_metering_1")         / 1000.0, 4).alias("kitchen_kwh"),
            F.round(F.sum("Sub_metering_2")         / 1000.0, 4).alias("laundry_kwh"),
            F.round(F.sum("Sub_metering_3")         / 1000.0, 4).alias("hvac_kwh"),
            F.round(F.sum("Sub_metering_remainder") / 1000.0, 4).alias("other_kwh"),
            F.round(F.sum("Global_active_power")    / 60.0,   4).alias("total_kwh"),
        )
        # Percentage share of each sub-meter in total daily consumption
        .withColumn("kitchen_pct",
            F.round(F.col("kitchen_kwh") / F.col("total_kwh") * 100, 2))
        .withColumn("laundry_pct",
            F.round(F.col("laundry_kwh") / F.col("total_kwh") * 100, 2))
        .withColumn("hvac_pct",
            F.round(F.col("hvac_kwh")    / F.col("total_kwh") * 100, 2))
        .withColumn("other_pct",
            F.round(F.col("other_kwh")   / F.col("total_kwh") * 100, 2))
        .orderBy("Date")
    )
 
    total = df_sub.count()
    print(f"{total:,} daily sub-metering records computed")
    df_sub.show(5, truncate=False)
 
    # Overall averages across full dataset
    print("\n  Overall appliance share (average across all days):")
    df_sub.agg(
        F.round(F.avg("kitchen_pct"), 2).alias("kitchen_%"),
        F.round(F.avg("laundry_pct"), 2).alias("laundry_%"),
        F.round(F.avg("hvac_pct"),    2).alias("hvac_%"),
        F.round(F.avg("other_pct"),   2).alias("other_%"),
    ).show(truncate=False)
 
    return df_sub

# Aggregation layer E: Weather Joins
def compute_weather_join(df_daily, df_weather):
    print("\n[AGGREGATE] E — Weather join ...")
 
    # Ensuring matching Date column types
    df_weather_dated = df_weather.withColumn(
        "Date", F.to_date(F.col("Date").cast("string"), "yyyy-MM-dd")
    )
 
    df_joined = (
        df_daily
        .join(df_weather_dated.select(
            "Date",
            F.col("temperature_2m_mean").alias("temp_mean_c"),
            F.col("temperature_2m_max").alias("temp_max_c"),
            F.col("temperature_2m_min").alias("temp_min_c"),
            F.col("precipitation_sum").alias("precip_mm"),
        ), on="Date", how="left")
        .orderBy("Date")
    )
 
    total    = df_joined.count()
    matched  = df_joined.filter(F.col("temp_mean_c").isNotNull()).count()
    print(f"{total:,} daily rows | {matched:,} matched with weather data")
    df_joined.show(5, truncate=False)
 
    # Quick correlation: temperature vs consumption
    corr = df_joined.stat.corr("total_kwh", "temp_mean_c")
    print(f"\n  Pearson correlation (daily kWh vs mean temperature): {corr:.4f}")
    print(f"  Interpretation: {'negative' if corr < 0 else 'positive'} correlation — "
          f"{'colder days → higher consumption (heating effect)' if corr < 0 else 'warmer days → higher consumption (cooling effect)'}")
 
    return df_joined

# Writing output
def write_parquet(df, output_path: str, partition_cols: list, label: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shutil
 
    print(f"\n[WRITE] {label}  →  {output_path}")
 
    pdf = df.toPandas()
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
 
    table = pa.Table.from_pandas(pdf, preserve_index=False)
 
    if partition_cols:
        pq.write_to_dataset(
            table,
            root_path=output_path,
            partition_cols=partition_cols,
            existing_data_behavior="overwrite_or_ignore",
        )
    else:
        pq.write_table(table, os.path.join(output_path, "data.parquet"))
 
    files = [f for _, _, fs in os.walk(output_path) for f in fs if f.endswith(".parquet")]
    print(f" {len(files)} parquet file(s) written")

# Scalability and Partitioning analysis
def analyse_partitioning(df, spark):
    default_parts = df.rdd.getNumPartitions()
    print(f"\n[ANALYSE] Partitioning impact on performance ...")
    # Show partition size distribution
    print("  Partition size distribution (repartition by Year):")
    df_year = df.repartition("Year")
    df_year.groupBy("Year").count().orderBy("Year").show()
 
# Main
def main():
    t_start = time.time()
 
    spark = create_spark_session()
    print(f"\n[SPARK] Session started — master: local[*]")

    # Load datasets
    df_power, total_rows = load_cleaned_power(spark)
    df_weather           = load_weather(spark)

    # Compute aggregations
    df_hourly  = compute_hourly_patterns(df_power)
    df_daily   = compute_daily_totals(df_power)
    df_monthly = compute_monthly_summary(df_power)
    df_sub     = compute_submetering_breakdown(df_power)
    df_weather_joined = compute_weather_join(df_daily, df_weather)

    # Partitioning analysis
    analyse_partitioning(df_power, spark)

    # Write outputs
    os.makedirs(OUTPUT_BATCH, exist_ok=True)
 
    write_parquet(df_hourly,         os.path.join(OUTPUT_BATCH, "hourly_patterns"),    [],              "hourly patterns")
    write_parquet(df_daily,          os.path.join(OUTPUT_BATCH, "daily_totals"),       ["Year","Month"],"daily totals")
    write_parquet(df_monthly,        os.path.join(OUTPUT_BATCH, "monthly_summary"),    ["Year"],        "monthly summary")
    write_parquet(df_sub,            os.path.join(OUTPUT_BATCH, "submetering_daily"),  ["Year","Month"],"sub-metering breakdown")
    write_parquet(df_weather_joined, os.path.join(OUTPUT_BATCH, "daily_with_weather"), ["Year","Month"],"weather-joined daily")
 
    elapsed = time.time() - t_start
    spark.stop()
    print(f"\n[OVERALL] Total execution time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()


 
 

 