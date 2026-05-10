import os
import sys
import time
 
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
 
# Configuring
POWER_DATA_PATH   = r"C:\Users\user\Desktop\Energy Consumption\data\household_power_consumption.txt"
WEATHER_DATA_PATH = r"C:\Users\user\Desktop\Energy Consumption\data\open-meteo-48.82N2.29E43m.csv"
 
OUTPUT_BASE        = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output"
OUTPUT_POWER_RAW   = os.path.join(OUTPUT_BASE, "raw",       "power_consumption")
OUTPUT_WEATHER_RAW = os.path.join(OUTPUT_BASE, "raw",       "weather")
OUTPUT_POWER_CLEAN = os.path.join(OUTPUT_BASE, "processed", "power_cleaned")
 
NUMERIC_COLS = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
 
SEP = "=" * 72
 
# Preliminary checks before Spark starts
for fpath in [POWER_DATA_PATH, WEATHER_DATA_PATH]:
    if not os.path.isfile(fpath):
        print(f"[ERROR] Not found: {fpath}")
        sys.exit(1)
    print(f"[FILE CHECK] {os.path.basename(fpath)} ✓")
 
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    print(f"[CHECK] pyarrow {pa.__version__} ✓")
except ImportError:
    print("[ERROR] pyarrow not installed. Run: pip install pyarrow")
    sys.exit(1)
 
for d in [OUTPUT_POWER_RAW, OUTPUT_WEATHER_RAW, OUTPUT_POWER_CLEAN]:
    os.makedirs(d, exist_ok=True)
 
 
# Spark session
def create_spark_session() -> SparkSession:
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
 
    spark = (
        SparkSession.builder
        .appName("M1_02_DataIngestion_Local")
        .master("local[*]")
        .config("spark.driver.memory",          "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress",  "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
 
 
# Schemas 
POWER_SCHEMA = StructType([
    StructField("Date",                  StringType(), True),
    StructField("Time",                  StringType(), True),
    StructField("Global_active_power",   StringType(), True),
    StructField("Global_reactive_power", StringType(), True),
    StructField("Voltage",               StringType(), True),
    StructField("Global_intensity",      StringType(), True),
    StructField("Sub_metering_1",        StringType(), True),
    StructField("Sub_metering_2",        StringType(), True),
    StructField("Sub_metering_3",        StringType(), True),
])
 
 
# Loading data 
def load_power_data(spark):
    print("\n[LOAD] Household power consumption data ...")
    df = (
        spark.read
        .option("header",    "true")
        .option("delimiter", ";")
        .schema(POWER_SCHEMA)
        .csv(POWER_DATA_PATH)
    )
    raw_count = df.count()
    print(f"  ✓ {raw_count:,} rows loaded")
    return df, raw_count
 
 
def load_weather_data(spark):
    import pandas as pd
    print("\n[LOAD] Weather data ...")
    pdf = pd.read_csv(WEATHER_DATA_PATH, skiprows=3)
    pdf.columns = [
        c.split("(")[0].strip().replace(" ", "_").replace("-", "_").lower()
        for c in pdf.columns
    ]
    df = spark.createDataFrame(pdf)
    count = df.count()
    print(f"  ✓ {count:,} rows loaded")
    return df, count
 
 
# Cleaning and transformations
# Cleaning power dataset 
def clean_power_data(df_raw, raw_count):
    print("\n[CLEAN] Power consumption dataset ...")
    df = df_raw
 
    for col in NUMERIC_COLS:
        df = df.withColumn(
        col,
        F.when(
            F.col(col).isNull() |
            (F.trim(F.col(col)) == "") |
            (F.trim(F.col(col)) == "?"),
            None
        ).otherwise(F.col(col).cast(DoubleType()))
    )
    print("Missing values handled and numeric columns casted")
 
    df = df.withColumn(
        "Datetime",
        F.to_timestamp(F.concat_ws(" ", F.col("Date"), F.col("Time")), "d/M/yyyy HH:mm:ss")
    )
    print("datetime column created from Date and Time")
 
    df = df.withColumn(
        "Sub_metering_remainder",
        F.round(
            (F.col("Global_active_power") * 1000.0 / 60.0)
            - F.col("Sub_metering_1") - F.col("Sub_metering_2") - F.col("Sub_metering_3"), 4
        )
    )
    print("Sub_metering_remainder column calculated for unmetered power")
 
    df_clean    = df.filter(F.col("Global_active_power").isNotNull())
    clean_count = df_clean.count()
    dropped     = raw_count - clean_count
    print(f" Dropped {dropped:,} NULL rows ({dropped/raw_count*100:.2f}%)")
 
    df_clean = (
        df_clean
        .withColumn("Year",  F.year("Datetime"))
        .withColumn("Month", F.month("Datetime"))
    )
    print("Year and month partition columns created from Datetime")
 
    df_clean = df_clean.drop("Date", "Time")
    print("Raw Date and Time columns removed")
 
    return df_clean, clean_count
 
 
# Cleaning weather dataset
def clean_weather_data(df_raw):
    print("\n[CLEAN] Weather dataset ...")
    df = df_raw.withColumnRenamed("time", "Date")
    df = df.withColumn("Date",  F.to_date(F.col("Date"), "yyyy-MM-dd"))
    df = df.withColumn("Year",  F.year("Date"))
    df = df.withColumn("Month", F.month("Date"))
    count = df.count()
    print(f"{count:,} rows cleaned")
    return df, count
 
 
# Writting Parquet via pyarrow (bypassing Hadoop NativeIO and file committer)
def write_parquet(df, output_path: str, partition_cols: list, label: str):
    """
    Converts the Spark DataFrame to pandas, then writes Parquet via pyarrow.
    This completely bypasses Hadoop's file committer and NativeIO on Windows.
    The result is valid partitioned Parquet readable by Spark, pandas, or any
    other Parquet-compatible tool.
    """
    print(f"\n[WRITE] {label}")
    print(f"  Path: {output_path}")
    print(f"  Converting to pandas (this may take ~30s for large datasets)...")
 
    pdf = df.toPandas()
    print(f"  Rows collected: {len(pdf):,}")
 
    # Clear existing output so overwrite is clean
    import shutil
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
 
    # Count written files for confirmation
    written = [
        f for _, _, files in os.walk(output_path)
        for f in files if f.endswith(".parquet")
    ]
    print(f"Written — {len(written)} parquet file(s)")
 
 
# Validation 
def validate_output(output_path: str, expected_count: int, label: str):
    """Read-back validation using pyarrow — no Spark needed."""
    print(f"\n[VALIDATE] {label}")
 
    if not os.path.isdir(output_path):
        print(f"FAIL — directory not found: {output_path}")
        return False
 
    dataset = pq.read_table(output_path)
    actual  = len(dataset)
    status  = "PASS" if actual == expected_count else "FAIL"
    print(f"  Expected: {expected_count:,} | Read-back: {actual:,}  {status}")
 
    if actual != expected_count:
        return False
 
    # Show schema and sample
    print(f"  Schema: {dataset.schema}")
    print(f"  Sample (3 rows):\n{dataset.slice(0, 3).to_pandas().to_string()}")
    return True
 
 
# Main execution
def main():
    print(SEP)
    print("  M1 Data Ingestion — LOCAL EXECUTION")
    print("  Write engine: pyarrow (no Hadoop native IO)")
    print(SEP)
 
    t_start = time.time()
    spark   = create_spark_session()
    print(f"\n[SPARK] Session started — master: local[*]")
 
    # Load
    df_power_raw,   raw_power_count   = load_power_data(spark)
    df_weather_raw, raw_weather_count = load_weather_data(spark)
 
    # Clean
    df_power_clean,  clean_count   = clean_power_data(df_power_raw,  raw_power_count)
    df_weather_clean, weather_count = clean_weather_data(df_weather_raw)
 
    # Write — pyarrow handles all three datasets
    write_parquet(df_power_raw,   OUTPUT_POWER_RAW,   [],                "raw power")
    write_parquet(df_weather_raw, OUTPUT_WEATHER_RAW, [],                "raw weather")
    write_parquet(df_power_clean, OUTPUT_POWER_CLEAN, ["Year", "Month"], "cleaned power")
 
    spark.stop()
 
    # Validate — pyarrow read-back, no Spark required
    print(f"\n{SEP}\n  VALIDATION\n{SEP}")
    ok1 = validate_output(OUTPUT_POWER_CLEAN, clean_count,       "cleaned power")
    ok2 = validate_output(OUTPUT_WEATHER_RAW, raw_weather_count, "raw weather")
 
    if not (ok1 and ok2):
        print("\n[ERROR] Validation failed.")
        sys.exit(1)
 
    elapsed = time.time() - t_start
    print(f"\n{SEP}")
    print(f"  MILESTONE 1 INGESTION COMPLETE — {elapsed:.1f}s")
    print(f"  Output folder: {OUTPUT_BASE}")
    print(f"  Next step    : python scripts/milestone1/upload_minio.py")
    print(SEP)
 
 
if __name__ == "__main__":
    main()