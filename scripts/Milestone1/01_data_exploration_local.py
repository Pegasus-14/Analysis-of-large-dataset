import sys
import time
import os
 
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

# Configuring
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
POWER_DATA_PATH   = r"C:\\Users\\user\\Desktop\\Energy Consumption\\data\\household_power_consumption.txt"
WEATHER_DATA_PATH = r"C:\\Users\\user\\Desktop\\Energy Consumption\\data\\open-meteo-48.82N2.29E43m.csv"

# Verify both files exist before Spark starts
for _p in [POWER_DATA_PATH, WEATHER_DATA_PATH]:
    if not os.path.isfile(_p):
        print(f"[ERROR] Still not found: {_p}")
        print(f"        Directory contents: {os.listdir(os.path.dirname(_p))}")
        sys.exit(1)
    print(f"[FILE CHECK] {os.path.basename(_p)}")

# Column groups for power dataset
NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]
 
SEP = "=" * 72

# Spark session
def create_spark_session() -> SparkSession:
    import sys
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder
        .appName("M1_01_DataExploration_Local")
        .master("local[*]")
        .config("spark.driver.memory",              "4g")
        .config("spark.driver.maxResultSize",       "2g")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
        .config("spark.sql.shuffle.partitions",     "4")
        .config("spark.ui.showConsoleProgress",     "false")
        .config("spark.sql.files.maxPartitionBytes","134217728")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark
# Schemas
POWER_SCHEMA = StructType([
    StructField("Date",                  StringType(), True),
    StructField("Time",                  StringType(), True),
    StructField("Global_active_power",   StringType(), True),   # String first;
    StructField("Global_reactive_power", StringType(), True),   # missing values
    StructField("Voltage",               StringType(), True),   # are empty strings
    StructField("Global_intensity",      StringType(), True),   # not NULLs — cast
    StructField("Sub_metering_1",        StringType(), True),   # to Double in
    StructField("Sub_metering_2",        StringType(), True),   # ingestion script
    StructField("Sub_metering_3",        StringType(), True),
])

# Data loading
def load_power_data(spark: SparkSession):
     return (
        spark.read
        .option("header", "true")
        .option("delimiter", ";")
        .option("nullValue","?")
        .schema(POWER_SCHEMA)
        .csv(POWER_DATA_PATH)
    )

def load_weather_data(spark: SparkSession):
    import pandas as pd
    print(f"[DEBUG] Trying to read: {WEATHER_DATA_PATH}")
    pdf = pd.read_csv(WEATHER_DATA_PATH, skiprows=3)
    pdf.columns = [
        c.split("(")[0].strip().replace(" ", "_").replace("-", "_").lower()
        for c in pdf.columns
    ]
    return spark.createDataFrame(pdf)

# Analysis modules
def analyse_volume(df_power, df_weather):
     power_count   = df_power.count()
     weather_count = df_weather.count()

     print(f"{SEP}\nVolume Analysis\n{SEP}")
     print(f"Power dataset record count: {power_count}")
     print(f"Weather dataset record count: {weather_count}")
     
     return power_count, weather_count

def analyse_velocity(df_power, df_weather):
     date_range = df_power.agg(
        F.min("Date").alias("first"),
        F.max("Date").alias("last"),
    ).collect()[0]
     
     print(f"{SEP}\nVelocity Analysis\n{SEP}")
     print(f"Power dataset date range: {date_range['first']} to {date_range['last']}")

def analyse_variety(df_power, df_weather):
     print("\n  [Power Consumption Dataset — Schema]")
     df_power.printSchema()

     print("\n  [Weather Dataset — Schema]")
     df_weather.printSchema()

def analyse_missing_values(df_power):
    total = df_power.count()
    print(f"{SEP}\nMissing Values Analysis\n{SEP}")
    for col in NUMERIC_COLS:
        missing = df_power.filter(
            F.col(col).isNull() |
            (F.trim(F.col(col)) == "") |
            (F.trim(F.col(col)) == "?")
        ).count()
        pct = (missing / total) * 100
        print(f"{col}: {missing} missing ({pct:.2f}%)")

def analyse_statistics(df_power): 
    df_numeric = df_power
    for col in NUMERIC_COLS:
        df_numeric = df_numeric.withColumn(
            col,
            F.when(F.trim(F.col(col)) == "", None)
             .otherwise(F.col(col).cast(DoubleType()))
        )

    df_numeric.select(NUMERIC_COLS).describe().show(truncate=False)

def analyse_temporal_distribution(df_power):
    df_dated = df_power.withColumn(
        "Date_parsed",
        F.to_date(F.col("Date"), "d/M/yyyy")
    ).filter(F.col("Date_parsed").isNotNull())
 
    print("\n  Records per year:")
    df_dated.withColumn("Year", F.year("Date_parsed")) \
        .groupBy("Year").count() \
        .orderBy("Year") \
        .show()
    
    print("\n  Records per month:")
    df_dated.withColumn("Month", F.month("Date_parsed")) \
        .groupBy("Month").count() \
        .orderBy("Month") \
        .show()


# Main execution
def main():
    spark = create_spark_session()
# load datasets
    df_power = load_power_data(spark)
    df_weather = load_weather_data(spark)
    print(f"{SEP}\nData loaded successfully\n{SEP}")

# perform analyses
    n, m = analyse_volume(df_power, df_weather)
    analyse_velocity(df_power, df_weather)
    analyse_variety(df_power, df_weather)
    analyse_missing_values(df_power)
    analyse_statistics(df_power)
    analyse_temporal_distribution(df_power)

    spark.stop()

if __name__ == "__main__":
    main()
    


     


