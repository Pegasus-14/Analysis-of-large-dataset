import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

BATCH_OUTPUT   = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\batch"
WEATHER_CSV    = r"C:\Users\user\Desktop\Energy Consumption\data\open-meteo-48.82N2.29E43m.csv"
ML_OUTPUT      = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\ml"
FEATURE_OUTPUT = os.path.join(ML_OUTPUT, "features")
 
SEP = "=" * 72

# load batch outputs
def load_daily_with_weather():
    path = os.path.join(BATCH_OUTPUT, "daily_with_weather")
 
    if os.path.isdir(path):
        print(f"\n[LOAD] daily_with_weather from Milestone 2 batch output ...")
        table = pq.read_table(path)
        df    = table.to_pandas()
        print(f"  ✓ {len(df):,} rows | columns: {list(df.columns)}")
        return df
    else:
        print(f"\n[LOAD] Batch output not found — reconstructing from raw sources ...")
        return reconstruct_from_raw()

def reconstruct_from_raw():
    cleaned_path = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\processed\power_cleaned"
 
    if not os.path.isdir(cleaned_path):
        print(f"[ERROR] Cleaned Parquet not found at {cleaned_path}")
        print("        Run 02_data_ingestion_local.py first.")
        sys.exit(1)
 
    # Power data
    print("  Reading cleaned power data ...")
    power = pq.read_table(cleaned_path).to_pandas()
    power["Datetime"] = pd.to_datetime(power["Datetime"])
    power["Date"]     = power["Datetime"].dt.date
 
    daily = (
        power.groupby("Date")
        .agg(
            total_kwh         = ("Global_active_power", lambda x: x.sum() / 60.0),
            avg_kw            = ("Global_active_power", "mean"),
            peak_kw           = ("Global_active_power", "max"),
            minutes_recorded  = ("Global_active_power", "count"),
            kitchen_kwh       = ("Sub_metering_1",      lambda x: x.sum() / 1000.0),
            laundry_kwh       = ("Sub_metering_2",      lambda x: x.sum() / 1000.0),
            hvac_kwh          = ("Sub_metering_3",      lambda x: x.sum() / 1000.0),
            other_kwh         = ("Sub_metering_remainder", lambda x: x.sum() / 1000.0),
        )
        .reset_index()
    )
    daily["Date"] = pd.to_datetime(daily["Date"])
 
    # Weather data
    print("  Reading weather data ...")
    weather = pd.read_csv(WEATHER_CSV, skiprows=3)
    weather.columns = [
        c.split("(")[0].strip().replace(" ", "_").replace("-", "_").lower()
        for c in weather.columns
    ]
    weather = weather.rename(columns={"time": "Date"})
    weather["Date"] = pd.to_datetime(weather["Date"])
    weather = weather.rename(columns={
        "temperature_2m_mean": "temp_mean_c",
        "temperature_2m_max":  "temp_max_c",
        "temperature_2m_min":  "temp_min_c",
        "precipitation_sum":   "precip_mm",
    })
 
    df = daily.merge(weather[["Date","temp_mean_c","temp_max_c",
                               "temp_min_c","precip_mm"]], on="Date", how="left")
    print(f"  ✓ Reconstructed {len(df):,} daily records")
    return df
 
 
def load_submetering():
    """Load sub-metering daily breakdown from Milestone 2 if available."""
    path = os.path.join(BATCH_OUTPUT, "submetering_daily")
    if os.path.isdir(path):
        table = pq.read_table(path)
        df    = table.to_pandas()
        df["Date"] = pd.to_datetime(df["Date"].astype(str))
        return df[["Date", "kitchen_kwh", "laundry_kwh", "hvac_kwh", "other_kwh"]]
    return None

# Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
 
    original_rows = len(df)
    print(f"\n  Base rows: {original_rows:,}")

    # Group 1: Temporal features
    print("\n  [1/6] Temporal features ...")
 
    df["day_of_week"]  = df["Date"].dt.dayofweek        # 0=Mon, 6=Sun
    df["day_of_year"]  = df["Date"].dt.dayofyear
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["Date"].dt.month
    df["year"]         = df["Date"].dt.year
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
 
    # Season: 0=Winter, 1=Spring, 2=Summer, 3=Autumn
    df["season"] = df["month"].map({
        12: 0, 1: 0, 2: 0,    # Winter
        3: 1,  4: 1, 5: 1,    # Spring
        6: 2,  7: 2, 8: 2,    # Summer
        9: 3,  10: 3, 11: 3   # Autumn
    })
 
    # Cyclical encoding — preserves circular continuity
    # e.g. Sunday (6) and Monday (0) are adjacent, not far apart
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)
 
    print(f"Added: day_of_week, day_of_year, week_of_year, month, year,")
    print(f" is_weekend, season, dow_sin/cos, month_sin/cos, doy_sin/cos")

    # Group 2: Weather features
    print("\n  [2/6] Weather features ...")
 
    df["temp_range"]  = df["temp_max_c"] - df["temp_min_c"]
 
    # Heating Degree Days (HDD): energy needed for heating
    # Base temperature 18°C — standard for European residential
    df["hdd"] = np.maximum(0, 18.0 - df["temp_mean_c"])
 
    # Cooling Degree Days (CDD): energy needed for cooling
    # Base temperature 22°C
    df["cdd"] = np.maximum(0, df["temp_mean_c"] - 22.0)
 
    # Comfort zone flag: 16-22°C → minimal HVAC usage
    df["in_comfort_zone"] = (
        (df["temp_mean_c"] >= 16) & (df["temp_mean_c"] <= 22)
    ).astype(int)
 
    # Precipitation indicator
    df["has_rain"] = (df["precip_mm"] > 0.1).astype(int)
 
    print(f"    Added: temp_range, hdd, cdd, in_comfort_zone, has_rain")

    # Group 3: Lag features
    print("\n  [3/6] Lag features (no data leakage — past values only) ...")
 
    df["lag_1_kwh"]  = df["total_kwh"].shift(1)   # yesterday
    df["lag_2_kwh"]  = df["total_kwh"].shift(2)
    df["lag_7_kwh"]  = df["total_kwh"].shift(7)   # same day last week
    df["lag_14_kwh"] = df["total_kwh"].shift(14)
    df["lag_30_kwh"] = df["total_kwh"].shift(30)
 
    # Lag weather (yesterday's temperature)
    df["lag_1_temp"] = df["temp_mean_c"].shift(1)
 
    print(f"    Added: lag_1, lag_2, lag_7, lag_14, lag_30 kWh; lag_1_temp")

    # Group 4: Rolling window features
    print("\n  [4/6] Rolling statistics ...")
 
    # min_periods ensures we don't produce NaN for the full window
    df["roll_7_mean"]  = df["total_kwh"].shift(1).rolling(7,  min_periods=3).mean()
    df["roll_7_std"]   = df["total_kwh"].shift(1).rolling(7,  min_periods=3).std()
    df["roll_7_max"]   = df["total_kwh"].shift(1).rolling(7,  min_periods=3).max()
    df["roll_30_mean"] = df["total_kwh"].shift(1).rolling(30, min_periods=7).mean()
    df["roll_30_std"]  = df["total_kwh"].shift(1).rolling(30, min_periods=7).std()
 
    # Week-over-week change
    df["wow_change"] = df["total_kwh"].shift(1) - df["total_kwh"].shift(8)
 
    print(f"    Added: roll_7_mean/std/max, roll_30_mean/std, wow_change")

    # Group 5: Sub-metering ratios
    print("\n  [5/6] Sub-metering ratios ...")
 
    sm_cols = ["kitchen_kwh", "laundry_kwh", "hvac_kwh", "other_kwh"]
    if all(c in df.columns for c in sm_cols):
        safe_total = df["total_kwh"].replace(0, np.nan)
        df["kitchen_ratio"] = df["kitchen_kwh"] / safe_total
        df["laundry_ratio"] = df["laundry_kwh"] / safe_total
        df["hvac_ratio"]    = df["hvac_kwh"]    / safe_total
        df["other_ratio"]   = df["other_kwh"]   / safe_total
 
        # Lag sub-metering ratios (yesterday's appliance usage pattern)
        df["lag_1_hvac_ratio"] = df["hvac_ratio"].shift(1)
        print(f"    Added: kitchen/laundry/hvac/other_ratio, lag_1_hvac_ratio")
    else:
        print(f"    [SKIP] Sub-metering columns not in dataset — skipping ratios")

    # Group 6: Interaction features
    print("\n  [6/6] Interaction features ...")
 
    df["hdd_x_weekend"]  = df["hdd"] * df["is_weekend"]
    df["hdd_x_season"]   = df["hdd"] * df["season"]
    df["temp_x_weekend"] = df["temp_mean_c"] * df["is_weekend"]
 
    print(f"    Added: hdd_x_weekend, hdd_x_season, temp_x_weekend")
 
    # ── Drop rows with NaN target or insufficient lag history ─────────────
    df_clean = df.dropna(subset=["total_kwh", "lag_7_kwh", "roll_7_mean"])
    dropped  = original_rows - len(df_clean)
 
    print(f"\n  Rows before feature engineering : {original_rows:,}")
    print(f"  Rows dropped (insufficient lag) : {dropped:,}  "
          f"(first 30 days — no lag history)")
    print(f"  Rows after feature engineering  : {len(df_clean):,}")
 
    return df_clean

# feature summary
def print_feature_summary(df: pd.DataFrame):
    feature_groups = {
        "Temporal"     : ["day_of_week","day_of_year","week_of_year","month",
                        "year","is_weekend","season","dow_sin","dow_cos",
                        "month_sin","month_cos","doy_sin","doy_cos"],
        "Weather"      : ["temp_mean_c","temp_max_c","temp_min_c","precip_mm",
                        "temp_range","hdd","cdd","in_comfort_zone","has_rain"],
        "Lag"          : ["lag_1_kwh","lag_2_kwh","lag_7_kwh","lag_14_kwh",
                        "lag_30_kwh","lag_1_temp"],
        "Rolling"      : ["roll_7_mean","roll_7_std","roll_7_max",
                        "roll_30_mean","roll_30_std","wow_change"],
        "Sub-metering" : ["kitchen_ratio","laundry_ratio","hvac_ratio",
                        "other_ratio","lag_1_hvac_ratio"],
        "Interaction"  : ["hdd_x_weekend","hdd_x_season","temp_x_weekend"],
    }

    total_features = 0
    for group, cols in feature_groups.items():
        available = [c for c in cols if c in df.columns]
    total_features += len(available)
    print(f"\n  {group} ({len(available)} features):")
    for c in available:
        missing = df[c].isna().sum()
        print(f"    {c:<25} "
            f"mean={df[c].mean():>8.3f}  "
            f"std={df[c].std():>7.3f}  "
            f"missing={missing}")

    print(f"\n  Total features : {total_features}")
    print(f"  Target column  : total_kwh")
    print(f"  Date range     : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Total rows     : {len(df):,}")

# Correlation with target
    print(f"\n  Top 10 features by absolute correlation with total_kwh:")
    feature_cols = [c for g in feature_groups.values() for c in g if c in df.columns]
    corr = df[feature_cols + ["total_kwh"]].corr()["total_kwh"].drop("total_kwh")
    top10 = corr.abs().sort_values(ascending=False).head(10)
    for feat, val in top10.items():
        direction = "↑" if corr[feat] > 0 else "↓"
        print(f"    {feat:<25} r={corr[feat]:>+7.4f}  {direction}")
    
# Feature table 
def save_feature_table(df: pd.DataFrame):
    os.makedirs(FEATURE_OUTPUT, exist_ok=True)
    out_path = os.path.join(FEATURE_OUTPUT, "feature_table.parquet")
 
    table = pa.Table.from_pandas(df, preserve_index=False)
    pa.parquet.write_table(table, out_path)
 
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n[SAVE] Feature table written:")
    print(f"  Path  : {out_path}")
    print(f"  Rows  : {len(df):,}")
    print(f"  Cols  : {len(df.columns)}")
    print(f"  Size  : {size_mb:.2f} MB")

# Main execution
def main():
    t_start = time.time()
    df_daily = load_daily_with_weather()
    df_sub   = load_submetering()

    if df_sub is not None:
        sm_cols = ["kitchen_kwh","laundry_kwh","hvac_kwh","other_kwh"]
        if not all(c in df_daily.columns for c in sm_cols):
            df_daily["Date"] = pd.to_datetime(df_daily["Date"].astype(str))
            df_sub["Date"]   = pd.to_datetime(df_sub["Date"].astype(str))
            df_daily = df_daily.merge(df_sub, on="Date", how="left")
            print(f"  ✓ Sub-metering joined")
 
    # Engineer features
    df_features = engineer_features(df_daily)
 
    # Summary
    print_feature_summary(df_features)
 
    # Save
    save_feature_table(df_features)
 
    elapsed = time.time() - t_start

if __name__ == "__main__":
    main()