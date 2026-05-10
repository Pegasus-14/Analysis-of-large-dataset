import os
import sys
import time
import json
import io
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
 
# Configuration
ML_OUTPUT      = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\ml"
FEATURE_PATH   = os.path.join(ML_OUTPUT, "features", "feature_table.parquet")
PRED_PATH      = os.path.join(ML_OUTPUT, "predictions", "all_predictions.parquet")
BATCH_OUTPUT   = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\batch"
 
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS   = "admin"
MINIO_SECRET   = "password123"
BUCKET         = "energy-lake"
 
# Monitoring thresholds
MAE_ALERT_THRESHOLD  = 5.0    # alert if rolling MAE exceeds 5 kWh
MAPE_ALERT_THRESHOLD = 15.0   # alert if rolling MAPE exceeds 15%
 
# Drift detection thresholds
PSI_WARNING  = 0.10   # PSI > 0.10 = moderate drift
PSI_CRITICAL = 0.25   # PSI > 0.25 = significant drift, consider retraining
KS_PVALUE    = 0.05   # p-value threshold for KS test
 
# Train/test split ratio (must match m4_02_model_training.py)
TRAIN_RATIO  = 0.80
 
SEP = "=" * 72

# minio client
def create_minio_client():
    import boto3
    from botocore.client import Config
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url          = MINIO_ENDPOINT,
            aws_access_key_id     = MINIO_ACCESS,
            aws_secret_access_key = MINIO_SECRET,
            config                = Config(signature_version="s3v4"),
            region_name           = "us-east-1",
        )
        s3.list_buckets()
        print(f"  ✓ MinIO connected")
        return s3
    except Exception as e:
        print(f"  [WARN] MinIO unavailable: {e}")
        return None
 
 
def write_json_to_minio(s3, key: str, data: dict):
    if s3 is None:
        return
    try:
        s3.put_object(
            Bucket=BUCKET, Key=key,
            Body=json.dumps(data, indent=2, default=str).encode(),
            ContentType="application/json"
        )
        print(f"  → s3://{BUCKET}/{key}")
    except Exception as e:
        print(f"  [WARN] {e}")
 
 
def write_parquet_to_minio(s3, key: str, df: pd.DataFrame):
    if s3 is None or df is None:
        return
    try:
        buf   = io.BytesIO()
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, buf)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue(),
                      ContentType="application/octet-stream")
        print(f"  → s3://{BUCKET}/{key}  ({len(df):,} rows)")
    except Exception as e:
        print(f"  [WARN] {e}")

# Load data
def load_data():
    print(f"\n[LOAD] Features and predictions ...")
 
    if not os.path.isfile(FEATURE_PATH):
        print(f"[ERROR] Features not found. Run m4_01_feature_engineering.py first.")
        sys.exit(1)
    if not os.path.isfile(PRED_PATH):
        print(f"[ERROR] Predictions not found. Run m5_01_model_deployment.py first.")
        sys.exit(1)
 
    df_features = pq.read_table(FEATURE_PATH).to_pandas()
    df_preds    = pq.read_table(PRED_PATH).to_pandas()
 
    df_features["Date"] = pd.to_datetime(df_features["Date"])
    df_preds["Date"]    = pd.to_datetime(df_preds["Date"])
 
    df_features = df_features.sort_values("Date").reset_index(drop=True)
    df_preds    = df_preds.sort_values("Date").reset_index(drop=True)
 
    n_train = int(len(df_features) * TRAIN_RATIO)
 
    df_train = df_features.iloc[:n_train].copy()
    df_test  = df_features.iloc[n_train:].copy()
 
    print(f"  ✓ Features : {len(df_features):,} rows")
    print(f"  ✓ Predictions: {len(df_preds):,} rows")
    print(f"  ✓ Train set: {len(df_train):,} rows | "
          f"Test set: {len(df_test):,} rows")
 
    return df_features, df_preds, df_train, df_test

# Prediction monitoring
def monitor_predictions(df_preds: pd.DataFrame, s3):
    df = df_preds.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
 
    # 30-day rolling metrics
    df["rolling_mae"]  = df["abs_error"].rolling(30,  min_periods=7).mean()
    df["rolling_mape"] = df["pct_error"].rolling(30,  min_periods=7).mean()
 
    # Flag alerts
    df["mae_alert"]  = df["rolling_mae"]  > MAE_ALERT_THRESHOLD
    df["mape_alert"] = df["rolling_mape"] > MAPE_ALERT_THRESHOLD
 
    n_mae_alerts  = df["mae_alert"].sum()
    n_mape_alerts = df["mape_alert"].sum()
 
    print(f"""
  Monitoring window : 30-day rolling
  MAE threshold     : {MAE_ALERT_THRESHOLD} kWh  → {n_mae_alerts} alert periods
  MAPE threshold    : {MAPE_ALERT_THRESHOLD}%     → {n_mape_alerts} alert periods
 
  Monthly monitoring summary:""")
 
    monthly = (
        df.groupby(["Year","Month"])
        .agg(
            avg_mae       = ("abs_error",   "mean"),
            avg_mape      = ("pct_error",   "mean"),
            n_mae_alerts  = ("mae_alert",   "sum"),
            n_mape_alerts = ("mape_alert",  "sum"),
            n_days        = ("abs_error",   "count"),
        )
        .reset_index()
    )
 
    print(f"\n  {'Year-Mon':<10} {'Avg MAE':>8} {'Avg MAPE':>9} "
          f"{'MAE Alrt':>9} {'MAPE Alrt':>10}")
    print(f"  {'─'*10} {'─'*8} {'─'*9} {'─'*9} {'─'*10}")
    for _, row in monthly.iterrows():
        flag = "Alert" if row["n_mae_alerts"] > 5 or row["n_mape_alerts"] > 5 else ""
        print(f"  {int(row['Year'])}-{int(row['Month']):02d}     "
              f"{row['avg_mae']:>8.3f} {row['avg_mape']:>8.2f}% "
              f"{int(row['n_mae_alerts']):>9} {int(row['n_mape_alerts']):>10}{flag}")
 
    # Worst 5 prediction days
    print(f"\n 5 worst prediction days:")
    worst = df.nlargest(5, "abs_error")[
        ["Date","actual_kwh","predicted_kwh","abs_error","pct_error"]
    ]
    for _, row in worst.iterrows():
        print(f"    {str(row['Date'].date())}: "
              f"actual={row['actual_kwh']:.2f}  "
              f"pred={row['predicted_kwh']:.2f}  "
              f"err={row['abs_error']:.2f} kWh  ({row['pct_error']:.1f}%)")
 
    # Save monitoring log to MinIO
    write_parquet_to_minio(
        s3, "serving/monitoring/rolling_metrics.parquet",
        df[["Date","rolling_mae","rolling_mape","mae_alert","mape_alert"]].dropna()
    )
 
    return df

# Data drift detection
def compute_psi(train_vals: np.ndarray, test_vals: np.ndarray,n_bins: int = 10) -> float:
    _, bin_edges = np.histogram(train_vals, bins=n_bins)
    bin_edges[0]  -= 1e-6    # include minimum value
    bin_edges[-1] += 1e-6    # include maximum value
 
    train_counts = np.histogram(train_vals, bins=bin_edges)[0]
    test_counts  = np.histogram(test_vals,  bins=bin_edges)[0]
 
    # Convert to percentages; avoid division by zero
    train_pct = np.where(train_counts == 0, 1e-6,
                         train_counts / len(train_vals))
    test_pct  = np.where(test_counts  == 0, 1e-6,
                         test_counts  / len(test_vals))
 
    psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
    return float(psi)
 
 
def detect_drift(df_train: pd.DataFrame, df_test: pd.DataFrame, s3):
    from scipy.stats import ks_2samp
 
    key_features = [
        "temp_mean_c", "hdd", "cdd", "lag_1_kwh",
        "roll_7_mean", "roll_30_mean",
        "is_weekend", "month",
    ]
    # Keep only features that exist in both DataFrames
    key_features = [f for f in key_features
                    if f in df_train.columns and f in df_test.columns]
 
    drift_results = []
 
    print(f"\n  {'Feature':<22} {'PSI':>8} {'Status':<18} "
          f"{'KS p-value':>11} {'KS Result':<12}")
    print(f"  {'─'*22} {'─'*8} {'─'*18} {'─'*11} {'─'*12}")
 
    for feat in key_features:
        train_vals = df_train[feat].dropna().values
        test_vals  = df_test[feat].dropna().values
 
        if len(train_vals) < 10 or len(test_vals) < 10:
            continue
 
        psi = compute_psi(train_vals, test_vals)
        ks_stat, ks_pval = ks_2samp(train_vals, test_vals)
 
        if psi < PSI_WARNING:
            psi_status = "✓ No drift"
        elif psi < PSI_CRITICAL:
            psi_status = "⚠ Moderate drift"
        else:
            psi_status = "✗ Significant drift"
 
        ks_result = "drift detected" if ks_pval < KS_PVALUE else "no drift"
 
        print(f"  {feat:<22} {psi:>8.4f} {psi_status:<18} "
              f"{ks_pval:>11.4f} {ks_result:<12}")
 
        drift_results.append({
            "feature"    : feat,
            "psi"        : round(psi, 4),
            "psi_status" : psi_status,
            "ks_pvalue"  : round(float(ks_pval), 4),
            "ks_result"  : ks_result,
        })
    write_json_to_minio(
        s3, "serving/monitoring/drift_report.json",
        {
            "run_at"          : time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_version"   : "v1.0",
            "features_checked": len(drift_results),
            "results"         : drift_results,
        }
    )
 
    return drift_results

# Pipeline optimization
def document_pipeline_optimisations(df_features: pd.DataFrame):
    t0 = time.time()
    _ = pq.read_table(FEATURE_PATH).to_pandas()
    cold_read_ms = (time.time() - t0) * 1000
 
    # Measure feature table read time (warm — OS file cache)
    t0 = time.time()
    _ = pq.read_table(FEATURE_PATH).to_pandas()
    warm_read_ms = (time.time() - t0) * 1000

# main execution
def main():
    t_start = time.time()
 
    s3 = create_minio_client()
 
    df_features, df_preds, df_train, df_test = load_data()
 
    df_monitored  = monitor_predictions(df_preds, s3)
    drift_results = detect_drift(df_train, df_test, s3)
    document_pipeline_optimisations(df_features)
    elapsed = time.time() - t_start
    print(f"\n[SUMMARY] Monitoring and drift detection completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()