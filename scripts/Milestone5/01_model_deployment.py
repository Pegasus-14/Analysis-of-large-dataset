import os
import sys
import time
import json
import pickle
import io
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# configuration
ML_OUTPUT      = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\ml"
MODEL_PATH     = os.path.join(ML_OUTPUT, "models", "best_model.pkl")
FEATURE_PATH   = os.path.join(ML_OUTPUT, "features", "feature_table.parquet")
 
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS   = "admin"
MINIO_SECRET   = "password123"
BUCKET         = "energy-lake"
 
MODEL_VERSION  = "v1.0"
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
        print(f"MinIO connected at {MINIO_ENDPOINT}")
        return s3
    except Exception as e:
        print(f"  [WARN] MinIO unavailable: {e}")
        return None
 
 
def write_parquet_to_minio(s3, key: str, df: pd.DataFrame, label: str):
    if s3 is None or df is None:
        return
    try:
        buf   = io.BytesIO()
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, buf)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key=key,
                      Body=buf.getvalue(),
                      ContentType="application/octet-stream")
        print(f"s3://{BUCKET}/{key}  ({len(df):,} rows)")
    except Exception as e:
        print(f"  [WARN] MinIO write failed ({label}): {e}")
 
 
def write_json_to_minio(s3, key: str, data: dict):
    if s3 is None:
        return
    try:
        s3.put_object(
            Bucket      = BUCKET,
            Key         = key,
            Body        = json.dumps(data, indent=2, default=str).encode(),
            ContentType = "application/json",
        )
        print(f"s3://{BUCKET}/{key}")
    except Exception as e:
        print(f"  [WARN] MinIO JSON write failed: {e}")
 
 
def write_bytes_to_minio(s3, key: str, data: bytes, label: str):
    if s3 is None:
        return
    try:
        s3.put_object(Bucket=BUCKET, Key=key, Body=data,
                      ContentType="application/octet-stream")
        print(f"s3://{BUCKET}/{key}  ({len(data)/1024:.1f} KB)  [{label}]")
    except Exception as e:
        print(f"  [WARN] MinIO bytes write failed: {e}")

# loading model
def load_model():
    print(f"\n[LOAD] Best model from {MODEL_PATH} ...")
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Not found. Run m4_02_model_training.py first.")
        sys.exit(1)
 
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
 
    model        = payload["model"]
    feature_cols = payload["feature_cols"]
    metrics      = payload["metrics"]
    trained_at   = payload.get("trained_at", "unknown")
 
    model_name = type(model).__name__
    print(f"  ✓ Model     : {model_name}")
    print(f"  ✓ Features  : {len(feature_cols)}")
    print(f"  ✓ Trained   : {trained_at}")
    print(f"  ✓ Test R²   : {metrics.get('R2', 'N/A'):.4f}")
    print(f"  ✓ Test MAE  : {metrics.get('MAE', 'N/A'):.4f} kWh")
 
    return model, feature_cols, metrics, model_name

# Registering model in minio
def register_model_in_minio(s3, model, feature_cols: list,metrics: dict, model_name: str):
    if s3 is None:
        print("  [SKIP] MinIO not available")
        return
 
    # Upload model binary
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    write_bytes_to_minio(
        s3, f"model-registry/{MODEL_VERSION}/model.pkl",
        model_bytes, model_name
    )
 
    # Upload metadata
    metadata = {
        "version"       : MODEL_VERSION,
        "model_type"    : model_name,
        "registered_at" : time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics"       : metrics,
        "n_features"    : len(feature_cols),
        "target"        : "total_kwh",
        "training_data" : "2007-01-21 to 2009-11-09 (80% time split)",
        "test_data"     : "2009-11-10 to 2010-11-26 (20% held out)",
        "status"        : "production",
    }
    write_json_to_minio(
        s3, f"model-registry/{MODEL_VERSION}/metadata.json", metadata
    )
 
    # Upload feature schema
    feature_schema = {
        "version"      : MODEL_VERSION,
        "feature_cols" : feature_cols,
        "n_features"   : len(feature_cols),
    }
    write_json_to_minio(
        s3, f"model-registry/{MODEL_VERSION}/feature_schema.json", feature_schema
    )
 
    # Update latest pointer
    write_json_to_minio(
        s3, "model-registry/latest.json",
        {"latest_version": MODEL_VERSION, "updated_at": metadata["registered_at"]}
    )

# load features
def load_features():
    print(f"\n[LOAD] Feature table from {FEATURE_PATH} ...")
    if not os.path.isfile(FEATURE_PATH):
        print(f"[ERROR] Not found. Run m4_01_feature_engineering.py first.")
        sys.exit(1)
 
    df = pq.read_table(FEATURE_PATH).to_pandas()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  ✓ {len(df):,} rows | {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df

# Batch inference 
def run_batch_inference(model, df: pd.DataFrame,
                         feature_cols: list, metrics: dict):
    X = df[feature_cols].copy().fillna(df[feature_cols].median())
 
    print(f"\n  Running inference on {len(X):,} records ...")
    t0   = time.time()
    preds = model.predict(X)
    elapsed = time.time() - t0
    print(f"  ✓ {len(preds):,} predictions in {elapsed:.2f}s "
          f"({len(preds)/elapsed:.0f} predictions/sec)")
 
    # Prediction intervals
    mae = metrics.get("MAE", 2.0)
    model_name = type(model).__name__
 
    if hasattr(model, "estimators_"):
        # Random Forest: use std of tree predictions
        tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
        pred_std   = tree_preds.std(axis=0)
        lower      = preds - 1.96 * pred_std
        upper      = preds + 1.96 * pred_std
        interval_method = "95% interval from tree ensemble std"
    else:
        # GBM: use MAE-based interval
        lower = preds - 1.5 * mae
        upper = preds + 1.5 * mae
        interval_method = f"±1.5×MAE interval (MAE={mae:.3f} kWh)"
 
    print(f"  Interval method: {interval_method}")
 
    # Build output DataFrame
    df_preds = pd.DataFrame({
        "Date"           : df["Date"],
        "actual_kwh"     : df["total_kwh"],
        "predicted_kwh"  : np.round(preds, 4),
        "lower_bound"    : np.round(np.maximum(lower, 0), 4),
        "upper_bound"    : np.round(upper, 4),
        "abs_error"      : np.round(np.abs(df["total_kwh"] - preds), 4),
        "pct_error"      : np.round(
            np.abs(df["total_kwh"] - preds) /
            np.maximum(df["total_kwh"].abs(), 0.1) * 100, 2
        ),
        "Year"           : df["Year"],
        "Month"          : df["Month"],
        "model_version"  : MODEL_VERSION,
    })
 
    # Summary statistics
    within_interval = (
        (df_preds["actual_kwh"] >= df_preds["lower_bound"]) &
        (df_preds["actual_kwh"] <= df_preds["upper_bound"])
    ).mean() * 100
 
    print(f"""
  Inference summary:
    Records scored         : {len(df_preds):,}
    Mean predicted kWh/day : {df_preds['predicted_kwh'].mean():.3f}
    Mean actual kWh/day    : {df_preds['actual_kwh'].mean():.3f}
    Mean absolute error    : {df_preds['abs_error'].mean():.4f} kWh
    Mean pct error         : {df_preds['pct_error'].mean():.2f}%
    Actuals within interval: {within_interval:.1f}%
    """)
 
    # Sample output
    print("  Sample predictions (10 random days):")
    sample = df_preds.sample(10, random_state=42).sort_values("Date")
    print(f"  {'Date':<12} {'Actual':>8} {'Pred':>8} {'Lower':>8} "
          f"{'Upper':>8} {'Err%':>7}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
    for _, row in sample.iterrows():
        print(f"  {str(row['Date'].date()):<12} "
              f"{row['actual_kwh']:>8.3f} "
              f"{row['predicted_kwh']:>8.3f} "
              f"{row['lower_bound']:>8.3f} "
              f"{row['upper_bound']:>8.3f} "
              f"{row['pct_error']:>6.1f}%")
 
    return df_preds

# Predictions
def save_predictions(s3, df_preds: pd.DataFrame):
    write_parquet_to_minio(
        s3,
        f"serving/predictions/{MODEL_VERSION}/all_predictions.parquet",
        df_preds,
        "full predictions"
    )
 
    # Save monthly aggregated predictions (lighter for dashboard queries)
    df_monthly = (
        df_preds.groupby(["Year", "Month"])
        .agg(
            actual_kwh_sum   = ("actual_kwh",    "sum"),
            predicted_kwh_sum= ("predicted_kwh", "sum"),
            mean_pct_error   = ("pct_error",     "mean"),
            n_days           = ("actual_kwh",    "count"),
        )
        .reset_index()
    )
    write_parquet_to_minio(
        s3,
        f"serving/predictions/{MODEL_VERSION}/monthly_aggregated.parquet",
        df_monthly,
        "monthly aggregated"
    )
 
    # Also save locally for Milestone 5 monitoring script
    local_pred_dir = os.path.join(
        r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\ml",
        "predictions"
    )
    os.makedirs(local_pred_dir, exist_ok=True)
    out_path = os.path.join(local_pred_dir, "all_predictions.parquet")
    pq.write_table(
        pa.Table.from_pandas(df_preds, preserve_index=False), out_path
    )

# main execution
def main():
    t_start = time.time()
 
    s3 = create_minio_client()
    model, feature_cols, metrics, model_name = load_model()
    register_model_in_minio(s3, model, feature_cols, metrics, model_name)
    df = load_features()
    df_preds = run_batch_inference(model, df, feature_cols, metrics)
    save_predictions(s3, df_preds)
    elapsed = time.time() - t_start
    print(f"\nTotal deployment time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()