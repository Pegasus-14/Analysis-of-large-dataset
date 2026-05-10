import os
import sys
import time
import json
import io
from collections import deque, defaultdict
from datetime import datetime, timedelta

# Configuring
KAFKA_BROKER         = "localhost:9092"
KAFKA_TOPIC          = "energy-readings"
CONSUMER_GROUP       = "energy-analytics-consumer"

MINIO_ENDPOINT       = "http://localhost:9000"
MINIO_ACCESS_KEY     = "admin"
MINIO_SECRET_KEY     = "password123"
BUCKET               = "energy-lake"
 
TRIGGER_INTERVAL_SEC = 10     # compute analytics every 10 seconds
TUMBLING_WINDOW_MIN  = 5      # 5-minute tumbling window
SLIDING_WINDOW_MIN   = 10     # 10-minute sliding window
SLIDING_STEP_MIN     = 5      # slide every 5 minutes
WATERMARK_MIN        = 10     # drop events older than 10 minutes
STREAM_TIMEOUT_SEC   = 600    # stop after 600s (None = run forever)
IDLE_STOP_SEC     = 200     # stop if no messages received for 30s (None = wait indefinitely)
 
SEP = "=" * 72

# MinIO Client
def create_minio_client():
    import boto3
    from botocore.client import Config
    from botocore.exceptions import EndpointConnectionError
 
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url          = MINIO_ENDPOINT,
            aws_access_key_id     = MINIO_ACCESS_KEY,
            aws_secret_access_key = MINIO_SECRET_KEY,
            config                = Config(signature_version="s3v4"),
            region_name           = "us-east-1",
        )
        s3.list_buckets()   # connection test
        print(f"Connected to MinIO at {MINIO_ENDPOINT}")
        return s3
 
    except Exception as e:
        print(f"""
[ERROR] Cannot connect to MinIO: {e}""")
        return None
 
 
def ensure_streaming_prefix(s3):
    """Creating streaming prefix structure in the energy-lake bucket."""
    if s3 is None:
        return
    prefixes = [
        "streaming/tumbling_windows/",
        "streaming/sliding_windows/",
        "streaming/submetering/",
        "streaming/session_summary/",
    ]
    for prefix in prefixes:
        s3.put_object(Bucket=BUCKET, Key=prefix, Body=b"")
    print(f"MinIO prefix s3://{BUCKET}/streaming/ ready")
 
 
def write_to_minio(s3, key: str, df, label: str):
    if s3 is None or df is None or len(df) == 0:
        return
 
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
 
        buf   = io.BytesIO()
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, buf)
        buf.seek(0)
 
        s3.put_object(
            Bucket      = BUCKET,
            Key         = key,
            Body        = buf.getvalue(),
            ContentType = "application/octet-stream",
        )
        print(f" MinIO: s3://{BUCKET}/{key}  ({len(df)} rows)")
 
    except Exception as e:
        print(f"  [WARN] MinIO write failed for {key}: {e}")
 
 
def write_json_to_minio(s3, key: str, data: dict):
    """Write a JSON summary object to MinIO."""
    if s3 is None:
        return
    try:
        s3.put_object(
            Bucket      = BUCKET,
            Key         = key,
            Body        = json.dumps(data, indent=2).encode("utf-8"),
            ContentType = "application/json",
        )
        print(f"MinIO: s3://{BUCKET}/{key}")
    except Exception as e:
        print(f"  [WARN] MinIO JSON write failed: {e}")
 
# Kafka Consumer
def create_consumer():
    from kafka import KafkaConsumer
    from kafka.errors import NoBrokersAvailable
 
    print(f"\n[CONNECT] Connecting to Kafka broker at {KAFKA_BROKER} ...")
 
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers      = KAFKA_BROKER,
            group_id               = CONSUMER_GROUP,
            auto_offset_reset      = "earliest",
            enable_auto_commit     = True,
            consumer_timeout_ms    = 1000,
            value_deserializer     = lambda v: json.loads(v.decode("utf-8")),
            key_deserializer       = lambda k: k.decode("utf-8") if k else None,
            session_timeout_ms     = 30000,
            heartbeat_interval_ms  = 10000,
        )
        print(f" Connected | Group: {CONSUMER_GROUP} | Topic: {KAFKA_TOPIC}")
        return consumer
    except NoBrokersAvailable:
        print(f" Failed to connect to Kafka broker at {KAFKA_BROKER}")
        sys.exit(1)

# Event buffer
class EventBuffer:
    def __init__(self, watermark_minutes: int):
        self.watermark_minutes = watermark_minutes
        self.buffer            = deque()
        self.total_received    = 0
        self.total_latency_ms  = 0.0
 
    def add(self, record: dict) -> float:
        now_ms      = int(time.time() * 1000)
        producer_ts = record.get("event_timestamp", now_ms)
        latency_ms  = now_ms - producer_ts
 
        self.buffer.append(record)
        self.total_received   += 1
        self.total_latency_ms += max(0, latency_ms)
 
        if len(self.buffer) > 12000:
            self.buffer.popleft()
 
        return latency_ms
 
    def get_records(self):
        return list(self.buffer)
 
    @property
    def avg_latency_ms(self):
        return self.total_latency_ms / max(self.total_received, 1)
    
# Windowed Aggregations
def compute_tumbling_windows(records: list, window_minutes: int):
    import pandas as pd
    if not records:
        return None
 
    pdf = pd.DataFrame(records)
    pdf["event_time"] = pd.to_datetime(pdf["Datetime"], errors="coerce")
    pdf["Global_active_power"] = pd.to_numeric(
        pdf["Global_active_power"], errors="coerce"
    ).fillna(0)
    pdf = pdf.dropna(subset=["event_time"])
    if pdf.empty:
        return None
 
    pdf["window_start"] = pdf["event_time"].dt.floor(f"{window_minutes}min")
    agg = (
        pdf.groupby("window_start")
        .agg(
            avg_kw    = ("Global_active_power", "mean"),
            peak_kw   = ("Global_active_power", "max"),
            total_kwh = ("Global_active_power", lambda x: x.sum() / 60.0),
            count     = ("Global_active_power", "count"),
        )
        .reset_index()
    )
    agg["window_start"] = agg["window_start"].astype(str)
    return agg
 
 
def compute_sliding_windows(records: list, window_minutes: int, step_minutes: int):
    import pandas as pd
    if not records:
        return None
 
    pdf = pd.DataFrame(records)
    pdf["event_time"] = pd.to_datetime(pdf["Datetime"], errors="coerce")
    pdf["Global_active_power"] = pd.to_numeric(
        pdf["Global_active_power"], errors="coerce"
    ).fillna(0)
    pdf = pdf.dropna(subset=["event_time"])
    if pdf.empty:
        return None
 
    min_time = pdf["event_time"].min().floor(f"{step_minutes}min")
    max_time = pdf["event_time"].max()
    windows  = []
    t = min_time
 
    while t <= max_time:
        w_end  = t + timedelta(minutes=window_minutes)
        w_data = pdf[
            (pdf["event_time"] >= t) & (pdf["event_time"] < w_end)
        ]["Global_active_power"]
        if len(w_data) > 0:
            windows.append({
                "window_start"    : str(t),
                "window_end"      : str(w_end),
                "sliding_avg_kw"  : round(float(w_data.mean()), 4),
                "sliding_peak_kw" : round(float(w_data.max()),  4),
                "count"           : int(len(w_data)),
            })
        t += timedelta(minutes=step_minutes)
 
    return pd.DataFrame(windows[-10:]) if windows else None
 
 
def compute_submetering(records: list):
    import pandas as pd
    if not records:
        return None
 
    pdf = pd.DataFrame(records)
    for col in ["Sub_metering_1","Sub_metering_2",
                "Sub_metering_3","Sub_metering_remainder","Global_active_power"]:
        pdf[col] = pd.to_numeric(pdf.get(col, 0), errors="coerce").fillna(0)
 
    total_kwh = pdf["Global_active_power"].sum() / 60.0
    result = pd.DataFrame([{
        "kitchen_kwh"  : round(pdf["Sub_metering_1"].sum()         / 1000.0, 3),
        "laundry_kwh"  : round(pdf["Sub_metering_2"].sum()         / 1000.0, 3),
        "hvac_kwh"     : round(pdf["Sub_metering_3"].sum()         / 1000.0, 3),
        "other_kwh"    : round(pdf["Sub_metering_remainder"].sum() / 1000.0, 3),
        "total_kwh"    : round(total_kwh, 3),
        "records"      : len(pdf),
        "avg_latency_ms": round(0, 1),
    }])
    return result

# Display 
def display_results(tumbling, sliding, submetering, trigger_n, buffer, elapsed):
    print(f"\n{'─'*72}")
    print(f"  TRIGGER {trigger_n}  |  "
          f"total_received={buffer.total_received:,}  |  "
          f"rate={buffer.total_received/max(elapsed,0.001):.1f} rec/s  |  "
          f"avg_latency={buffer.avg_latency_ms:.0f}ms")
    print(f"{'─'*72}")
 
    if tumbling is not None and len(tumbling) > 0:
        print(f"\n  5-MIN TUMBLING WINDOWS:")
        print(f"  {'Window Start':<22} {'Avg kW':>8} {'Peak kW':>8} "
              f"{'kWh':>8} {'Count':>6}")
        print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
        for _, row in tumbling.tail(6).iterrows():
            print(f"  {str(row['window_start']):<22} "
                  f"{row['avg_kw']:>8.3f} {row['peak_kw']:>8.3f} "
                  f"{row['total_kwh']:>8.4f} {row['count']:>6,}")
 
    if sliding is not None and len(sliding) > 0:
        print(f"\n  10-MIN SLIDING WINDOWS (5-min step):")
        print(f"  {'Window Start':<22} {'Sliding Avg':>11} "
              f"{'Peak':>8} {'Count':>6}")
        print(f"  {'─'*22} {'─'*11} {'─'*8} {'─'*6}")
        for _, row in sliding.tail(4).iterrows():
            print(f"  {str(row['window_start']):<22} "
                  f"{row['sliding_avg_kw']:>11.3f} "
                  f"{row['sliding_peak_kw']:>8.3f} "
                  f"{row['count']:>6,}")
 
    if submetering is not None and len(submetering) > 0:
        row = submetering.iloc[0]
        total = max(row["total_kwh"], 0.001)
        print(f"\n  REAL-TIME SUB-METERING  (total: {row['total_kwh']:.3f} kWh, "
              f"{row['records']:,} records):")
        for label, col in [("Kitchen","kitchen_kwh"),("Laundry","laundry_kwh"),
                            ("HVAC","hvac_kwh"),("Other","other_kwh")]:
            pct = row[col] / total * 100
            bar = "█" * int(pct / 5)
            print(f"    {label:<10} {row[col]:>8.3f} kWh  {pct:>5.1f}%  {bar}")
    
# Main streaming loop
def run_streaming_pipeline(consumer, s3):
    print(f"  Trigger    : every {TRIGGER_INTERVAL_SEC}s")
    print(f"  Persistence: MinIO s3://{BUCKET}/streaming/")
    print(f"  Timeout    : {STREAM_TIMEOUT_SEC}s | Idle stop: {IDLE_STOP_SEC}s")
    print(f"\n  Waiting for messages from producer ...\n")
 
    buffer         = EventBuffer(watermark_minutes=WATERMARK_MIN)
    t_start        = time.time()
    t_trigger      = time.time()
    t_last_message = time.time()
    trigger_n      = 0
 
    try:
        while True:
            elapsed = time.time() - t_start
 
            if STREAM_TIMEOUT_SEC and elapsed >= STREAM_TIMEOUT_SEC:
                print(f"\n[STOP] Timeout ({STREAM_TIMEOUT_SEC}s).")
                break
 
            if buffer.total_received > 0:
                if time.time() - t_last_message >= IDLE_STOP_SEC:
                    print(f"\n[STOP] No messages for {IDLE_STOP_SEC}s — "
                          f"producer finished.")
                    break
 
            # Poll Kafka
            try:
                for message in consumer:
                    record = message.value
                    if isinstance(record, dict):
                        buffer.add(record)
                        t_last_message = time.time()
                    if time.time() - t_trigger >= TRIGGER_INTERVAL_SEC:
                        break
            except StopIteration:
                pass
 
            # Trigger analytics + MinIO write
            if time.time() - t_trigger >= TRIGGER_INTERVAL_SEC:
                trigger_n += 1
                records    = buffer.get_records()
 
                if records:
                    tumbling    = compute_tumbling_windows(records, TUMBLING_WINDOW_MIN)
                    sliding     = compute_sliding_windows(records, SLIDING_WINDOW_MIN,
                                                          SLIDING_STEP_MIN)
                    submetering = compute_submetering(records)
 
                    # Display to console
                    display_results(tumbling, sliding, submetering,
                                    trigger_n, buffer, elapsed)
 
                    # Persist to MinIO — Speed Layer storage
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    write_to_minio(
                        s3,
                        f"streaming/tumbling_windows/trigger_{trigger_n:04d}_{ts}.parquet",
                        tumbling, "tumbling windows"
                    )
                    write_to_minio(
                        s3,
                        f"streaming/sliding_windows/trigger_{trigger_n:04d}_{ts}.parquet",
                        sliding, "sliding windows"
                    )
                    write_to_minio(
                        s3,
                        f"streaming/submetering/trigger_{trigger_n:04d}_{ts}.parquet",
                        submetering, "sub-metering"
                    )
                else:
                    print(f"\n  [TRIGGER {trigger_n}] No records yet — "
                          f"waiting for producer ...")
 
                t_trigger = time.time()
 
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.")
 
    finally:
        consumer.close()
        elapsed = time.time() - t_start
 
        # Write session summary to MinIO
        records = buffer.get_records()
        summary = {
            "session_end"      : time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_received"   : buffer.total_received,
            "total_triggers"   : trigger_n,
            "elapsed_sec"      : round(elapsed, 1),
            "avg_latency_ms"   : round(buffer.avg_latency_ms, 2),
            "throughput_rps"   : round(buffer.total_received / max(elapsed, 0.001), 2),
            "total_kwh"        : round(
                sum(float(r.get("Global_active_power") or 0)
                    for r in records) / 60.0, 4
            ),
            "unique_days"      : len(set(
                str(r.get("Datetime",""))[:10] for r in records
            )),
        }
        write_json_to_minio(
            s3,
            f"streaming/session_summary/session_{time.strftime('%Y%m%d_%H%M%S')}.json",
            summary
        )
 
        return buffer, elapsed, summary

# Main execution
def main():
    s3       = create_minio_client()
    if s3:
        ensure_streaming_prefix(s3)
 
    consumer = create_consumer()
    buffer, elapsed, summary = run_streaming_pipeline(consumer, s3)
 
 
if __name__ == "__main__":
    main()

