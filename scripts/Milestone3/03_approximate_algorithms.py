import os
import sys
import time
import json
import math
import hashlib
import struct
import io
from collections import defaultdict


# Configuration
KAFKA_BROKER      = "localhost:9092"
KAFKA_TOPIC       = "energy-readings"
CONSUMER_GROUP    = "energy-algorithms-consumer"   
 
MINIO_ENDPOINT    = "http://localhost:9000"
MINIO_ACCESS_KEY  = "admin"
MINIO_SECRET_KEY  = "password123"
BUCKET            = "energy-lake"
 
BATCH_OUTPUT      = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\batch"
 
# How many Kafka messages to process before reporting
PROCESS_N_RECORDS = 5000    # matches MAX_RECORDS in producer
 
SEP = "=" * 72

# minio client setup
def create_minio_client():
    import boto3
    from botocore.client import Config
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url          = MINIO_ENDPOINT,
            aws_access_key_id     = MINIO_ACCESS_KEY,
            aws_secret_access_key = MINIO_SECRET_KEY,
            config                = Config(signature_version="s3v4"),
            region_name           = "us-east-1",
        )
        s3.list_buckets()
        print(f"  ✓ MinIO connected")
        return s3
    except Exception as e:
        print(f"  [WARN] MinIO not reachable: {e} — results won't be persisted")
        return None
 
 
def write_to_minio(s3, key: str, data):
    """Write dict/list as JSON to MinIO."""
    if s3 is None:
        return
    try:
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        s3.put_object(Bucket=BUCKET, Key=key, Body=body,
                      ContentType="application/json")
        print(f"  → MinIO: s3://{BUCKET}/{key}")
    except Exception as e:
        print(f"  [WARN] MinIO write failed: {e}")


# BLOOM filter
class BloomFilter:
    def __init__(self, capacity: int, false_pos_rate: float = 0.01):
        self.capacity       = capacity
        self.false_pos_rate = false_pos_rate
        self.m = math.ceil(-(capacity * math.log(false_pos_rate)) /
                           (math.log(2) ** 2))
        self.k = max(1, round((self.m / capacity) * math.log(2)))
        self.bit_array  = bytearray(math.ceil(self.m / 8))
        self.n_inserted = 0
 
    def _positions(self, item: str) -> list:
        h1 = int(hashlib.sha256(item.encode()).hexdigest(), 16) % self.m
        h2 = int(hashlib.md5(item.encode()).hexdigest(),    16) % self.m
        return [(h1 + i * h2) % self.m for i in range(self.k)]
 
    def add(self, item: str):
        for pos in self._positions(item):
            self.bit_array[pos // 8] |= (1 << (pos % 8))
        self.n_inserted += 1
 
    def contains(self, item: str) -> bool:
        return all(
            self.bit_array[pos // 8] & (1 << (pos % 8))
            for pos in self._positions(item)
        )
 
    @property
    def est_false_pos_rate(self) -> float:
        return (1 - math.exp(-self.k * self.n_inserted / self.m)) ** self.k
 
    def summary(self) -> dict:
        return {
            "capacity"       : self.capacity,
            "target_fp_rate" : self.false_pos_rate,
            "bit_array_kb"   : round(len(self.bit_array) / 1024, 2),
            "hash_functions" : self.k,
            "n_inserted"     : self.n_inserted,
            "est_fp_rate"    : round(self.est_false_pos_rate, 6),
        }

# Count-Min Sketch
class CountMinSketch:
    def __init__(self, width: int = 2000, depth: int = 7):
        self.width  = width
        self.depth  = depth
        self.table  = [[0] * width for _ in range(depth)]
        self.total  = 0
        self.seeds  = [i * 2654435761 % (2**32) for i in range(1, depth + 1)]
 
    def _hash(self, item: str, seed: int) -> int:
        raw = hashlib.sha256(struct.pack(">I", seed) + item.encode()).digest()
        return int.from_bytes(raw[:4], "big") % self.width
 
    def update(self, item: str, count: int = 1):
        for i, seed in enumerate(self.seeds):
            self.table[i][self._hash(item, seed)] += count
        self.total += count
 
    def query(self, item: str) -> int:
        return min(
            self.table[i][self._hash(item, seed)]
            for i, seed in enumerate(self.seeds)
        )
 
    def summary(self) -> dict:
        return {
            "width"         : self.width,
            "depth"         : self.depth,
            "total_updates" : self.total,
            "memory_kb"     : round(self.width * self.depth * 4 / 1024, 2),
            "error_bound_epsilon": round(math.e / self.width, 6),
            "error_prob_delta"   : round((0.5 ** self.depth), 6),
        }
    
# Kafka consumer + algorithm application
def create_algorithm_consumer():
    from kafka import KafkaConsumer
    from kafka.errors import NoBrokersAvailable
 
    print(f"\n[CONNECT] Kafka topic '{KAFKA_TOPIC}' ...")
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers     = KAFKA_BROKER,
            group_id              = CONSUMER_GROUP,
            auto_offset_reset     = "earliest",
            enable_auto_commit    = True,
            consumer_timeout_ms   = 5000,
            value_deserializer    = lambda v: json.loads(v.decode("utf-8")),
            session_timeout_ms    = 30000,
            heartbeat_interval_ms = 10000,
        )
        print(f"Connected | Group: {CONSUMER_GROUP}")
        return consumer
    except NoBrokersAvailable:
        print("[ERROR] Kafka not reachable. Start: docker compose up -d kafka")
        sys.exit(1)
 
 
def get_consumption_bucket(gap) -> str:
    """Map Global_active_power (kW) to a categorical bucket."""
    if gap is None or (isinstance(gap, float) and math.isnan(gap)):
        return "unknown"
    gap = float(gap)
    if gap < 0.5:  return "0-0.5kW"
    if gap < 1.0:  return "0.5-1kW"
    if gap < 2.0:  return "1-2kW"
    if gap < 3.0:  return "2-3kW"
    if gap < 5.0:  return "3-5kW"
    return "5+kW"
 
 
def apply_algorithms_to_stream(consumer):
    """
    Process PROCESS_N_RECORDS live Kafka messages.
    For each message:
      • Bloom Filter checks if the Datetime key was seen before
      • Count-Min Sketch counts its consumption bucket
 
    This is the correct approach — algorithms run on the live stream,
    not on pre-loaded data.
    """
    print(f"PROCESSING {PROCESS_N_RECORDS:,} LIVE KAFKA MESSAGES")
 
    bf  = BloomFilter(capacity=PROCESS_N_RECORDS, false_pos_rate=0.01)
    cms = CountMinSketch(width=2000, depth=7)
 
    print(f"\n  BloomFilter   : {bf.m:,} bits = {len(bf.bit_array)/1024:.1f} KB | "
          f"{bf.k} hash functions")
    print(f"  CountMinSketch: {cms.width}×{cms.depth} = "
          f"{cms.width*cms.depth:,} counters = "
          f"{cms.width*cms.depth*4/1024:.1f} KB")
    print(f"\n  Processing messages ...\n")
 
    processed        = 0
    duplicates_caught = 0
    bucket_exact     = defaultdict(int)
    latencies        = []
    t_start          = time.time()
    report_every     = max(1, PROCESS_N_RECORDS // 10)
 
    BUCKETS = ["0-0.5kW","0.5-1kW","1-2kW","2-3kW","3-5kW","5+kW"]
 
    try:
        for message in consumer:
            record = message.value
            if not isinstance(record, dict):
                continue
 
            # ── Bloom Filter: duplicate Datetime detection ─────────────────
            key = str(record.get("Datetime", processed))
 
            if bf.contains(key):
                duplicates_caught += 1
            else:
                bf.add(key)
 
            # ── Count-Min Sketch: consumption bucket frequency ─────────────
            gap    = record.get("Global_active_power")
            bucket = get_consumption_bucket(gap)
            cms.update(bucket)
            bucket_exact[bucket] += 1
 
            # ── Latency measurement ────────────────────────────────────────
            now_ms      = int(time.time() * 1000)
            producer_ts = record.get("event_timestamp", now_ms)
            latencies.append(now_ms - producer_ts)
 
            processed += 1
 
            if processed % report_every == 0:
                elapsed = time.time() - t_start
                rate    = processed / max(elapsed, 0.001)
                print(f"  [{processed:>6,}/{PROCESS_N_RECORDS:,}]  "
                      f"rate={rate:.0f} rec/s  |  "
                      f"duplicates={duplicates_caught}  |  "
                      f"bf_fp_est={bf.est_false_pos_rate*100:.4f}%")
 
            if processed >= PROCESS_N_RECORDS:
                break
 
    except StopIteration:
        print(f"  [INFO] Consumer timeout — processed {processed:,} messages "
              f"(producer may have finished)")
 
    elapsed = time.time() - t_start
    consumer.close()
 
    return bf, cms, bucket_exact, processed, duplicates_caught, latencies, elapsed

# Results 
def report_bloom_filter(bf: BloomFilter, processed: int, duplicates: int):
    bf_mem  = len(bf.bit_array) / 1024
    all_mem = processed * 20 / 1024
    saving  = all_mem / max(bf_mem, 0.001)
 
    print(f"""
  Records processed  : {processed:,}
  Duplicates caught  : {duplicates:,}
  Unique keys added  : {bf.n_inserted:,}
  Est. false pos rate: {bf.est_false_pos_rate*100:.4f}%  (target: 1.00%)
 
  Memory comparison:
    Bloom Filter      : {bf_mem:.1f} KB   (bit array only)
    Exact HashSet     : ~{all_mem:.1f} KB  (store all {processed:,} keys)
    Memory saving     : ~{saving:.0f}×""")
    return {
        "processed"           : processed,
        "duplicates_caught"   : duplicates,
        "unique_keys"         : bf.n_inserted,
        "est_fp_rate"         : round(bf.est_false_pos_rate, 6),
        "memory_kb"           : round(bf_mem, 2),
        "memory_saving_factor": round(saving, 1),
        **bf.summary()
    }

def report_count_min_sketch(cms: CountMinSketch, bucket_exact: dict, processed: int):
    BUCKETS = ["0-0.5kW","0.5-1kW","1-2kW","2-3kW","3-5kW","5+kW","unknown"]
 
    print(f"  Total messages: {processed:,}")
    print(f"  CMS counters  : {cms.width}×{cms.depth} = "
          f"{cms.width*cms.depth:,} ({cms.width*cms.depth*4/1024:.1f} KB)")
    print(f"\n  {'Bucket':<12} {'CMS Est':>10} {'Exact':>10} "
          f"{'Error':>8} {'Err%':>7}")
    print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*8} {'─'*7}")
 
    results = []
    total_err = 0
    for b in BUCKETS:
        est   = cms.query(b)
        truth = bucket_exact.get(b, 0)
        err   = est - truth
        pct   = (err / max(truth, 1)) * 100
        total_err += abs(err)
        bar   = "█" * min(40, int(truth / max(processed, 1) * 200))
        print(f"  {b:<12} {est:>10,} {truth:>10,} {err:>+8,} {pct:>+6.2f}%  {bar}")
        results.append({"bucket": b, "cms_estimate": est,
                         "exact": truth, "error": err})
 
    err_rate = total_err / max(processed, 1) * 100
    print(f"""
  Total absolute error : {total_err:,}
  Error rate           : {err_rate:.4f}%""")
    return results

# Batch vs Stream comparison
def batch_vs_streaming_comparison(processed: int, latencies: list, s3):
    print("BATCH vs STREAMING COMPARISON")
 
    import pyarrow.parquet as pq
 
    # ── Batch side: read from M2 daily_totals output ──────────────────────
    batch_path = os.path.join(BATCH_OUTPUT, "daily_totals")
    if os.path.isdir(batch_path):
        batch_pdf       = pq.read_table(batch_path).to_pandas()
        batch_days      = len(batch_pdf)
        batch_total_kwh = float(batch_pdf["total_kwh"].sum())
        batch_avg_kwh   = float(batch_pdf["total_kwh"].mean())
        batch_source    = "Milestone 2 batch pipeline (Spark + pyarrow)"
    else:
        batch_days = batch_total_kwh = batch_avg_kwh = 0
        batch_source = "NOT FOUND — run m2_01_batch_processing.py first"
 
    # ── Streaming side: stats from THIS run ───────────────────────────────
    avg_latency = sum(latencies) / max(len(latencies), 1) if latencies else 0
    stream_source = f"Kafka topic '{KAFKA_TOPIC}' — live messages (this session)"
    return {
        "batch_days"      : batch_days,
        "batch_total_kwh" : round(batch_total_kwh, 4),
        "stream_records"  : processed,
        "stream_latency_ms": round(avg_latency, 2),
    }

# Main execution
def main():
    s3       = create_minio_client()
    consumer = create_algorithm_consumer()
 
    # Run algorithms on live Kafka stream
    (bf, cms, bucket_exact,processed, duplicates,latencies, elapsed) = apply_algorithms_to_stream(consumer)
 
    # Report results
    bf_results  = report_bloom_filter(bf, processed, duplicates)
    cms_results = report_count_min_sketch(cms, bucket_exact, processed)
    comparison  = batch_vs_streaming_comparison(processed, latencies, s3)
 
    # Persist algorithm results to MinIO
    ts = time.strftime("%Y%m%d_%H%M%S")
    write_to_minio(s3, f"streaming/algorithms/bloom_filter_{ts}.json", {
        "algorithm"   : "BloomFilter",
        "run_at"      : ts,
        "results"     : bf_results,
    })
    write_to_minio(s3, f"streaming/algorithms/count_min_sketch_{ts}.json", {
        "algorithm"   : "CountMinSketch",
        "run_at"      : ts,
        "bucket_results": cms_results,
        "cms_summary" : cms.summary(),
    })
    write_to_minio(s3, f"streaming/algorithms/batch_vs_stream_{ts}.json", {
        "comparison"  : comparison,
        "run_at"      : ts,
    })

if __name__ == "__main__":
    main()