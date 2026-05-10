import os
import sys
import json
import time

# Configuration for Kafka producer
INPUT_CLEAN      = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\processed\power_cleaned"
KAFKA_BROKER     = "localhost:9092"
KAFKA_TOPIC      = "energy-readings"
REPLAY_SPEED_MS  = 100      
MAX_RECORDS      = 5000      
 
SEP = "=" * 72

# Kafka connection
def create_producer():
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
 

    print(f"\n[CONNECT] Connecting to Kafka broker at {KAFKA_BROKER} ...")

    try:
        producer = KafkaProducer(
            bootstrap_servers  = KAFKA_BROKER,
            value_serializer   = lambda v: json.dumps(v).encode("utf-8"),
            key_serializer     = lambda k: k.encode("utf-8") if k else None,
            acks               = "all",
            retries            = 3,
            linger_ms          = 10,
            request_timeout_ms = 10000,
        )
        # Testing the connection
        producer.bootstrap_connected()
        print(f"Connected to Kafka broker")
        return producer
    except NoBrokersAvailable:
        print(f"""
[ERROR] Cannot connect to Kafka at {KAFKA_BROKER}""")
        sys.exit(1)

# Topic setup
def ensure_topic_exists():
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError
 
    print(f"\n[TOPIC] Ensuring topic '{KAFKA_TOPIC}' exists ...")
 
    try:
        admin = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
        existing = admin.list_topics()
 
        if KAFKA_TOPIC in existing:
            print(f"  Topic '{KAFKA_TOPIC}' already exists")
        else:
            topic = NewTopic(
                name               = KAFKA_TOPIC,
                num_partitions     = 16,
                replication_factor = 1,
            )
            admin.create_topics([topic])
            print(f"  Topic '{KAFKA_TOPIC}' created successfully")
            print(f"  Partitions: 16  |  Replication: 1")
 
        admin.close()
 
    except TopicAlreadyExistsError:
        print(f"  Topic '{KAFKA_TOPIC}' already exists")
    except Exception as e:
        print(f"  [WARNING] Could not create topic via AdminClient: {e}")
        print(f"  Topic will be auto-created when first message is produced.")

# Loading data
def load_data():
    import pyarrow.parquet as pq
 
    print(f"\n[LOAD] Reading cleaned power data ...")
    table = pq.read_table(INPUT_CLEAN)
    pdf   = table.to_pandas()
 
    # Sort chronologically
    pdf = pdf.sort_values("Datetime").reset_index(drop=True)
 
    # Limit records if MAX_RECORDS is set
    if MAX_RECORDS is not None:
        pdf = pdf.head(MAX_RECORDS)
        print(f"Loaded {len(pdf):,} rows (capped at MAX_RECORDS={MAX_RECORDS})")
    else:
        print(f"Loaded {len(pdf):,} rows (full dataset)")
 
    return pdf

# Poducer messages
def stream_to_kafka(producer, pdf):
    print(f"\n[STREAM] Sending {len(pdf):,} records to topic '{KAFKA_TOPIC}' ...")
    print(f"  Speed    : {REPLAY_SPEED_MS}ms between records")
    print(f"  Rate     : ~{1000/REPLAY_SPEED_MS:.0f} records/sec")
    print(f"  Est. time: ~{len(pdf)*REPLAY_SPEED_MS/1000:.0f}s")
    print(f"\n  Press Ctrl+C to stop early.\n")
 
    sent       = 0
    errors     = 0
    t_start    = time.time()
    report_every = max(1, len(pdf) // 20)   
 
    for _, row in pdf.iterrows():
        try:
            # Build message — convert NaN to None for clean JSON
            record = {}
            for col in pdf.columns:
                val = row[col]
                if hasattr(val, 'item'):
                    val = val.item()           
                if val != val:                
                    val = None
                if hasattr(val, 'isoformat'):  
                    val = str(val)
                record[col] = val
 
            # Adding wall-clock production timestamp for latency measurement
            record["event_timestamp"] = int(time.time() * 1000)
 
            # Key = Datetime string → consistent partition routing
            key = record.get("Datetime", str(sent))
 
            producer.send(KAFKA_TOPIC, key=key, value=record)
            sent += 1
 
            # Progress report
            if sent % report_every == 0:
                elapsed  = time.time() - t_start
                rate     = sent / max(elapsed, 0.001)
                pct      = sent / len(pdf) * 100
                print(f"  [{pct:5.1f}%] Sent {sent:>6,} records  |  "
                      f"{rate:.1f} rec/s  |  {elapsed:.1f}s elapsed")
 
            time.sleep(REPLAY_SPEED_MS / 1000.0)
 
        except KeyboardInterrupt:
            print(f"\n  [STOP] Interrupted by user after {sent:,} records.")
            break
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [WARN] Failed to send record {sent}: {e}")
 
    # Flush remaining buffered messages
    producer.flush()
    elapsed = time.time() - t_start
 
    print(f"""
  ─────────────────────────────────────────────────────────────────
  PRODUCER COMPLETE
    Records sent    : {sent:,}
    Errors          : {errors}
    Elapsed time    : {elapsed:.1f}s
    Throughput      : {sent/max(elapsed,0.001):.1f} records/sec
    Topic           : {KAFKA_TOPIC}
  ─────────────────────────────────────────────────────────────────
    """)
 
    return sent

# Main execution
def main():
    ensure_topic_exists()
    producer = create_producer()
    pdf      = load_data()
    stream_to_kafka(producer, pdf)
    producer.close()

if __name__ == "__main__":
    main()