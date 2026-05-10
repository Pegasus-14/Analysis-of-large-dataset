import os
import sys

try:
    import boto3
    from botocore.client import Config
    from botocore.exceptions import ClientError
except ImportError:
    print("[ERROR] boto3 not installed. Run: pip install boto3")
    sys.exit(1)
 
# Configuring
LOCAL_OUTPUT = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output"
 
MINIO_ENDPOINT   = "http://localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"
BUCKET           = "energy-lake"
 
SEP = "=" * 60
 
 
# Connecting to MinIO
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url          = MINIO_ENDPOINT,
        aws_access_key_id     = MINIO_ACCESS_KEY,
        aws_secret_access_key = MINIO_SECRET_KEY,
        config                = Config(signature_version="s3v4"),
        region_name           = "us-east-1",
    )
 
 
def ensure_bucket(s3, bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
        print(f"  Bucket '{bucket}' already exists ✓")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            s3.create_bucket(Bucket=bucket)
            print(f"  Bucket '{bucket}' created ✓")
        else:
            raise
 
 
# Uploading 
def upload_directory(s3, local_dir: str, s3_prefix: str, bucket: str):
    if not os.path.isdir(local_dir):
        print(f"  [SKIP] Not found locally: {local_dir}")
        return 0
 
    count       = 0
    total_bytes = 0
 
    for root, dirs, files in os.walk(local_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
 
        for fname in files:
            if fname.startswith(".") or fname == "_SUCCESS":
                continue
 
            local_path = os.path.join(root, fname)
            rel_path   = os.path.relpath(local_path, local_dir)
            s3_key     = s3_prefix + "/" + rel_path.replace("\\", "/")
 
            fsize = os.path.getsize(local_path)
            total_bytes += fsize
 
            s3.upload_file(local_path, bucket, s3_key)
            count += 1
            print(f"  ↑ {s3_key}  ({fsize/1024:.1f} KB)")
 
    print(f"  Uploaded {count} files  ({total_bytes/1024/1024:.2f} MB total)")
    return count
 
 
# Main execution
def main():
    print(SEP)
    print("  MinIO Upload — Milestone 1 Parquet Files")
    print(SEP)
 
    # Check output folder has actual files before even connecting
    total_local = sum(
        len(files)
        for _, _, files in os.walk(LOCAL_OUTPUT)
        if files
    )
    if total_local == 0:
        print(f"""
[ERROR] No files found in: {LOCAL_OUTPUT}
 
The ingestion script must be run successfully before uploading.
Run first:
  python scripts/milestone1/02_data_ingestion_local.py
 
Then re-run this script.
""")
        sys.exit(1)
 
    print(f"\n  Found files locally ({total_local} total) ✓")
    print(f"\nConnecting to MinIO at {MINIO_ENDPOINT} ...")
 
    try:
        s3 = get_s3_client()
        s3.list_buckets()   # connection test
        print("  Connection ✓")
    except Exception as e:
        print(f"""
[ERROR] Cannot connect to MinIO: {e}
 
Check:
  1. Docker Desktop is running
  2. energy_storage container is running (green in Docker Desktop)
  3. Port 9000 is not blocked by a firewall
 
If the container is stopped:
  docker compose up -d storage
""")
        sys.exit(1)
 
    ensure_bucket(s3, BUCKET)
 
    datasets = [
        (os.path.join(LOCAL_OUTPUT, "raw",       "power_consumption"), "raw/power_consumption"),
        (os.path.join(LOCAL_OUTPUT, "raw",       "weather"),           "raw/weather"),
        (os.path.join(LOCAL_OUTPUT, "processed", "power_cleaned"),     "processed/power_cleaned"),
    ]
 
    total_files = 0
    for local_dir, s3_prefix in datasets:
        print(f"\n  Uploading: {s3_prefix}/")
        n = upload_directory(s3, local_dir, s3_prefix, BUCKET)
        total_files += n
 
    print(f"""
{SEP}
  UPLOAD COMPLETE — {total_files} files in s3://{BUCKET}/
 
  Verify in MinIO console:
    http://localhost:9001   (admin / password123)
    Browse: {BUCKET} → raw/ and processed/
{SEP}
""")
 
 
if __name__ == "__main__":
    main()