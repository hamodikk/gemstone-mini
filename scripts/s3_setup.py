import os
import argparse
from pathlib import Path
import boto3

parser = argparse.ArgumentParser()
parser.add_argument("--bucket", required=True)
parser.add_argument("--src", required=True, help="Local dataset root with class folders")
parser.add_argument("--prefix", default="dataset/train")
args = parser.parse_args()

s3 = boto3.client("s3")
root = Path(args.src)

for p in root.rglob("*"):
    if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png"):
        rel = p.relative_to(root).as_posix()
        key = f"{args.prefix}/{rel}"
        s3.upload_file(str(p), args.bucket, key)
        print(f"Uploaded s3://{args.bucket}/{key}")