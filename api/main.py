import os, io, json
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import boto3
from botocore.exceptions import ClientError
try:
    # Preferred: kafka-python-ng import path
    from kafka3 import KafkaProducer, KafkaConsumer
except ImportError:
    # Fallback if someone runs the code with old kafka-python
    from kafka import KafkaProducer, KafkaConsumer

# Environment
MODEL_URI = os.getenv("MODEL_URI", "model.pt")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
PRED_TOPIC = os.getenv("PREDICTIONS_TOPIC", "predictions")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_DATA_PREFIX = os.getenv("S3_DATA_PREFIX", "dataset/train")

# App
app = FastAPI(title="Gemstone Classification API")
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
PRED_COUNTER = Counter("predictions_total", "Total predictions served")

# Transforms
_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# S3 helpers
def parse_s3_uri(uri: str) -> Tuple[str, str]:
    assert uri.startswith("s3://")
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key

def s3_download(bucket: str, key: str, dest: str):
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    s3.download_file(bucket, key, dest)

def s3_try_load_json(bucket: str, key: str) -> Optional[dict]:
    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise

def list_class_folders_from_s3(bucket: str, prefix: str) -> list:
    s3 = boto3.client("s3")
    classes = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix if prefix.endswith("/") else prefix + "/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            parts = cp["Prefix"].rstrip("/").split("/")
            if parts:
                classes.append(parts[-1])
    return sorted(classes)

# Model + classes cache
_model_cache = None
_class_names: Optional[List[str]] = None

def resolve_classes() -> List[str]:
    global _class_names
    if _class_names is not None:
        return _class_names
    classes = None
    if MODEL_URI.startswith("s3://"):
        bucket, key = parse_s3_uri(MODEL_URI)
        classes_key = "/".join(key.split("/")[:-1]) + "/classes.json"
        meta = s3_try_load_json(bucket, classes_key)
        if meta and isinstance(meta, dict) and "classes" in meta:
            classes = meta["classes"]
    if classes is None and S3_BUCKET and S3_DATA_PREFIX:
        classes = list_class_folders_from_s3(S3_BUCKET, S3_DATA_PREFIX)
    if not classes:
        raise RuntimeError("Unable to resolve class names from S3 (no classes.json and no dataset folders).")
    _class_names = list(classes)
    return _class_names

def load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    local_path = "/app/model/model.pt"
    if MODEL_URI.startswith("s3://"):
        bucket, key = parse_s3_uri(MODEL_URI)
        s3_download(bucket, key, local_path)
        path = local_path
    else:
        path = MODEL_URI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # FIX: cuda
    model = torch.load(path, map_location=device)
    model.eval()
    _model_cache = (model, device)
    return _model_cache

# Kafka producer
_producer = None
def get_producer():
    global _producer
    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),  # FIX: dumps
            linger_ms=50,
        )
    return _producer

def publish_event(payload: dict):
    try:
        prod = get_producer()
        prod.send(PRED_TOPIC, payload)
    except Exception as e:
        print(f"[WARN] Kafka send failed: {e}")

# Routes
@app.get("/health")
def health():
    try:
        cls = resolve_classes()
        return {"status": "ok", "num_classes": len(cls)}
    except Exception:
        return {"status": "ok", "num_classes": None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        classes = resolve_classes()
        model, device = load_model()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x = _transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        klass = classes[idx] if idx < len(classes) else str(idx)
        prob = float(probs[idx])
        PRED_COUNTER.inc()
        publish_event({"filename": file.filename, "class": klass, "prob": prob})
        return {"klass": klass, "prob": prob, "probs": [float(p) for p in probs]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
