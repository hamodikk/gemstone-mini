import os, time, json, argparse, shutil
from pathlib import Path
import boto3, mlflow, optuna
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy("file_system")
except Exception:
    pass

# Config from env
AWS_REGION   = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
BUCKET       = os.getenv("S3_BUCKET")
DATA_PREFIX  = os.getenv("S3_DATA_PREFIX", "dataset/train")
MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "models")
EXP_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "gemstone-classifier")
IMG_SIZE     = int(os.getenv("IMAGE_SIZE", "224"))

# DataLoader tuning
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
PREFETCH_FACTOR = int(os.getenv("PREFETCH_FACTOR", "2"))
PIN_MEMORY = os.getenv("PIN_MEMORY", "1") == "1"

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/model")

# Dataset sync
def s3_client():
    return boto3.client("s3", region_name=AWS_REGION) if AWS_REGION else boto3.client("s3")

def sync_s3_dataset(local_dir="/data/train"):
    s3 = s3_client()
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=DATA_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.lower().endswith((".jpg",".jpeg",".png")): continue
            rel = key[len(DATA_PREFIX):].lstrip("/")
            out = Path(local_dir) / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, key, str(out))
    return local_dir

# Transforms
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Data loaders
def get_loaders_and_classes(batch_size=32):
    root = sync_s3_dataset()
    full = datasets.ImageFolder(root, transform=train_tfms)
    classes = full.classes
    n = len(full); n_val = max(1, int(0.2*n))
    train_ds, val_ds = random_split(full, [n-n_val, n_val])
    val_ds.dataset.transform = val_tfms
    
    loader_kwargs = dict(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    if NUM_WORKERS > 0:
        loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=PREFETCH_FACTOR))

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    return train_loader, val_loader, classes

# Train objective
def build_model(n_classes: int, device: torch.device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.to(device)
    return model

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        
@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return correct / max(1, total)

def objective_factory(EPOCHS, BATCH_SPACE):
    def objective(trial):
        bs = trial.suggest_categorical("batch_size", BATCH_SPACE)
        train_loader, val_loader, classes = get_loaders_and_classes(batch_size=bs)
        n_classes = len(classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        with mlflow.start_run(nested=True, run_name=f"trial-{trial.number:03d}"):
            model = build_model(n_classes, device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            loss_fn = nn.CrossEntropyLoss()
            
            mlflow.log_params({"lr": lr, "weight_decay": wd, "batch_size": bs, "num_classes": n_classes})
            mlflow.log_text("\n".join(classes), artifact_file="classes.txt")

            best = 0.0
            for epoch in range(EPOCHS):
                train_one_epoch(model, train_loader, opt, loss_fn, device)
                val_acc = eval_acc(model, val_loader, device)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                best = max(best, val_acc)
            
            Path(SM_MODEL_DIR).mkdir(parents=True, exist_ok=True)
            trial_model = f"{SM_MODEL_DIR}/trial-{trial.number}-model.pt"
            trial_classes = f"{SM_MODEL_DIR}/trial-{trial.number}-classes.json"
            torch.save(model, trial_model)
            with open(trial_classes, "w", encoding="utf-8") as f:
                json.dump({"classes": classes}, f)
            
            try:
                mlflow.log_artifact(trial_model, artifact_path="model")
                mlflow.log_artifact(trial_classes, artifact_path="model")
            except Exception as e:
                print(f"[WARN] mlflow.log_artifacts failed: {e}")
                
            return best
    return objective

# Upload Helper
def upload_artifacts_to_s3(model_path=None, classes_path=None):
    model_path = model_path or f"{SM_MODEL_DIR}/model.pt"
    classes_path = classes_path or f"{SM_MODEL_DIR}/classes.json"
    
    s3 = s3_client()
    ts = int(time.time())
    model_key = f"{MODEL_PREFIX}/model-{ts}.pt"
    classes_key = f"{MODEL_PREFIX}/classes-{ts}.json"
    
    s3.upload_file(model_path, BUCKET, model_key)
    s3.upload_file(classes_path, BUCKET, classes_key)
    
    s3.copy_object(Bucket=BUCKET, CopySource={"Bucket": BUCKET, "Key": model_key},
                   Key=f"{MODEL_PREFIX}/latest/model.pt")
    s3.copy_object(Bucket=BUCKET, CopySource={"Bucket": BUCKET, "Key": classes_key},
                   Key=f"{MODEL_PREFIX}/latest/classes.json")
    print(f"Uploaded:\n - s3://{BUCKET}/{model_key}\n - s3://{BUCKET}/{classes_key}\nUpdated latest/ pointers ✓")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=int(os.getenv("N_TRIALS", "6")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "8")))
    parser.add_argument("--batch_sizes", type=str, default=os.getenv("BATCH_SIZES", "16,32,64"))
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    BATCH_SPACE = [int(x) for x in args.batch_sizes.split(",")]
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000"))
    mlflow.set_experiment(EXP_NAME)
    
    study = optuna.create_study(direction="maximize")
    with mlflow.start_run(run_name="optuna-study"):
        objective = objective_factory(EPOCHS, BATCH_SPACE)
        study.optimize(objective, n_trials=args.n_trials)
        print("Best:", study.best_value, study.best_trial.params)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})
        mlflow.log_metric("best_val_acc", study.best_value)
        
        best_n = study.best_trial.number
        src_model = f"{SM_MODEL_DIR}/trial-{best_n}-model.pt"
        src_classes = f"{SM_MODEL_DIR}/trial-{best_n}-classes.json"
        dst_model = f"{SM_MODEL_DIR}/model.pt"
        dst_classes = f"{SM_MODEL_DIR}/classes.json"
        
        if not (Path(src_model).exists() and Path(src_classes).exists()):
            raise FileNotFoundError(f"Best trial artifacts not found: {src_model} / {src_classes}")

        Path(SM_MODEL_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_model, dst_model)
        shutil.copyfile(src_classes, dst_classes)

        # Optionally log final artifacts for this parent run
        try:
            mlflow.log_artifact(dst_model, artifact_path="final_model")
            mlflow.log_artifact(dst_classes, artifact_path="final_model")
        except Exception as e:
            print(f"[WARN] mlflow.log_artifact (final) failed: {e}")
    upload_artifacts_to_s3()
