import os, time, json
from pathlib import Path
import boto3, mlflow, optuna
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

BUCKET       = os.getenv("S3_BUCKET")
DATA_PREFIX  = os.getenv("S3_DATA_PREFIX", "dataset/train")
MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "models")
EXP_NAME     = os.getenv("MLFLOW_EXPERIMENT_NAME", "gemstone-classifier")
IMG_SIZE     = int(os.getenv("IMAGE_SIZE", "224"))

def sync_s3_dataset(local_dir="/data/train"):
    s3 = boto3.client("s3")
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

def get_loaders_and_classes():
    root = sync_s3_dataset()
    full = datasets.ImageFolder(root, transform=train_tfms)
    classes = full.classes
    n = len(full); n_val = max(1, int(0.2*n))
    train_ds, val_ds = random_split(full, [n-n_val, n_val])
    val_ds.dataset.transform = val_tfms
    return (DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2),
            DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2),
            classes)

def objective(trial):
    train_loader, val_loader, classes = get_loaders_and_classes()
    n_classes = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [16,32,64])
    train_loader.batch_size = bs; val_loader.batch_size = bs

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    mlflow.log_params({"lr": lr, "weight_decay": wd, "batch_size": bs, "num_classes": n_classes})
    mlflow.log_text("\n".join(classes), artifact_file="classes.txt")

    best = 0.0
    for epoch in range(8):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out,y)
            loss.backward()      # important
            opt.step()           # important
        # val
        model.eval()
        correct=0; total=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                total += y.size(0); correct += (pred==y).sum().item()
        val_acc = correct/max(1,total)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        best = max(best, val_acc)

    Path("/model").mkdir(exist_ok=True)
    torch.save(model, "/model/model.pt")
    with open("/model/classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f)
    return best

def upload_artifacts_to_s3(model_path="/model/model.pt", classes_path="/model/classes.json"):
    s3 = boto3.client("s3")
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
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000"))
    mlflow.set_experiment(EXP_NAME)
    study = optuna.create_study(direction="maximize")
    with mlflow.start_run():
        study.optimize(objective, n_trials=6)
        print("Best:", study.best_value, study.best_trial.params)
    upload_artifacts_to_s3()
