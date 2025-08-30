import time, sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = "arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerExecutionRole>"  # <-- set me

est = PyTorch(
    entry_point="train.py",
    source_dir=".",                 # assumes you're running from training/ dir; else set to "training"
    role=role,
    framework_version="2.4",        # pick a supported version close to your local
    py_version="py311",
    instance_type="ml.g5.xlarge",   # or "ml.m5.2xlarge" for CPU
    instance_count=1,
    volume_size=100,
    hyperparameters={
        "n_trials": 8,
        "epochs": 8,
        "batch_sizes": "16,32,64",
    },
    environment={
        "AWS_REGION": "us-east-2",
        "S3_BUCKET": "hamdi-gemstone-dataset",
        "S3_DATA_PREFIX": "dataset/train",
        "S3_MODEL_PREFIX": "models",
        "MLFLOW_TRACKING_URI": "http://ec2-3-136-189-20.us-east-2.compute.amazonaws.com/mlflow",
        "MLFLOW_EXPERIMENT_NAME": "gemstone-classifier",

        # DataLoader tuning
        "NUM_WORKERS": "4",
        "PREFETCH_FACTOR": "2",
        "PIN_MEMORY": "1",
    },
    enable_sagemaker_metrics=True,
    use_spot_instances=True,
    max_run=3*60*60,
    max_wait=4*60*60,
)

job_name = f"gemstone-train-{int(time.time())}"
est.fit(job_name=job_name, wait=True)
print("Model tar:", est.model_data)
