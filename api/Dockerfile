# PyTorch base already includes Python + torch/torchvision (CPU build)
FROM pytorch/pytorch:2.4.0-cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .

# Optional safety: remove legacy 'kafka' if it ever sneaks in via deps
RUN pip install --upgrade pip && \
    pip uninstall -y kafka || true && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .
EXPOSE 8000
# Use uvicorn directly (same as your old image)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
