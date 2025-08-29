FROM pytorch/pytorch:2.4.0-cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefer-binary -r requirements.txt
COPY . .
CMD ["python", "train.py"]
