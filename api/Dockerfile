FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip uninstall -y kafka || true && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.0 torchvision==0.19.0
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
