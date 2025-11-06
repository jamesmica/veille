FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY . .

# Chemin du Parquet par défaut (monte un Volume Railway sur /data)
ENV PARQUET_PATH=/data/data.parquet
# Pour CORS, définis ALLOW_ORIGINS (ex: https://tonlogin.github.io)
# ENV ALLOW_ORIGINS=https://tonlogin.github.io

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
