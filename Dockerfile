FROM python:3.11-slim

# PySpark a besoin de Java (pas de winutils sous Linux)
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir pyspark==3.5.2 numpy fastapi uvicorn

COPY src/api.py src/api.py
COPY data/about_file.md data/about_file.md
COPY models models

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
