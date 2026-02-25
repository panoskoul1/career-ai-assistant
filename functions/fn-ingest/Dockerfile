FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image layer so startup is instant.
# The model is ~130 MB and is downloaded once during build.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .

EXPOSE 8080

# Starts the minimal Nuclio-compatible HTTP runner (no FastAPI, no uvicorn)
CMD ["python", "nuclio_runner.py"]
