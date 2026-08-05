FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The corpus index ships with the image — no vector database to reach at boot,
# so a cold start is a process start.
COPY app/ app/
COPY corpus/chunks.jsonl corpus/embeddings.npy corpus/MANIFEST.json corpus/
COPY policy.yaml .

ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
