FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baixa previamente o modelo buffalo_l (apenas detecção e reconhecimento para caber em < 250MB)
RUN python3 -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l', root='/app/models', allowed_modules=['detection', 'recognition']); app.prepare(ctx_id=-1, det_size=(640,640))"

COPY main.py .

ENV PORT=8080
ENV MODELS_PATH=/app/models

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
