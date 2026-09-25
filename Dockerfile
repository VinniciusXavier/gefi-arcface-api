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

# Baixa previamente o modelo buffalo_l e remove os 260MB de modelos desnecessários (3D, 2D e genderage)
RUN python3 -c "import onnxruntime as ort; so = ort.SessionOptions(); so.enable_cpu_mem_arena = False; so.intra_op_num_threads = 1; import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l', root='/app/models', allowed_modules=['detection', 'recognition'], sess_options=so); app.prepare(ctx_id=-1, det_size=(640,640))" \
    && rm -f /app/models/models/buffalo_l/1k3d68.onnx \
    && rm -f /app/models/models/buffalo_l/2d106det.onnx \
    && rm -f /app/models/models/buffalo_l/genderage.onnx

COPY main.py .

ENV PORT=8080
ENV MODELS_PATH=/app/models
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
