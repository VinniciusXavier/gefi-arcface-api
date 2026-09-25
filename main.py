import os
import sys
import io
import time
import base64
import numpy as np
import cv2

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import insightface
from insightface.app import FaceAnalysis

app = FastAPI(title="FOTOS GEFI — ArcFace Biometric API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Carrega o modelo ArcFace InsightFace (buffalo_l: ResNet-50 512-d)
print("🧠 Inicializando ArcFace (buffalo_l)...")
models_root = os.path.abspath(os.getenv("MODELS_PATH", "models"))
face_analyzer = FaceAnalysis(name="buffalo_l", root=models_root)
face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ ArcFace carregado com sucesso e pronto para inferência!")

class ExtractRequest(BaseModel):
    image: str  # Base64 data URL ou base64 raw

def decode_image(image_input: str) -> np.ndarray:
    try:
        clean_b64 = image_input
        if "," in clean_b64:
            clean_b64 = clean_b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(clean_b64)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Falha ao decodificar imagem JPEG/PNG.")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Imagem inválida: {str(e)}")

@app.get("/")
def health():
    return {
        "status": "online",
        "service": "FOTOS GEFI ArcFace API",
        "model": "InsightFace buffalo_l (ResNet-50)",
        "vector_dim": 512,
        "time": time.time()
    }

@app.post("/extract-face")
def extract_face(req: ExtractRequest):
    t0 = time.time()
    img = decode_image(req.image)

    faces = face_analyzer.get(img)
    if not faces:
        return {
            "success": True,
            "detected": False,
            "message": "Nenhum rosto detectado na imagem.",
            "faces": []
        }

    # Seleciona as faces encontradas ordenadas por área/confiança
    result_faces = []
    for f in faces:
        bbox = f.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        width = int(x2 - x1)
        height = int(y2 - y1)
        confidence = float(f.det_score)
        embedding = [float(x) for x in f.normed_embedding]

        result_faces.append({
            "confidence": round(confidence, 4),
            "box": {
                "x": int(x1),
                "y": int(y1),
                "width": width,
                "height": height
            },
            "embedding": embedding
        })

    # Ordena pelo maior rosto (maior área)
    result_faces.sort(key=lambda x: x["box"]["width"] * x["box"]["height"], reverse=True)

    dt_ms = round((time.time() - t0) * 1000, 2)
    return {
        "success": True,
        "detected": True,
        "process_ms": dt_ms,
        "faces_count": len(result_faces),
        "primary_face": result_faces[0],
        "all_faces": result_faces
    }

@app.post("/extract-face-upload")
async def extract_face_upload(file: UploadFile = File(...)):
    t0 = time.time()
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Arquivo de imagem inválido.")

    faces = face_analyzer.get(img)
    if not faces:
        return {
            "success": True,
            "detected": False,
            "message": "Nenhum rosto detectado.",
            "faces": []
        }

    best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    bbox = best_face.bbox.astype(int)
    dt_ms = round((time.time() - t0) * 1000, 2)

    return {
        "success": True,
        "detected": True,
        "process_ms": dt_ms,
        "primary_face": {
            "confidence": round(float(best_face.det_score), 4),
            "box": {
                "x": int(bbox[0]),
                "y": int(bbox[1]),
                "width": int(bbox[2] - bbox[0]),
                "height": int(bbox[3] - bbox[1])
            },
            "embedding": [float(x) for x in best_face.normed_embedding]
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
