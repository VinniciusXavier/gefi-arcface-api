---
title: Gefi Arcface Api
emoji: ⚜️
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# FOTOS GEFI — ArcFace Biometric API
Microserviço de extração de vetores biométricos faciais de 512 dimensões (InsightFace / ArcFace ResNet-50) para o sistema de fotos do Grupo Escoteiro Franca do Imperador.

## Endpoints:
- `GET /`: Healthcheck
- `POST /extract-face`: Recebe `{ "image": "<base64>" }` e retorna o vetor biométrico normalizado de 512 dimensões.
