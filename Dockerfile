# ============================================================
#  Image d'inference — API FastAPI du modele fine-tune
# ============================================================
#  Build multi-etapes : les outils de compilation restent dans l'etage
#  builder et n'alourdissent pas l'image finale.
#
#  Les POIDS NE SONT PAS EMBARQUES dans l'image (6 Go pour le modele fusionne,
#  120 Mo pour l'adaptateur). Une image doit rester legere, versionnee avec le
#  CODE, et reconstruite a chaque changement de code — pas a chaque
#  reentrainement. Les poids sont montes en volume ou telecharges au demarrage.
#
#  Build :
#    docker build -t qwen-pyds-api .
#
#  Run (adaptateur monte depuis l'hote, GPU) :
#    docker run --gpus all -p 8000:8000 \
#      -v "$(pwd)/outputs/qwen2.5-3b-pyds-lora:/models/adapter:ro" \
#      -e ADAPTER_PATH=/models/adapter \
#      -e HF_HOME=/models/hf-cache \
#      -v "$(pwd)/.hf-cache:/models/hf-cache" \
#      qwen-pyds-api
# ------------------------------------------------------------

# ---------- Etage 1 : dependances ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch depuis l'index CUDA, puis le reste des dependances.
# Les couches sont separees pour maximiser le cache Docker : modifier
# requirements.txt ne force pas la reinstallation de torch (~2,5 Go).
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Etage 2 : image finale ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/models/hf-cache \
    ADAPTER_PATH=/models/adapter

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Uniquement le code necessaire au service d'inference
COPY src/ ./src/
COPY api_server.py inference.py train.py ./

# Utilisateur non privilegie : ne jamais servir en root
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /models && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

# Le healthcheck interroge le vrai endpoint : un conteneur dont le modele n'a
# pas pu se charger doit etre signale comme non sain, pas seulement "demarre".
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
