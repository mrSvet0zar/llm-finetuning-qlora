"""
Serveur d'inference FastAPI pour le modele fine-tune.

Le modele est charge UNE SEULE FOIS au demarrage (evenement startup), puis
expose via un endpoint /generate. Par defaut, charge base 4-bit + adaptateur
LoRA ; on peut pointer un modele fusionne via la variable d'env MERGED_MODEL.

Lancement :
    uvicorn api_server:app --host 0.0.0.0 --port 8000
    # ou : python api_server.py

Test :
    curl -X POST http://localhost:8000/generate \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Qu-est-ce que LoRA ?"}'
Docs interactives : http://localhost:8000/docs
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Config  # noqa: E402

# Etat global du modele (charge au startup)
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from inference import load_model  # import tardif (charge torch)
    cfg = Config()
    merged = os.environ.get("MERGED_MODEL")
    adapter = os.environ.get("ADAPTER_PATH", cfg.train.output_dir)

    print("Chargement du modele...")
    model, tokenizer = load_model(cfg, adapter, merged)
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["cfg"] = cfg
    print("Modele pret.")
    yield
    _state.clear()


app = FastAPI(
    title="Qwen2.5-3B Python/DS/ML — Fine-tune API",
    description="API d'inference d'un LLM fine-tune (QLoRA) expert Python/Data Science/ML.",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Question a poser au modele")
    max_new_tokens: int = Field(320, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    prompt: str
    response: str
    latency_ms: float
    tokens_generated: int


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(req: GenerateRequest):
    from inference import generate
    model, tokenizer = _state["model"], _state["tokenizer"]

    t0 = time.perf_counter()
    text = generate(
        model, tokenizer, req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    latency = (time.perf_counter() - t0) * 1000
    n_tokens = len(tokenizer(text)["input_ids"])

    return GenerateResponse(
        prompt=req.prompt,
        response=text,
        latency_ms=round(latency, 1),
        tokens_generated=n_tokens,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
