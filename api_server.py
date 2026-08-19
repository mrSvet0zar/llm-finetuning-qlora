"""
Serveur d'inference FastAPI pour le modele fine-tune.

Points de conception
--------------------
1. LA GENERATION NE BLOQUE PLUS LA BOUCLE D'EVENEMENTS.
   La version initiale declarait `async def generate_endpoint(...)` puis
   appelait `generate(...)`, une fonction SYNCHRONE de ~12 s. Une coroutine qui
   effectue un travail bloquant monopolise la boucle : pendant une generation,
   le serveur ne pouvait repondre a AUCUNE autre requete, pas meme /health —
   ce qui, derriere un orchestrateur, fait redemarrer un conteneur pourtant
   parfaitement sain. Le travail bloquant part desormais dans un threadpool
   (`run_in_threadpool`), la boucle reste libre.

2. ACCES GPU SERIALISE. Un seul GPU de 8 Go ne peut pas traiter plusieurs
   generations simultanees sans risque de saturation memoire. Un semaphore
   borne la concurrence (MAX_CONCURRENCY, defaut 1) : les requetes attendent
   leur tour au lieu de provoquer un OOM. La boucle d'evenements, elle, reste
   disponible pour /health, /metrics et le rejet des requetes en trop.

3. STREAMING. `/generate/stream` emet les tokens au fil de l'eau (SSE). Cela
   ne reduit pas la latence TOTALE mais transforme la latence PERCUE : le
   premier token arrive en une fraction de seconde au lieu de ~12 s.

Endpoints
---------
    GET  /health          liveness  : le processus repond
    GET  /ready           readiness : le modele est charge
    GET  /metrics         latences p50/p95/p99, debit, taux d'erreur
    GET  /metrics/prometheus
    POST /generate        generation complete (JSON)
    POST /generate/stream generation en flux (SSE)

Configuration (variables d'environnement)
-----------------------------------------
    ADAPTER_PATH      chemin de l'adaptateur LoRA (defaut : config du projet)
    MERGED_MODEL      chemin d'un modele fusionne (prioritaire)
    API_KEY           si definie, exigee dans l'en-tete X-API-Key
    MAX_CONCURRENCY   generations simultanees (defaut 1)
    RATE_LIMIT_RPS    requetes/seconde par client (defaut 1.0)
    RATE_LIMIT_BURST  rafale toleree (defaut 5)
    REQUEST_TIMEOUT_S delai maximal d'une generation (defaut 120)

Lancement :
    uvicorn api_server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Config  # noqa: E402
from src.logging_utils import setup_logging  # noqa: E402
from src.serving import MetricsCollector, RateLimiter  # noqa: E402

# --- Configuration ---------------------------------------------------------
API_KEY = os.environ.get("API_KEY")
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "1"))
RATE_LIMIT_RPS = float(os.environ.get("RATE_LIMIT_RPS", "1.0"))
RATE_LIMIT_BURST = float(os.environ.get("RATE_LIMIT_BURST", "5"))
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "120"))

MAX_PROMPT_CHARS = 4000

logger = setup_logging(name="api")
metrics = MetricsCollector()
rate_limiter = RateLimiter(rate=RATE_LIMIT_RPS, capacity=RATE_LIMIT_BURST)

_state: dict = {}


# ---------------------------------------------------------------------------
#  Cycle de vie
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    from inference import load_model  # import tardif : charge torch

    cfg = Config()
    merged = os.environ.get("MERGED_MODEL")
    adapter = os.environ.get("ADAPTER_PATH", cfg.train.output_dir)

    logger.info("Chargement du modele...")
    t0 = time.perf_counter()
    # Chargement dans un thread : le serveur repond a /health des le demarrage
    # (en 503 tant que le modele n'est pas pret) au lieu de paraitre fige.
    model, tokenizer = await run_in_threadpool(load_model, cfg, adapter, merged)
    _state.update({
        "model": model,
        "tokenizer": tokenizer,
        "cfg": cfg,
        "semaphore": asyncio.Semaphore(MAX_CONCURRENCY),
        "load_time_s": round(time.perf_counter() - t0, 1),
    })
    logger.info(f"Modele pret en {_state['load_time_s']} s "
                f"(concurrence max : {MAX_CONCURRENCY})")
    yield
    _state.clear()


app = FastAPI(
    title="Qwen2.5-3B Python/DS/ML — API d'inference",
    description="Modele fine-tune par QLoRA, expert Python / Data Science / ML.",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
#  Securite et quotas
# ---------------------------------------------------------------------------
async def authorize(request: Request,
                    x_api_key: str | None = Header(default=None)) -> str:
    """Authentifie le client et applique la limitation de debit.

    Sans API_KEY definie, le service reste ouvert (usage local) mais le debit
    est tout de meme limite : un GPU unique se sature tres vite.
    """
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            metrics.rejected_unauthorized += 1
            raise HTTPException(status_code=401, detail="Cle API invalide ou absente.")
        client_id = x_api_key
    else:
        client_id = request.client.host if request.client else "inconnu"

    allowed, retry_after = rate_limiter.allow(client_id)
    if not allowed:
        metrics.rejected_rate_limit += 1
        raise HTTPException(
            status_code=429,
            detail=f"Trop de requetes. Reessayer dans {retry_after} s.",
            headers={"Retry-After": str(retry_after)},
        )
    return client_id


def require_model() -> tuple:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Modele non encore charge.")
    return _state["model"], _state["tokenizer"]


# ---------------------------------------------------------------------------
#  Schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_CHARS,
                        description="Question posee au modele")
    max_new_tokens: int = Field(320, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)


class GenerateResponse(BaseModel):
    request_id: str
    prompt: str
    response: str
    latency_ms: float
    tokens_generated: int
    tokens_per_s: float


# ---------------------------------------------------------------------------
#  Endpoints d'observabilite
# ---------------------------------------------------------------------------
@app.get("/health", tags=["observabilite"])
async def health():
    """Liveness : le processus repond. Ne dit rien du modele (voir /ready)."""
    return {"status": "ok"}


@app.get("/ready", tags=["observabilite"])
async def ready(response: Response):
    """Readiness : le modele est charge et le service peut traiter du trafic."""
    if "model" not in _state:
        response.status_code = 503
        return {"status": "chargement", "model_loaded": False}
    return {"status": "ok", "model_loaded": True,
            "load_time_s": _state.get("load_time_s")}


@app.get("/metrics", tags=["observabilite"])
async def get_metrics():
    return metrics.snapshot()


@app.get("/metrics/prometheus", response_class=PlainTextResponse,
         tags=["observabilite"])
async def get_metrics_prometheus():
    return metrics.prometheus()


# ---------------------------------------------------------------------------
#  Generation
# ---------------------------------------------------------------------------
def _generate_blocking(prompt: str, max_new_tokens: int,
                       temperature: float, top_p: float) -> tuple[str, int]:
    """Travail synchrone, execute dans un threadpool (jamais dans la boucle)."""
    from inference import generate
    model, tokenizer = _state["model"], _state["tokenizer"]
    text = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p)
    n_tokens = len(tokenizer(text)["input_ids"])
    return text, n_tokens


@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
async def generate_endpoint(req: GenerateRequest,
                            client_id: str = Depends(authorize)):
    require_model()
    request_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    try:
        # Le semaphore borne l'acces au GPU ; la boucle d'evenements reste
        # libre pendant l'attente comme pendant la generation.
        async with _state["semaphore"]:
            text, n_tokens = await asyncio.wait_for(
                run_in_threadpool(_generate_blocking, req.prompt,
                                  req.max_new_tokens, req.temperature, req.top_p),
                timeout=REQUEST_TIMEOUT_S,
            )
    except asyncio.TimeoutError:
        metrics.record_error()
        logger.warning("Generation expiree",
                       extra={"ctx_request_id": request_id,
                              "ctx_timeout_s": REQUEST_TIMEOUT_S})
        raise HTTPException(status_code=504,
                            detail=f"Generation expiree ({REQUEST_TIMEOUT_S} s).") from None
    except Exception as exc:
        metrics.record_error()
        logger.exception("Echec de generation",
                         extra={"ctx_request_id": request_id})
        raise HTTPException(status_code=500, detail="Erreur interne.") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    metrics.record_success(latency_ms, n_tokens)
    logger.info("Generation terminee", extra={
        "ctx_request_id": request_id,
        "ctx_client": client_id,
        "ctx_latency_ms": round(latency_ms, 1),
        "ctx_tokens": n_tokens,
    })

    return GenerateResponse(
        request_id=request_id,
        prompt=req.prompt,
        response=text,
        latency_ms=round(latency_ms, 1),
        tokens_generated=n_tokens,
        tokens_per_s=round(n_tokens / (latency_ms / 1000), 2) if latency_ms else 0.0,
    )


async def _stream_tokens(req: GenerateRequest, request_id: str) -> AsyncIterator[str]:
    """Emet les tokens au fil de l'eau au format SSE.

    `TextIteratorStreamer` de transformers alimente une file depuis un thread
    dedie ou tourne `model.generate`. On draine cette file sans jamais bloquer
    la boucle d'evenements.
    """
    from threading import Thread

    from transformers import TextIteratorStreamer

    from train import SYSTEM_PROMPT

    model, tokenizer = _state["model"], _state["tokenizer"]
    t0 = time.perf_counter()
    n_tokens = 0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.prompt},
    ]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                    skip_special_tokens=True)

    kwargs = {
        "input_ids": enc["input_ids"].to(model.device),
        "attention_mask": enc["attention_mask"].to(model.device),
        "max_new_tokens": req.max_new_tokens,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }
    if req.temperature > 0:
        kwargs.update(do_sample=True, temperature=req.temperature, top_p=req.top_p)
    else:
        kwargs.update(do_sample=False)

    thread = Thread(target=model.generate, kwargs=kwargs, daemon=True)
    thread.start()

    yield f"data: {json.dumps({'request_id': request_id, 'event': 'start'})}\n\n"
    try:
        loop = asyncio.get_running_loop()
        iterator = iter(streamer)
        while True:
            # next() bloque : on l'execute hors de la boucle d'evenements.
            chunk = await loop.run_in_executor(None, lambda: next(iterator, None))
            if chunk is None:
                break
            n_tokens += 1
            yield f"data: {json.dumps({'token': chunk})}\n\n"
    except Exception:
        metrics.record_error()
        logger.exception("Echec du streaming", extra={"ctx_request_id": request_id})
        yield f"data: {json.dumps({'event': 'error'})}\n\n"
        return

    latency_ms = (time.perf_counter() - t0) * 1000
    metrics.record_success(latency_ms, n_tokens)
    yield ("data: " + json.dumps({
        "event": "end",
        "tokens_generated": n_tokens,
        "latency_ms": round(latency_ms, 1),
    }) + "\n\n")
    yield "data: [DONE]\n\n"


@app.post("/generate/stream", tags=["generation"])
async def generate_stream(req: GenerateRequest,
                          client_id: str = Depends(authorize)):
    """Generation en flux (Server-Sent Events).

    Ne reduit pas la latence totale mais fait apparaitre le premier token
    presque immediatement, ce qui change entierement l'experience percue.
    """
    require_model()
    request_id = uuid.uuid4().hex[:12]

    async def guarded() -> AsyncIterator[str]:
        async with _state["semaphore"]:
            async for chunk in _stream_tokens(req, request_id):
                yield chunk

    return StreamingResponse(
        guarded(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
