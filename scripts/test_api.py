"""Test de fumee de l'API FastAPI (via TestClient : exerce lifespan + endpoints)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starlette.testclient import TestClient

from api_server import app

with TestClient(app) as client:      # le 'with' declenche le startup (chargement modele)
    h = client.get("/health").json()
    print("GET /health ->", h)
    assert h["status"] == "ok" and h["model_loaded"]

    r = client.post("/generate", json={
        "prompt": "Qu'est-ce que le broadcasting en NumPy ?",
        "max_new_tokens": 200,
        "temperature": 0.0,
    }).json()
    print("\nPOST /generate ->")
    print("  latence :", r["latency_ms"], "ms")
    print("  tokens  :", r["tokens_generated"])
    print("  reponse :", r["response"][:400])
    assert len(r["response"]) > 0

print("\nAPI SMOKE TEST OK")
