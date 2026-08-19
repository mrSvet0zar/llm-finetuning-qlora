"""
Test de charge de l'API d'inference.

Mesure ce qui compte reellement en production : les percentiles de latence
(p50/p95/p99) plutot que la moyenne, le debit en tokens/s, et le taux d'erreur
sous concurrence. La moyenne masque exactement ce qui degrade l'experience —
la queue de distribution.

Le serveur doit tourner :
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Usage :
    python scripts/load_test.py                       # 10 requetes, 2 en parallele
    python scripts/load_test.py -n 20 -c 4
    python scripts/load_test.py --stream              # mesure le time-to-first-token
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.serving import percentile  # noqa: E402

QUESTIONS = [
    "Qu'est-ce que la vectorisation en Python ?",
    "Explique LoRA en deux phrases.",
    "Quelle est la difference entre .loc et .iloc dans Pandas ?",
    "Pourquoi separer les donnees en train, validation et test ?",
    "Qu'est-ce que le gradient checkpointing ?",
    "Comment gerer un jeu de donnees desequilibre ?",
]


async def une_requete(client: httpx.AsyncClient, question: str,
                      headers: dict, max_new_tokens: int) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post("/generate", headers=headers, json={
            "prompt": question, "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        })
        latence = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "latency_ms": latence}
        data = r.json()
        return {"ok": True, "status": 200, "latency_ms": latence,
                "tokens": data["tokens_generated"]}
    except Exception as exc:                                   # noqa: BLE001
        return {"ok": False, "status": type(exc).__name__,
                "latency_ms": (time.perf_counter() - t0) * 1000}


async def une_requete_stream(client: httpx.AsyncClient, question: str,
                             headers: dict, max_new_tokens: int) -> dict:
    """Mesure le time-to-first-token : ce que l'utilisateur percoit vraiment."""
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    try:
        async with client.stream("POST", "/generate/stream", headers=headers,
                                 json={"prompt": question,
                                       "max_new_tokens": max_new_tokens,
                                       "temperature": 0.0}) as r:
            if r.status_code != 200:
                return {"ok": False, "status": r.status_code,
                        "latency_ms": (time.perf_counter() - t0) * 1000}
            async for ligne in r.aiter_lines():
                if not ligne.startswith("data: "):
                    continue
                charge = ligne[6:]
                if charge == "[DONE]":
                    break
                evenement = json.loads(charge)
                if "token" in evenement:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    tokens += 1
        return {"ok": True, "status": 200,
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "ttft_ms": ttft, "tokens": tokens}
    except Exception as exc:                                   # noqa: BLE001
        return {"ok": False, "status": type(exc).__name__,
                "latency_ms": (time.perf_counter() - t0) * 1000}


async def executer(url: str, n: int, concurrence: int, stream: bool,
                   max_new_tokens: int, api_key: str | None) -> list[dict]:
    headers = {"X-API-Key": api_key} if api_key else {}
    limite = asyncio.Semaphore(concurrence)
    fn = une_requete_stream if stream else une_requete

    async with httpx.AsyncClient(base_url=url, timeout=300.0) as client:
        # Verifie que le modele est pret avant de mesurer quoi que ce soit
        try:
            pret = await client.get("/ready")
            if pret.status_code != 200:
                raise SystemExit(f"Modele non pret ({pret.status_code}) : {pret.text}")
        except httpx.ConnectError:
            raise SystemExit(f"Serveur injoignable sur {url}.") from None

        async def bornee(i: int) -> dict:
            async with limite:
                return await fn(client, QUESTIONS[i % len(QUESTIONS)],
                                headers, max_new_tokens)

        t0 = time.perf_counter()
        resultats = await asyncio.gather(*(bornee(i) for i in range(n)))
        duree = time.perf_counter() - t0

    for r in resultats:
        r["_duree_totale"] = duree
    return resultats


def rapport(resultats: list[dict], concurrence: int, stream: bool) -> None:
    ok = [r for r in resultats if r["ok"]]
    ko = [r for r in resultats if not r["ok"]]
    duree = resultats[0]["_duree_totale"] if resultats else 0.0
    latences = [r["latency_ms"] for r in ok]
    tokens = sum(r.get("tokens", 0) for r in ok)

    print("\n" + "=" * 62)
    print(f"RESULTATS  ({len(resultats)} requetes, concurrence {concurrence}"
          f"{', streaming' if stream else ''})")
    print("=" * 62)
    print(f"  Reussites        : {len(ok)}/{len(resultats)}")
    if ko:
        codes = {}
        for r in ko:
            codes[r["status"]] = codes.get(r["status"], 0) + 1
        print(f"  Echecs           : {codes}")
    if not ok:
        return

    print(f"  Duree totale     : {duree:.1f} s")
    print(f"  Debit            : {len(ok) / duree:.2f} req/s | "
          f"{tokens / duree:.1f} tokens/s")
    print("\n  Latence (ms)")
    print(f"    p50            : {percentile(latences, 0.50):.0f}")
    print(f"    p95            : {percentile(latences, 0.95):.0f}")
    print(f"    p99            : {percentile(latences, 0.99):.0f}")
    print(f"    min / max      : {min(latences):.0f} / {max(latences):.0f}")
    print(f"    moyenne        : {statistics.mean(latences):.0f}")

    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    if ttfts:
        print("\n  Time-to-first-token (ms) — la latence reellement percue")
        print(f"    p50            : {percentile(ttfts, 0.50):.0f}")
        print(f"    p95            : {percentile(ttfts, 0.95):.0f}")
        gain = percentile(latences, 0.50) / max(percentile(ttfts, 0.50), 1)
        print(f"    -> le premier token arrive {gain:.0f}x plus tot que la "
              f"reponse complete")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("-n", "--requests", type=int, default=10)
    p.add_argument("-c", "--concurrency", type=int, default=2)
    p.add_argument("--stream", action="store_true",
                   help="mesure aussi le time-to-first-token")
    p.add_argument("--max-new-tokens", type=int, default=160)
    args = p.parse_args()

    resultats = asyncio.run(executer(
        args.url, args.requests, args.concurrency, args.stream,
        args.max_new_tokens, os.environ.get("API_KEY")))
    rapport(resultats, args.concurrency, args.stream)


if __name__ == "__main__":
    main()
