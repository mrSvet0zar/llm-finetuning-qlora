"""
Test de non-regression du bug de blocage de la boucle d'evenements.

Le bug
------
`api_server.generate_endpoint` etait declare `async def` mais appelait une
fonction de generation SYNCHRONE de ~12 s. Une coroutine qui effectue un
travail bloquant monopolise la boucle d'evenements : pendant une generation,
le serveur ne repondait plus a rien, pas meme /health. Derriere un
orchestrateur, cela fait redemarrer un conteneur pourtant sain.

Le test
-------
On simule une generation lente (sans GPU ni modele) et l'on COMPTE les tours de
boucle d'evenements effectues pendant ce temps. Boucle libre -> des dizaines de
tours ; boucle bloquee -> quasiment aucun.

Deux versions anterieures de ce test ont ete ecartees :
  * mesurer la latence de /health APRES la generation : le blocage se produisait
    avant meme le debut de la mesure, le test passait donc aussi avec le bug ;
  * mesurer le retard d'un `asyncio.sleep` : discriminant (17 ms contre 955 ms)
    mais INSTABLE, car un seuil en temps reel echoue quand la machine est
    chargee. Un test intermittent est pire qu'absent.

Compter les tours de boucle garde le pouvoir discriminant sans dependre d'un
seuil temporel : meme sur une machine saturee, une boucle libre effectue
beaucoup de tours, une boucle bloquee n'en fait aucun.

Ces tests n'importent pas torch : `api_server` ne le charge qu'a l'interieur
des fonctions, jamais au niveau module. Ils tournent donc en CI.
"""
from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("fastapi", reason="fastapi requis")
pytest.importorskip("httpx", reason="httpx requis")

import httpx  # noqa: E402

import api_server  # noqa: E402

GENERATION_S = 1.0        # duree simulee d'une generation


@pytest.fixture
def app_pret(monkeypatch):
    """Prepare l'app avec un faux modele : ni GPU ni chargement de poids."""
    class FauxTokenizer:
        def __call__(self, text):
            return {"input_ids": list(range(len(text.split())))}

    monkeypatch.setitem(api_server._state, "model", object())
    monkeypatch.setitem(api_server._state, "tokenizer", FauxTokenizer())
    monkeypatch.setitem(api_server._state, "semaphore", asyncio.Semaphore(2))
    monkeypatch.setitem(api_server._state, "load_time_s", 0.0)

    def fausse_generation(prompt, max_new_tokens, temperature, top_p):
        time.sleep(GENERATION_S)          # bloquant, comme la vraie generation
        return "reponse simulee du modele", 4

    monkeypatch.setattr(api_server, "_generate_blocking", fausse_generation)

    # Quotas larges : ce fichier teste la concurrence, pas le rate limiting.
    monkeypatch.setattr(api_server, "rate_limiter",
                        api_server.RateLimiter(rate=1000.0, capacity=1000.0))
    monkeypatch.setattr(api_server, "metrics", api_server.MetricsCollector())
    monkeypatch.setattr(api_server, "API_KEY", None)
    return api_server.app


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_la_boucle_reste_libre_pendant_une_generation(app_pret):
    """LE test de non-regression.

    On compte les tours de boucle d'evenements effectues pendant qu'une
    generation bloquante est en cours. Avec le bug d'origine, la boucle est
    monopolisee et le compteur reste a zero.
    """
    TOURS_MINIMUM = 10          # tres en dessous de ce qu'une boucle libre fait

    async def scenario():
        async with _client(app_pret) as client:
            tours = 0
            generation = asyncio.create_task(
                client.post("/generate", json={"prompt": "question"}))

            # Tant que la generation tourne, la boucle doit continuer a nous
            # rendre la main.
            while not generation.done():
                await asyncio.sleep(0.01)
                tours += 1
                if tours > 500:      # garde-fou : ne jamais boucler sans fin
                    break

            sante = await client.get("/health")
            return tours, sante, await generation

    tours, sante, reponse = asyncio.run(scenario())

    assert sante.status_code == 200
    assert reponse.status_code == 200
    assert tours >= TOURS_MINIMUM, (
        f"seulement {tours} tours de boucle pendant une generation de "
        f"{GENERATION_S} s : la boucle est monopolisee au lieu de deleguer "
        f"le travail bloquant a un threadpool")


def test_deux_generations_concurrentes_ne_se_serialisent_pas_deux_fois(app_pret):
    """Avec une concurrence de 2, deux requetes doivent se recouvrir."""
    async def scenario():
        async with _client(app_pret) as client:
            t0 = time.perf_counter()
            await asyncio.gather(
                client.post("/generate", json={"prompt": "a"}),
                client.post("/generate", json={"prompt": "b"}),
            )
            return time.perf_counter() - t0

    duree = asyncio.run(scenario())
    assert duree < GENERATION_S * 1.8, (
        f"{duree:.2f} s pour deux generations concurrentes : elles semblent "
        f"entierement serialisees")


def test_le_semaphore_borne_la_concurrence(monkeypatch, app_pret):
    """Avec une concurrence de 1, les generations sont serialisees : c'est
    voulu, un GPU de 8 Go ne peut pas en traiter deux a la fois."""
    monkeypatch.setitem(api_server._state, "semaphore", asyncio.Semaphore(1))

    async def scenario():
        async with _client(app_pret) as client:
            t0 = time.perf_counter()
            await asyncio.gather(
                client.post("/generate", json={"prompt": "a"}),
                client.post("/generate", json={"prompt": "b"}),
            )
            return time.perf_counter() - t0

    duree = asyncio.run(scenario())
    assert duree >= GENERATION_S * 1.8


def test_ready_signale_labsence_de_modele(monkeypatch):
    """Readiness distincte de liveness : sans modele, /ready doit renvoyer 503."""
    monkeypatch.setattr(api_server, "_state", {})

    async def scenario():
        async with _client(api_server.app) as client:
            return await client.get("/ready"), await client.get("/health")

    ready, health = asyncio.run(scenario())
    assert ready.status_code == 503
    assert health.status_code == 200        # le processus vit, lui


def test_reponse_contient_les_metriques_de_la_requete(app_pret):
    async def scenario():
        async with _client(app_pret) as client:
            return await client.post("/generate", json={"prompt": "question"})

    data = asyncio.run(scenario()).json()
    assert data["response"] == "reponse simulee du modele"
    assert data["tokens_generated"] == 4
    assert data["latency_ms"] > 0
    assert len(data["request_id"]) == 12


def test_prompt_trop_long_rejete(app_pret):
    async def scenario():
        async with _client(app_pret) as client:
            return await client.post(
                "/generate", json={"prompt": "x" * (api_server.MAX_PROMPT_CHARS + 1)})

    assert asyncio.run(scenario()).status_code == 422


def test_max_new_tokens_hors_bornes_rejete(app_pret):
    async def scenario():
        async with _client(app_pret) as client:
            return await client.post(
                "/generate", json={"prompt": "q", "max_new_tokens": 99999})

    assert asyncio.run(scenario()).status_code == 422


def test_cle_api_exigee_si_configuree(monkeypatch, app_pret):
    monkeypatch.setattr(api_server, "API_KEY", "secret-attendu")

    async def scenario():
        async with _client(app_pret) as client:
            sans = await client.post("/generate", json={"prompt": "q"})
            mauvaise = await client.post("/generate", json={"prompt": "q"},
                                         headers={"X-API-Key": "faux"})
            bonne = await client.post("/generate", json={"prompt": "q"},
                                      headers={"X-API-Key": "secret-attendu"})
            return sans, mauvaise, bonne

    sans, mauvaise, bonne = asyncio.run(scenario())
    assert sans.status_code == 401
    assert mauvaise.status_code == 401
    assert bonne.status_code == 200


def test_rate_limiting_renvoie_429(monkeypatch, app_pret):
    monkeypatch.setattr(api_server, "rate_limiter",
                        api_server.RateLimiter(rate=0.001, capacity=1.0))

    async def scenario():
        async with _client(app_pret) as client:
            premiere = await client.post("/generate", json={"prompt": "q"})
            seconde = await client.post("/generate", json={"prompt": "q"})
            return premiere, seconde

    premiere, seconde = asyncio.run(scenario())
    assert premiere.status_code == 200
    assert seconde.status_code == 429
    assert "Retry-After" in seconde.headers


def test_endpoint_metrics(app_pret):
    async def scenario():
        async with _client(app_pret) as client:
            await client.post("/generate", json={"prompt": "q"})
            return await client.get("/metrics"), await client.get("/metrics/prometheus")

    metrics, prom = asyncio.run(scenario())
    assert metrics.json()["requests_total"] == 1
    assert "inference_requests_total 1" in prom.text
