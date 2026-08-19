"""Tests des briques du service d'inference (logique pure, sans GPU)."""
from __future__ import annotations

import pytest

from src.serving import MetricsCollector, RateLimiter, TokenBucket, percentile


# ---------------------------------------------------------------------------
#  Percentiles
# ---------------------------------------------------------------------------
def test_percentile_liste_vide():
    assert percentile([], 0.5) == 0.0


def test_percentile_valeur_unique():
    assert percentile([42.0], 0.95) == 42.0


def test_percentile_mediane():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3


def test_percentile_bornes():
    valeurs = [10, 20, 30, 40]
    assert percentile(valeurs, 0.0) == 10
    assert percentile(valeurs, 1.0) == 40


def test_percentile_ordonne_les_valeurs():
    assert percentile([5, 1, 3], 0.5) == 3


# ---------------------------------------------------------------------------
#  Metriques
# ---------------------------------------------------------------------------
def test_compte_succes_et_erreurs():
    m = MetricsCollector()
    m.record_success(100.0, 10)
    m.record_success(200.0, 20)
    m.record_error()

    s = m.snapshot()
    assert s["requests_total"] == 3
    assert s["errors_total"] == 1
    assert s["tokens_generated_total"] == 30
    assert s["error_rate"] == pytest.approx(1 / 3)


def test_taux_derreur_sans_requete():
    assert MetricsCollector().snapshot()["error_rate"] == 0.0


def test_percentiles_de_latence():
    m = MetricsCollector()
    for v in range(1, 101):
        m.record_success(float(v), 1)
    s = m.snapshot()
    assert s["latency_ms"]["p50"] == pytest.approx(50.5, abs=1)
    assert s["latency_ms"]["p95"] == pytest.approx(95.5, abs=1)
    assert s["latency_ms"]["max"] == 100.0


def test_fenetre_glissante_bornee():
    """La memoire ne doit pas croitre indefiniment en production."""
    m = MetricsCollector(window=10)
    for v in range(100):
        m.record_success(float(v), 1)
    s = m.snapshot()
    assert s["latency_ms"]["count"] == 10          # fenetre bornee
    assert s["requests_total"] == 100              # mais le compteur est complet


def test_debit_en_tokens_par_seconde():
    m = MetricsCollector()
    m.record_success(1000.0, 50)      # 50 tokens en 1 s
    assert m.snapshot()["tokens_per_s"]["p50"] == pytest.approx(50.0)


def test_format_prometheus():
    m = MetricsCollector()
    m.record_success(120.0, 12)
    texte = m.prometheus()
    assert "inference_requests_total 1" in texte
    assert 'inference_latency_ms{quantile="0.95"}' in texte
    assert texte.endswith("\n")


# ---------------------------------------------------------------------------
#  Limitation de debit
# ---------------------------------------------------------------------------
def test_le_seau_tolere_une_rafale_puis_bloque():
    bucket = TokenBucket(rate=1.0, capacity=3.0)
    now = 1000.0
    assert [bucket.allow(now) for _ in range(3)] == [True, True, True]
    assert bucket.allow(now) is False           # reserve epuisee


def test_le_seau_se_recharge_avec_le_temps():
    bucket = TokenBucket(rate=2.0, capacity=2.0)
    now = 500.0
    bucket.allow(now)
    bucket.allow(now)
    assert bucket.allow(now) is False
    assert bucket.allow(now + 1.0) is True      # 2 jetons/s -> recharge


def test_le_seau_ne_depasse_pas_sa_capacite():
    bucket = TokenBucket(rate=10.0, capacity=2.0)
    now = 0.0
    bucket.allow(now)
    # Longue inactivite : la reserve plafonne a `capacity`
    assert bucket.allow(now + 100) is True
    assert bucket.allow(now + 100) is True
    assert bucket.allow(now + 100) is False


def test_retry_after_est_positif_quand_bloque():
    bucket = TokenBucket(rate=1.0, capacity=1.0)
    now = 0.0
    bucket.allow(now)
    assert bucket.allow(now) is False
    assert bucket.retry_after_s() > 0


def test_parametres_invalides():
    with pytest.raises(ValueError):
        TokenBucket(rate=0, capacity=5)
    with pytest.raises(ValueError):
        TokenBucket(rate=1, capacity=0)


def test_les_clients_sont_isoles():
    """Un client bruyant ne doit pas consommer le quota des autres."""
    limiter = RateLimiter(rate=1.0, capacity=2.0)
    assert limiter.allow("client-a")[0] is True
    assert limiter.allow("client-a")[0] is True
    assert limiter.allow("client-a")[0] is False
    assert limiter.allow("client-b")[0] is True     # quota independant


def test_le_limiteur_renvoie_un_delai():
    limiter = RateLimiter(rate=1.0, capacity=1.0)
    limiter.allow("c")
    ok, retry_after = limiter.allow("c")
    assert ok is False
    assert retry_after > 0
