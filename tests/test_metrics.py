"""Tests des metriques d'evaluation (logique pure, sans GPU)."""
from __future__ import annotations

import pytest

from src.metrics import (
    compute_metrics,
    intervals_disjoint,
    per_example_rouge,
    strip_accents,
)

# Bootstrap reduit : ces tests verifient la logique, pas la precision statistique.
FAST = {"n_bootstrap": 50}


def test_strip_accents():
    assert strip_accents("métriques évaluées") == "metriques evaluees"
    assert strip_accents("déjà vu à Noël") == "deja vu a Noel"
    assert strip_accents("sans accent") == "sans accent"


def test_prediction_identique_a_la_reference_donne_rouge_1():
    textes = ["Le broadcasting evite les boucles explicites."]
    m = compute_metrics(textes, textes, **FAST)
    assert m["rouge1"] == pytest.approx(1.0)
    assert m["rougeL"] == pytest.approx(1.0)


def test_prediction_sans_rapport_donne_un_score_bas():
    m = compute_metrics(["chat chien oiseau"], ["tenseur gradient reseau"], **FAST)
    assert m["rouge1"] < 0.2


def test_la_normalisation_des_accents_change_le_score():
    """C'est l'artefact detecte au Tier 1 : sans normalisation, une difference
    purement orthographique fait chuter ROUGE."""
    pred = ["Les metriques sont evaluees sur le corpus"]
    ref = ["Les métriques sont évaluées sur le corpus"]

    brut = compute_metrics(pred, ref, normalize_accents=False, **FAST)
    norm = compute_metrics(pred, ref, normalize_accents=True, **FAST)

    assert norm["rouge1"] > brut["rouge1"]
    assert norm["rouge1"] == pytest.approx(1.0)


def test_la_normalisation_est_active_par_defaut():
    pred = ["Les metriques evaluees"]
    ref = ["Les métriques évaluées"]
    assert compute_metrics(pred, ref, **FAST)["rouge1"] == pytest.approx(1.0)


def test_les_intervalles_encadrent_lestimation_ponctuelle():
    preds = [f"reponse numero {i} sur le sujet traite" for i in range(12)]
    refs = [f"reponse numero {i} sur un autre sujet" for i in range(12)]
    m = compute_metrics(preds, refs, n_bootstrap=200)

    for key in ("rouge1", "rouge2", "rougeL", "bleu"):
        lo, hi = m["ci95"][key]
        assert lo <= m[key] <= hi, f"{key} hors de son IC"


def test_n_samples_est_correct():
    preds = ["a b c", "d e f"]
    assert compute_metrics(preds, preds, **FAST)["n_samples"] == 2


def test_erreur_si_tailles_incoherentes():
    with pytest.raises(ValueError, match="incoherentes"):
        compute_metrics(["a", "b"], ["a"], **FAST)


def test_erreur_si_aucune_prediction():
    with pytest.raises(ValueError, match="Aucune prediction"):
        compute_metrics([], [], **FAST)


def test_per_example_rouge_renvoie_un_score_par_exemple():
    scores = per_example_rouge(["a b", "c d"], ["a b", "x y"])
    assert len(scores) == 2
    assert scores[0]["rouge1"] > scores[1]["rouge1"]


def test_intervals_disjoint():
    a = {"ci95": {"rouge1": [0.10, 0.20]}}
    b = {"ci95": {"rouge1": [0.30, 0.40]}}
    c = {"ci95": {"rouge1": [0.15, 0.35]}}

    assert intervals_disjoint(a, b, "rouge1")
    assert not intervals_disjoint(a, c, "rouge1")
    assert intervals_disjoint(b, a, "rouge1")     # symetrique
