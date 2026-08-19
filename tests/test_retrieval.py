"""Tests du retrieveur TF-IDF utilise par la baseline RAG."""
from __future__ import annotations

import pytest

from src.retrieval import TfidfRetriever


@pytest.fixture
def docs() -> list[dict]:
    """Corpus jouet : deux concepts, chacun en trois reformulations."""
    base = [
        ("g1", "Qu'est-ce que le broadcasting en NumPy ?",
         "Le broadcasting applique une operation entre tableaux de formes "
         "differentes sans copier les donnees."),
        ("g2", "Comment fonctionne le mecanisme d'attention ?",
         "L'attention pondere l'importance des autres tokens via des Query, "
         "Key et Value."),
        ("g3", "Qu'est-ce qu'un decorateur en Python ?",
         "Un decorateur enveloppe une fonction pour en modifier le "
         "comportement sans toucher son code."),
    ]
    docs = []
    for gid, q, a in base:
        for prefixe in ("", "Peux-tu m'expliquer : ", "En quelques phrases, "):
            docs.append({"group_id": gid, "category": "c",
                         "instruction": prefixe + q, "output": a})
    return docs


def test_renvoie_le_nombre_demande(docs):
    assert len(TfidfRetriever(docs).query("broadcasting NumPy", k=2)) == 2


def test_deduplique_par_group_id(docs):
    """Sans deduplication, le top-3 renvoyait 3 paraphrases du meme concept."""
    resultats = TfidfRetriever(docs).query("broadcasting NumPy tableaux", k=3)
    gids = [r["group_id"] for r in resultats]
    assert len(gids) == len(set(gids)), f"group_id dupliques : {gids}"


def test_retrouve_le_concept_pertinent(docs):
    top = TfidfRetriever(docs).query(
        "Comment NumPy applique-t-il une operation entre tableaux "
        "de formes differentes ?", k=1)
    assert top[0]["group_id"] == "g1"


def test_k_superieur_au_nombre_de_concepts(docs):
    """Ne doit pas boucler ni dupliquer : 3 concepts seulement existent."""
    resultats = TfidfRetriever(docs).query("python", k=10)
    assert len(resultats) <= 3
    gids = [r["group_id"] for r in resultats]
    assert len(gids) == len(set(gids))


def test_corpus_vide_leve_une_erreur():
    with pytest.raises(ValueError, match="vide"):
        TfidfRetriever([])
