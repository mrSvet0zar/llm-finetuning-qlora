"""
Tests d'integrite du corpus reel (data/corpus/*.json).

Ces tests protegent la qualite des donnees a chaque ajout : ce sont elles qui
plafonnent la qualite du modele.
"""
from __future__ import annotations

import json
import unicodedata

import pytest

CORPUS_GLOB = "data/corpus/*.json"
MIN_INSTRUCTION = 10
MIN_OUTPUT = 20


@pytest.fixture(scope="module")
def corpus_files(request):
    root = request.config.rootpath
    files = sorted(root.glob(CORPUS_GLOB))
    assert files, f"aucun fichier trouve dans {CORPUS_GLOB}"
    return files


@pytest.fixture(scope="module")
def entries(corpus_files):
    out = []
    for path in corpus_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, entry in enumerate(data):
            out.append((path.stem, i, entry))
    return out


def test_les_fichiers_sont_du_json_valide(corpus_files):
    for path in corpus_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, list), f"{path.name} doit contenir une liste"
        assert data, f"{path.name} est vide"


def test_chaque_entree_a_instruction_et_output(entries):
    for stem, i, entry in entries:
        assert entry.get("instruction", "").strip(), f"{stem}[{i}] : instruction vide"
        assert entry.get("output", "").strip(), f"{stem}[{i}] : output vide"


def test_les_longueurs_minimales_sont_respectees(entries):
    for stem, i, entry in entries:
        assert len(entry["instruction"]) >= MIN_INSTRUCTION, f"{stem}[{i}] trop court"
        assert len(entry["output"]) >= MIN_OUTPUT, f"{stem}[{i}] : output trop court"


def test_aucune_instruction_dupliquee(entries):
    vues: dict[str, str] = {}
    for stem, i, entry in entries:
        cle = " ".join(entry["instruction"].lower().split())
        assert cle not in vues, f"{stem}[{i}] duplique {vues[cle]}"
        vues[cle] = f"{stem}[{i}]"


def test_les_group_id_generes_sont_uniques(corpus_files):
    """Le group_id est l'unite indivisible du split : il doit etre unique."""
    ids = set()
    for path in corpus_files:
        category = path.stem.replace("_", "-")
        data = json.loads(path.read_text(encoding="utf-8"))
        for i in range(len(data)):
            gid = f"{category}-{i:03d}"
            assert gid not in ids, f"group_id duplique : {gid}"
            ids.add(gid)


def test_le_corpus_est_ecrit_sans_accents(entries):
    """Le corpus est volontairement homogene (sans accents).

    Ce test verrouille cette convention : un melange accentue/non accentue
    biaiserait les metriques lexicales (cf. l'artefact documente dans le
    README). Le jour ou l'on repasse le corpus en francais accentue, il faudra
    inverser ce test — c'est justement le point : le choix doit rester
    EXPLICITE et verifie.
    """
    for stem, i, entry in entries:
        for champ in ("instruction", "output"):
            texte = entry[champ]
            sans = "".join(
                c for c in unicodedata.normalize("NFD", texte)
                if unicodedata.category(c) != "Mn"
            )
            assert texte == unicodedata.normalize("NFC", sans), (
                f"{stem}[{i}].{champ} contient des accents")


def test_le_corpus_couvre_plusieurs_categories(corpus_files):
    assert len(corpus_files) >= 5, "au moins 5 categories attendues"
