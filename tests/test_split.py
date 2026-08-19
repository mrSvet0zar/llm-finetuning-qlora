"""
Tests du decoupage train/val/test.

CE FICHIER EST LE GARDE-FOU DU PROJET. La v1 souffrait d'une fuite de donnees :
l'augmentation par paraphrase etait appliquee AVANT le decoupage, si bien que
les reponses de reference du test avaient deja ete vues a l'entrainement (8/8).
Ces tests echouent si cette regression revient, sous quelque forme que ce soit.
"""
from __future__ import annotations

import pytest

from prepare_dataset import (
    _normalize,
    assert_no_leakage,
    augment,
    group_aware_split,
    validate_and_dedup,
)
from src.config import Config


@pytest.fixture
def cfg() -> Config:
    return Config()


# ---------------------------------------------------------------------------
#  Non-regression : la fuite de donnees
# ---------------------------------------------------------------------------
def test_aucun_group_id_ne_traverse_le_split(synthetic_corpus, cfg):
    """Un concept doit appartenir a un seul split."""
    train, val, test = group_aware_split(synthetic_corpus, cfg)
    g_train = {i["group_id"] for i in train}
    g_val = {i["group_id"] for i in val}
    g_test = {i["group_id"] for i in test}

    assert g_train & g_val == set()
    assert g_train & g_test == set()
    assert g_val & g_test == set()


def test_aucune_reponse_de_test_nest_vue_a_lentrainement(synthetic_corpus, cfg):
    """Le coeur du bug v1 : meme apres augmentation, les reponses du test
    ne doivent jamais apparaitre dans le train."""
    train, val, test = group_aware_split(synthetic_corpus, cfg)
    train = augment(train, cfg)          # augmentation APRES le split

    outputs_train = {_normalize(i["output"]) for i in train}
    for split, name in ((val, "val"), (test, "test")):
        fuites = [i for i in split if _normalize(i["output"]) in outputs_train]
        assert not fuites, f"{len(fuites)} reponses de {name} presentes dans le train"


def test_le_garde_fou_detecte_bien_une_fuite():
    """Test NEGATIF : un controle qu'on n'a jamais vu echouer n'est pas un
    controle. On lui soumet un jeu volontairement fuyant."""
    train = [{"group_id": "g1", "instruction": "variante A", "output": "Reponse X"}]
    test = [{"group_id": "g1", "instruction": "variante B", "output": "Reponse X"}]

    with pytest.raises(SystemExit, match="FUITE DE DONNEES"):
        assert_no_leakage(train, [], test)


def test_le_garde_fou_passe_sur_un_jeu_sain():
    train = [{"group_id": "g1", "instruction": "Q1", "output": "Reponse X"}]
    val = [{"group_id": "g2", "instruction": "Q2", "output": "Reponse Y"}]
    test = [{"group_id": "g3", "instruction": "Q3", "output": "Reponse Z"}]
    assert_no_leakage(train, val, test)      # ne doit pas lever


def test_laugmentation_ne_touche_que_le_train(synthetic_corpus, cfg):
    """val et test ne doivent contenir qu'une question par concept."""
    train, val, test = group_aware_split(synthetic_corpus, cfg)
    n_groupes_train = len({i["group_id"] for i in train})
    train_aug = augment(train, cfg)

    assert len(train_aug) > len(train), "le train doit etre augmente"
    assert len(train_aug) == n_groupes_train * 3, "1 original + 2 paraphrases"
    # val et test restent a un exemple par concept
    assert len(val) == len({i["group_id"] for i in val})
    assert len(test) == len({i["group_id"] for i in test})


# ---------------------------------------------------------------------------
#  Stratification et determinisme
# ---------------------------------------------------------------------------
def test_toutes_les_categories_sont_dans_chaque_split(synthetic_corpus, cfg):
    """Sans stratification, une categorie entiere peut disparaitre du test."""
    train, val, test = group_aware_split(synthetic_corpus, cfg)
    attendues = {i["category"] for i in synthetic_corpus}

    for split, name in ((train, "train"), (val, "val"), (test, "test")):
        presentes = {i["category"] for i in split}
        assert presentes == attendues, f"categories manquantes dans {name}"


def test_le_split_est_deterministe(synthetic_corpus, cfg):
    a = group_aware_split(synthetic_corpus, cfg)
    b = group_aware_split(synthetic_corpus, cfg)
    for split_a, split_b in zip(a, b, strict=True):
        assert [i["group_id"] for i in split_a] == [i["group_id"] for i in split_b]


def test_tous_les_exemples_sont_repartis(synthetic_corpus, cfg):
    train, val, test = group_aware_split(synthetic_corpus, cfg)
    assert len(train) + len(val) + len(test) == len(synthetic_corpus)


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------
def test_la_validation_rejette_les_entrees_incompletes(cfg):
    data = [
        {"group_id": "g1", "instruction": "Une question assez longue ?",
         "output": "Une reponse de reference suffisamment longue.", "category": "c"},
        {"group_id": "g2", "instruction": "", "output": "Reponse", "category": "c"},
        {"group_id": "g3", "instruction": "Question ?", "output": "", "category": "c"},
        {"group_id": "g4", "instruction": "court", "output": "court", "category": "c"},
    ]
    assert len(validate_and_dedup(data, cfg)) == 1


def test_la_validation_deduplique_sur_linstruction(cfg):
    data = [
        {"group_id": "g1", "instruction": "Une question assez longue ?",
         "output": "Une reponse de reference suffisamment longue.", "category": "c"},
        {"group_id": "g2", "instruction": "  UNE QUESTION ASSEZ LONGUE ?  ",
         "output": "Une autre reponse de reference bien assez longue.", "category": "c"},
    ]
    assert len(validate_and_dedup(data, cfg)) == 1
