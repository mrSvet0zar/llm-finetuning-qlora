"""
Tests du masquage de prompt et du collateur.

Marques `slow` : ils necessitent torch, transformers et le telechargement du
tokenizer. La CI les exclut (`-m "not slow"`) pour rester rapide et sans reseau ;
ils sont a lancer en local avant un changement touchant la tokenisation.

    pytest -m slow
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch requis")
pytest.importorskip("transformers", reason="transformers requis")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    from src.config import Config
    tok = AutoTokenizer.from_pretrained(Config().model.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def exemple() -> dict:
    return {
        "instruction": "Qu'est-ce que LoRA ?",
        "input": "",
        "output": "LoRA entraine de petits adaptateurs de rang faible.",
    }


def test_le_prompt_est_masque_et_la_reponse_conservee(tokenizer, exemple):
    from src.config import Config
    from train import build_tokenize_fn

    out = build_tokenize_fn(tokenizer, Config())(exemple)
    labels = out["labels"]

    n_masques = sum(1 for x in labels if x == -100)
    n_appris = len(labels) - n_masques

    assert n_masques > 0, "le prompt doit etre masque"
    assert n_appris > 0, "la reponse doit contribuer a la loss"
    # Le masquage est un prefixe contigu : aucun -100 apres le premier token appris
    premier_appris = next(i for i, x in enumerate(labels) if x != -100)
    assert all(x != -100 for x in labels[premier_appris:])


def test_les_labels_reconstituent_la_reponse(tokenizer, exemple):
    from src.config import Config
    from train import build_tokenize_fn

    out = build_tokenize_fn(tokenizer, Config())(exemple)
    ids = [t for t, lab in zip(out["input_ids"], out["labels"], strict=True) if lab != -100]
    decode = tokenizer.decode(ids, skip_special_tokens=True)
    assert "adaptateurs" in decode


def test_longueurs_coherentes(tokenizer, exemple):
    from src.config import Config
    from train import build_tokenize_fn

    out = build_tokenize_fn(tokenizer, Config())(exemple)
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])


def test_troncature_a_max_seq_length(tokenizer):
    from src.config import Config
    from train import build_tokenize_fn

    cfg = Config()
    cfg.model.max_seq_length = 32
    out = build_tokenize_fn(tokenizer, cfg)({
        "instruction": "Question " * 200, "input": "", "output": "Reponse " * 200,
    })
    assert len(out["input_ids"]) <= 32


def test_le_collateur_pad_correctement(tokenizer):
    import torch

    from src.config import Config
    from train import CausalCollator, build_tokenize_fn

    fn = build_tokenize_fn(tokenizer, Config())
    a = fn({"instruction": "Question courte ?", "input": "", "output": "Reponse."})
    b = fn({"instruction": "Une question nettement plus longue que la premiere ?",
            "input": "", "output": "Une reponse elle aussi bien plus longue."})

    batch = CausalCollator(pad_token_id=tokenizer.pad_token_id)([a, b])

    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    # Le padding ne doit jamais contribuer a la loss
    padding = batch["attention_mask"] == 0
    assert torch.all(batch["labels"][padding] == -100)
