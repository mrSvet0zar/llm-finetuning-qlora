"""
Tests de coherence de la configuration.

Ces tests attrapent des incoherences qui, sinon, ne se manifesteraient qu'apres
plusieurs minutes d'entrainement — voire silencieusement.
"""
from __future__ import annotations

from src.config import Config


def test_save_steps_est_multiple_de_eval_steps():
    """Exigence de `load_best_model_at_end` : sans cela, transformers leve une
    erreur seulement au demarrage du Trainer, apres le chargement du modele."""
    cfg = Config()
    assert cfg.train.save_steps % cfg.train.eval_steps == 0


def test_les_ratios_de_split_sont_coherents():
    cfg = Config()
    assert 0 < cfg.data.train_ratio < 1
    assert 0 < cfg.data.val_ratio < 1
    assert cfg.data.train_ratio + cfg.data.val_ratio < 1, "il faut laisser du test"


def test_le_test_recoit_une_part_suffisante():
    cfg = Config()
    part_test = 1 - cfg.data.train_ratio - cfg.data.val_ratio
    assert part_test >= 0.10, "un jeu de test trop petit ne mesure rien"


def test_alpha_lora_suit_la_convention_2r():
    cfg = Config()
    assert cfg.lora.alpha == 2 * cfg.lora.r


def test_les_modules_lora_couvrent_attention_et_mlp():
    cfg = Config()
    attendus = {"q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"}
    assert attendus.issubset(set(cfg.lora.target_modules))


def test_le_batch_effectif_est_raisonnable():
    cfg = Config()
    effectif = (cfg.train.per_device_train_batch_size
                * cfg.train.gradient_accumulation_steps)
    assert 8 <= effectif <= 64


def test_early_stopping_active():
    """Une fois la fuite corrigee, la validation remonte des ~1 epoch :
    l'early stopping n'est pas optionnel."""
    cfg = Config()
    assert cfg.train.load_best_model_at_end
    assert cfg.train.metric_for_best_model == "eval_loss"
    assert cfg.train.greater_is_better is False
    assert cfg.train.early_stopping_patience >= 1


def test_la_config_est_serialisable():
    """Necessaire pour journaliser les hyperparametres d'un run."""
    d = Config().to_dict()
    assert d["model"]["model_name"]
    assert "learning_rate" in d["train"]


def test_attention_compatible_windows():
    """flash_attention_2 ne compile pas sous Windows : le projet cible SDPA."""
    assert Config().model.attn_implementation == "sdpa"
