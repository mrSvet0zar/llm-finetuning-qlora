"""
Configuration centrale du projet de fine-tuning.

Toute la config (modele, LoRA, hyperparametres, chemins) vit ici sous forme
de dataclasses. Les scripts (train, eval, merge, inference) importent `Config`
pour rester coherents et reproductibles.

Cible : Qwen2.5-3B-Instruct, QLoRA 4-bit, RTX 4070 Laptop 8 Go, Windows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# Racine du projet (dossier parent de src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    """Chemins et parametres liees aux donnees."""
    raw_file: str = str(PROJECT_ROOT / "data" / "raw" / "raw_qa_data.json")
    processed_dir: str = str(PROJECT_ROOT / "data" / "processed")

    train_ratio: float = 0.85
    val_ratio: float = 0.10
    # test_ratio est le reste (0.05)

    min_instruction_len: int = 10
    min_output_len: int = 20
    seed: int = 42


@dataclass
class ModelConfig:
    """Modele de base et quantization."""
    # Qwen2.5-3B-Instruct : recent, fort, tient en 8 Go en 4-bit.
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # --- Quantization 4-bit (QLoRA) ---
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"  # Ada (RTX 40xx) supporte bf16
    use_nested_quant: bool = True  # double quant : -0.4 Go environ

    # SDPA au lieu de flash_attention_2 (flash-attn ne compile pas sous Windows)
    attn_implementation: str = "sdpa"

    max_seq_length: int = 1024
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class LoraConfig_:
    """Hyperparametres des adaptateurs LoRA."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    # Cibler toutes les projections attention + MLP donne de meilleurs
    # resultats que q/v seuls, pour un cout memoire negligeable en QLoRA.
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class TrainConfig:
    """Hyperparametres d'entrainement."""
    output_dir: str = str(PROJECT_ROOT / "outputs" / "qwen2.5-3b-pyds-lora")

    num_train_epochs: int = 3
    # Batch effectif = per_device * grad_accum = 1 * 16 = 16
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16

    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 2  # transformers 5.x : warmup_ratio retire, on fixe les steps
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3

    optim: str = "paged_adamw_8bit"        # optimiseur pagine (economie VRAM)
    gradient_checkpointing: bool = True

    # Dataset compact -> ~7 steps/epoch : cadence d'eval/save resserree.
    logging_steps: int = 2
    eval_strategy: str = "steps"
    eval_steps: int = 5
    save_strategy: str = "steps"
    save_steps: int = 5  # doit etre un multiple de eval_steps (load_best_model_at_end)
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # "tensorboard" par defaut (zero login). Mettre "wandb" si configure.
    report_to: str = "tensorboard"
    run_name: str = "qwen2.5-3b-pyds-qlora"

    seed: int = 42


@dataclass
class Config:
    """Config racine agregeant les sous-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig_ = field(default_factory=LoraConfig_)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


if __name__ == "__main__":
    # Petit self-test : affiche la config resolue.
    import json
    cfg = Config()
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
