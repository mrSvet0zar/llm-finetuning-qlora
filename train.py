"""
Fine-tuning QLoRA de Qwen2.5-3B-Instruct sur le dataset Python/DS/ML.

Points cles de l'implementation (choix techniques documentes) :
  * Quantization 4-bit NF4 + double quant  -> tient dans 8 Go de VRAM
  * Adaptateurs LoRA sur toutes les projections attention + MLP
  * Chat template ChatML natif de Qwen (via apply_chat_template)
  * MASQUAGE DU PROMPT : la loss n'est calculee que sur la reponse de
    l'assistant (labels = -100 sur systeme + question)
  * SDPA (pas flash-attn : ne compile pas sous Windows)
  * gradient checkpointing + optimiseur pagine 8-bit
  * Suivi TensorBoard, early-stopping via load_best_model_at_end

Usage :
    python train.py
Puis, pour visualiser :
    tensorboard --logdir outputs/qwen2.5-3b-pyds-lora/logs
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Config  # noqa: E402


SYSTEM_PROMPT = (
    "Tu es un assistant expert en Python, data science et machine learning. "
    "Tu reponds de facon claire, correcte et concise, avec des exemples "
    "pertinents quand c'est utile."
)


# ---------------------------------------------------------------------------
#  Modele & tokenizer
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(cfg: Config):
    bnb_config = None
    if cfg.model.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.model.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.model.use_nested_quant,
        )

    print(f"Chargement du modele : {cfg.model.model_name} (4-bit={cfg.model.use_4bit})")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        quantization_config=bnb_config,
        device_map=cfg.model.device_map,
        trust_remote_code=cfg.model.trust_remote_code,
        attn_implementation=cfg.model.attn_implementation,
        torch_dtype=getattr(torch, cfg.model.bnb_4bit_compute_dtype),
    )
    model.config.use_cache = False        # incompatible avec gradient checkpointing
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"       # important en entrainement causal

    return model, tokenizer


def setup_lora(model, cfg: Config):
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
#  Dataset : tokenisation avec chat template + masquage du prompt
# ---------------------------------------------------------------------------
def build_tokenize_fn(tokenizer, cfg: Config):
    max_len = cfg.model.max_seq_length

    def tokenize(example):
        user_content = example["instruction"]
        if example.get("input"):
            user_content += "\n\n" + example["input"]

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": example["output"]},
        ]

        # Ids du prompt seul (avec l'amorce de generation de l'assistant).
        # transformers 5.x renvoie un BatchEncoding -> on extrait input_ids.
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=True,
        )["input_ids"]
        # Ids de la sequence complete (prompt + reponse + fin de tour)
        full_ids = tokenizer.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=True,
        )["input_ids"]

        # Troncature si trop long
        full_ids = full_ids[:max_len]

        # Masquage : -100 sur toute la partie prompt, reponse conservee
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return tokenize


@dataclass
class CausalCollator:
    """Padding dynamique : input_ids -> pad_id, labels -> -100, mask -> 0."""
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention, labels = [], [], []
        for f in features:
            n = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * n)
            attention.append(f["attention_mask"] + [0] * n)
            labels.append(f["labels"] + [-100] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def prepare_datasets(cfg: Config, tokenizer):
    processed = Path(cfg.data.processed_dir)
    ds = load_dataset("json", data_files={
        "train": str(processed / "train.jsonl"),
        "validation": str(processed / "val.jsonl"),
    })
    tokenize_fn = build_tokenize_fn(tokenizer, cfg)
    # num_proc=1 : evite les blocages de multiprocessing sous Windows
    ds = ds.map(tokenize_fn, remove_columns=ds["train"].column_names, num_proc=1)
    return ds["train"], ds["validation"]


# ---------------------------------------------------------------------------
#  Entrainement
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = Config()
    set_seed(cfg.train.seed)

    if not torch.cuda.is_available():
        print("ATTENTION : aucun GPU CUDA detecte. QLoRA requiert un GPU NVIDIA.")

    model, tokenizer = load_model_and_tokenizer(cfg)
    model = setup_lora(model, cfg)

    train_ds, eval_ds = prepare_datasets(cfg, tokenizer)
    print(f"\nDataset : {len(train_ds)} train / {len(eval_ds)} val")

    out_dir = Path(cfg.train.output_dir)
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_steps=cfg.train.warmup_steps,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optim,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.train.logging_steps,
        eval_strategy=cfg.train.eval_strategy,
        eval_steps=cfg.train.eval_steps,
        save_strategy=cfg.train.save_strategy,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        load_best_model_at_end=cfg.train.load_best_model_at_end,
        metric_for_best_model=cfg.train.metric_for_best_model,
        greater_is_better=cfg.train.greater_is_better,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=[cfg.train.report_to],
        run_name=cfg.train.run_name,
        seed=cfg.train.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CausalCollator(pad_token_id=tokenizer.pad_token_id),
    )

    print("\n=== Debut de l'entrainement ===")
    train_result = trainer.train()

    print("\n=== Sauvegarde de l'adaptateur LoRA ===")
    trainer.save_model(str(out_dir))                # adaptateur + config
    tokenizer.save_pretrained(str(out_dir))

    # Journalisation des metriques finales
    metrics = train_result.metrics
    eval_metrics = trainer.evaluate()
    metrics.update(eval_metrics)
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nMetriques finales :")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nAdaptateur sauvegarde dans : {out_dir}")


if __name__ == "__main__":
    main()
