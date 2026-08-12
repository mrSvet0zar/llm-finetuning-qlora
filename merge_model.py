"""
Fusion (merge) de l'adaptateur LoRA dans le modele de base.

Produit un modele autonome (checkpoint standard) exploitable sans PEFT, et
convertible en GGUF pour Ollama/llama.cpp. On recharge ici le modele de base
en pleine precision (fp16) -- PAS en 4-bit -- car on ne peut pas fusionner
proprement des poids quantises. Prevoir ~6 Go de RAM/VRAM pour un modele 3B.

Usage :
    python merge_model.py
    python merge_model.py --adapter ./outputs/... --out ./outputs/merged-model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Config  # noqa: E402


def merge(adapter_path: str, output_path: str, cfg: Config) -> None:
    print(f"Chargement du modele de base (fp16) : {cfg.model.model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",              # merge sur CPU : evite l'OOM GPU
        trust_remote_code=cfg.model.trust_remote_code,
    )

    print(f"Application de l'adaptateur : {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Fusion des poids (merge_and_unload)...")
    model = model.merge_and_unload()

    print(f"Sauvegarde du modele fusionne : {output_path}")
    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)
    print("Fusion terminee.")


def main() -> None:
    cfg = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=cfg.train.output_dir)
    parser.add_argument("--out", type=str,
                        default=str(Path(cfg.train.output_dir).parent / "merged-model"))
    args = parser.parse_args()
    merge(args.adapter, args.out, cfg)


if __name__ == "__main__":
    main()
