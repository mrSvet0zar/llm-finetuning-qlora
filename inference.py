"""
Inference avec le modele fine-tune.

Charge le modele de base + l'adaptateur LoRA (par defaut) OU un modele deja
fusionne (--merged), applique le chat template ChatML, et genere une reponse.

Usage :
    python inference.py                       # prompts de demonstration
    python inference.py --prompt "Ta question ?"
    python inference.py --merged ./outputs/merged-model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Config  # noqa: E402
from train import SYSTEM_PROMPT  # reutilise le meme system prompt  # noqa: E402


def load_model(cfg: Config, adapter_path: str | None, merged_path: str | None):
    """Charge soit base+adaptateur (4-bit), soit un modele fusionne."""
    if merged_path:
        print(f"Chargement du modele fusionne : {merged_path}")
        model = AutoModelForCausalLM.from_pretrained(
            merged_path, torch_dtype=torch.float16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(merged_path)
    else:
        from peft import PeftModel
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, cfg.model.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.model.use_nested_quant,
        )
        print(f"Chargement base {cfg.model.model_name} + adaptateur {adapter_path}")
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name, quantization_config=bnb, device_map="auto",
            attn_implementation=cfg.model.attn_implementation,
            torch_dtype=getattr(torch, cfg.model.bnb_4bit_compute_dtype),
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def generate_from_messages(model, tokenizer, messages: list[dict],
                           max_new_tokens: int = 320,
                           temperature: float = 0.7, top_p: float = 0.9) -> str:
    """Generation a partir d'une conversation arbitraire.

    Separee de `generate` pour permettre aux baselines (few-shot, RAG) de
    construire leurs propres messages tout en partageant exactement la meme
    procedure de decodage — condition d'une comparaison equitable.
    """
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0:
        # Echantillonnage : on ne passe temperature/top_p qu'en mode sampling
        # (sinon transformers 5.x emet un warning "flags ignored").
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    outputs = model.generate(**gen_kwargs)
    # Ne decoder que les tokens generes (apres le prompt)
    generated = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate(model, tokenizer, question: str, **kwargs) -> str:
    """Generation standard : system prompt du projet + question utilisateur."""
    return generate_from_messages(model, tokenizer, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ], **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None,
                        help="Chemin de l'adaptateur LoRA (defaut: output_dir de la config)")
    parser.add_argument("--merged", type=str, default=None,
                        help="Chemin d'un modele deja fusionne")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    cfg = Config()
    adapter = args.adapter or cfg.train.output_dir
    model, tokenizer = load_model(cfg, adapter, args.merged)

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "Qu'est-ce que la vectorisation en Python ?",
            "Explique LoRA en deux phrases.",
            "Quelle est la difference entre .loc et .iloc dans Pandas ?",
            "Pourquoi separer les donnees en train/validation/test ?",
        ]

    for p in prompts:
        print("\n" + "=" * 70)
        print(f"Q : {p}")
        print("-" * 70)
        print(generate(model, tokenizer, p,
                       max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature))


if __name__ == "__main__":
    main()
