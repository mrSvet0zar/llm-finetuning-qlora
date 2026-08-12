"""Smoke test rapide : valide le chat template + masquage sans charger le modele."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from src.config import Config
from train import build_tokenize_fn, SYSTEM_PROMPT, CausalCollator

cfg = Config()
tok = AutoTokenizer.from_pretrained(cfg.model.model_name, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

fn = build_tokenize_fn(tok, cfg)
example = {
    "instruction": "Qu'est-ce que LoRA ?",
    "input": "",
    "output": "LoRA entraine de petits adaptateurs de rang faible au lieu de tous les poids.",
}
out = fn(example)
n = len(out["input_ids"])
n_masked = sum(1 for l in out["labels"] if l == -100)
n_train = n - n_masked

print(f"Tokens totaux        : {n}")
print(f"Tokens masques (-100): {n_masked}  (prompt)")
print(f"Tokens de loss       : {n_train}  (reponse)")
assert n_train > 0, "Aucun token de reponse — masquage casse !"
assert n_masked > 0, "Aucun token masque — le prompt n'est pas masque !"

# Verifie que les tokens de reponse decodent bien la reponse
resp_ids = [t for t, l in zip(out["input_ids"], out["labels"]) if l != -100]
decoded = tok.decode(resp_ids, skip_special_tokens=True)
print(f"\nReponse reconstruite depuis les labels :\n  {decoded!r}")

# Test du collator sur un batch de 2 (longueurs differentes)
ex2 = fn({"instruction": "Explique .loc vs .iloc dans Pandas en detail.", "input": "",
          "output": ".loc par etiquette, .iloc par position entiere."})
batch = CausalCollator(pad_token_id=tok.pad_token_id)([out, ex2])
print(f"\nBatch collate : input_ids {tuple(batch['input_ids'].shape)}, "
      f"labels {tuple(batch['labels'].shape)}, mask {tuple(batch['attention_mask'].shape)}")
assert batch["input_ids"].shape == batch["labels"].shape
print("\nSMOKE TEST OK")
