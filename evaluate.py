"""
Evaluation du modele fine-tune sur le jeu de test.

Metriques :
  * ROUGE-1 / ROUGE-2 / ROUGE-L  (recouvrement lexical, oriente rappel)
  * BLEU (sacrebleu, precision n-grammes)
Optionnellement, compare au modele de BASE (--baseline) pour mesurer le gain
apporte par le fine-tuning.

Les metriques lexicales sont un PROXY : elles ne captent pas la semantique.
Voir README pour la discussion de leurs limites.

Usage :
    python evaluate.py
    python evaluate.py --baseline        # evalue aussi le modele de base
    python evaluate.py --n 8             # nombre d'exemples de test a evaluer
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from rouge_score import rouge_scorer
import sacrebleu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Config  # noqa: E402
from inference import load_model, generate  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from train import SYSTEM_PROMPT  # noqa: E402


def load_test(cfg: Config) -> list[dict]:
    path = Path(cfg.data.processed_dir) / "test.jsonl"
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
    )
    r1 = r2 = rl = 0.0
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    n = max(len(predictions), 1)

    # BLEU au niveau du corpus (sacrebleu)
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score

    return {
        "rouge1": r1 / n,
        "rouge2": r2 / n,
        "rougeL": rl / n,
        "bleu": bleu,
        "n_samples": len(predictions),
    }


def load_baseline(cfg: Config):
    """Charge le modele de base non fine-tune (4-bit) pour comparaison."""
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, cfg.model.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.model.use_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name, quantization_config=bnb, device_map="auto",
        attn_implementation=cfg.model.attn_implementation,
        torch_dtype=getattr(torch, cfg.model.bnb_4bit_compute_dtype),
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def evaluate_model(model, tokenizer, test_data: list[dict], label: str) -> dict:
    preds, refs = [], []
    print(f"\n--- Generation avec le modele : {label} ---")
    for i, item in enumerate(test_data, 1):
        pred = generate(model, tokenizer, item["instruction"],
                        max_new_tokens=320, temperature=0.0)
        preds.append(pred)
        refs.append(item["output"])
        print(f"  [{i}/{len(test_data)}] {item['instruction'][:55]}...")
    metrics = compute_metrics(preds, refs)
    return {"metrics": metrics, "predictions": preds, "references": refs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="Evalue aussi le modele de base (comparaison)")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--n", type=int, default=None,
                        help="Limiter le nombre d'exemples de test")
    args = parser.parse_args()

    cfg = Config()
    test_data = load_test(cfg)
    if args.n:
        test_data = test_data[:args.n]
    print(f"Jeu de test : {len(test_data)} exemples")

    results = {}

    # Modele fine-tune
    adapter = args.adapter or cfg.train.output_dir
    ft_model, ft_tok = load_model(cfg, adapter, None)
    ft = evaluate_model(ft_model, ft_tok, test_data, "fine-tune (LoRA)")
    results["finetuned"] = ft["metrics"]

    # Modele de base (optionnel)
    if args.baseline:
        del ft_model
        torch.cuda.empty_cache()
        base_model, base_tok = load_baseline(cfg)
        base = evaluate_model(base_model, base_tok, test_data, "base (non fine-tune)")
        results["baseline"] = base["metrics"]

    # Rapport
    print("\n" + "=" * 60)
    print("RESULTATS")
    print("=" * 60)
    for name, m in results.items():
        print(f"\n{name} :")
        print(f"  ROUGE-1 : {m['rouge1']:.4f}")
        print(f"  ROUGE-2 : {m['rouge2']:.4f}")
        print(f"  ROUGE-L : {m['rougeL']:.4f}")
        print(f"  BLEU    : {m['bleu']:.2f}")

    if "baseline" in results:
        d1 = results["finetuned"]["rouge1"] - results["baseline"]["rouge1"]
        db = results["finetuned"]["bleu"] - results["baseline"]["bleu"]
        print(f"\nGain du fine-tuning : ROUGE-1 {d1:+.4f} | BLEU {db:+.2f}")

    out = Path(cfg.train.output_dir) / "eval_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nMetriques ecrites dans : {out}")


if __name__ == "__main__":
    main()
