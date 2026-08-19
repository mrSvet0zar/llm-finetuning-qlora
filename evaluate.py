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
import random
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


def _per_example_rouge(predictions: list[str], references: list[str]) -> list[dict]:
    """Scores ROUGE par exemple (necessaires pour le bootstrap)."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
    )
    out = []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        out.append({k: s[k].fmeasure for k in ("rouge1", "rouge2", "rougeL")})
    return out


def compute_metrics(predictions: list[str], references: list[str],
                    n_bootstrap: int = 1000, seed: int = 42) -> dict:
    """Metriques ponctuelles + intervalles de confiance a 95 % par bootstrap.

    Sur un jeu de test de quelques dizaines d'exemples, un score isole n'est
    qu'une estimation bruitee. On reechantillonne le jeu avec remise pour
    estimer la dispersion : si les intervalles de deux modeles se recouvrent
    largement, la difference observee n'est pas etablie.
    """
    n = len(predictions)
    per_ex = _per_example_rouge(predictions, references)

    point = {k: sum(e[k] for e in per_ex) / max(n, 1)
             for k in ("rouge1", "rouge2", "rougeL")}
    point["bleu"] = sacrebleu.corpus_bleu(predictions, [references]).score

    # --- Bootstrap ---
    rng = random.Random(seed)
    samples = {k: [] for k in ("rouge1", "rouge2", "rougeL", "bleu")}
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        for k in ("rouge1", "rouge2", "rougeL"):
            samples[k].append(sum(per_ex[i][k] for i in idx) / n)
        samples["bleu"].append(sacrebleu.corpus_bleu(
            [predictions[i] for i in idx], [[references[i] for i in idx]]).score)

    ci = {}
    for k, vals in samples.items():
        vals.sort()
        lo = vals[int(0.025 * n_bootstrap)]
        hi = vals[int(0.975 * n_bootstrap) - 1]
        ci[k] = [lo, hi]

    return {**point, "ci95": ci, "n_samples": n}


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

    results, dumps = {}, {}

    # Modele fine-tune
    adapter = args.adapter or cfg.train.output_dir
    ft_model, ft_tok = load_model(cfg, adapter, None)
    ft = evaluate_model(ft_model, ft_tok, test_data, "fine-tune (LoRA)")
    results["finetuned"], dumps["finetuned"] = ft["metrics"], ft["predictions"]

    # Modele de base (optionnel)
    if args.baseline:
        del ft_model
        torch.cuda.empty_cache()
        base_model, base_tok = load_baseline(cfg)
        base = evaluate_model(base_model, base_tok, test_data, "base (non fine-tune)")
        results["baseline"], dumps["baseline"] = base["metrics"], base["predictions"]

    # Rapport
    print("\n" + "=" * 66)
    print("RESULTATS  (IC 95 % par bootstrap, 1000 reechantillonnages)")
    print("=" * 66)
    for name, m in results.items():
        print(f"\n{name} (n={m['n_samples']}) :")
        for key, label, fmt in (("rouge1", "ROUGE-1", ".4f"),
                                ("rouge2", "ROUGE-2", ".4f"),
                                ("rougeL", "ROUGE-L", ".4f"),
                                ("bleu", "BLEU   ", ".2f")):
            lo, hi = m["ci95"][key]
            print(f"  {label} : {m[key]:{fmt}}  [IC95 {lo:{fmt}} - {hi:{fmt}}]")

    if "baseline" in results:
        print("\nGain du fine-tuning :")
        for key, label in (("rouge1", "ROUGE-1"), ("rougeL", "ROUGE-L"),
                           ("bleu", "BLEU")):
            ft_v, base_v = results["finetuned"][key], results["baseline"][key]
            ft_lo = results["finetuned"]["ci95"][key][0]
            base_hi = results["baseline"]["ci95"][key][1]
            # Heuristique lisible : les IC se recouvrent-ils ?
            verdict = "IC disjoints" if ft_lo > base_hi else "IC qui se recouvrent"
            print(f"  {label:8s} {base_v:.4f} -> {ft_v:.4f} "
                  f"({(ft_v-base_v)/base_v*100:+.0f} %)  [{verdict}]")

    out_dir = Path(cfg.train.output_dir)
    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # Predictions brutes : indispensables pour l'analyse qualitative des echecs
    with open(out_dir / "eval_predictions.json", "w", encoding="utf-8") as f:
        json.dump({
            "questions": [d["instruction"] for d in test_data],
            "references": [d["output"] for d in test_data],
            **dumps,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nMetriques ecrites dans   : {out_dir / 'eval_metrics.json'}")
    print(f"Predictions ecrites dans : {out_dir / 'eval_predictions.json'}")


if __name__ == "__main__":
    main()
