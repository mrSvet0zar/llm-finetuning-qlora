"""
Baselines alternatives au fine-tuning, sur le MEME jeu de test.

Question posee : le fine-tuning etait-il seulement necessaire ? Comparer un
modele fine-tune a son seul modele de base repond a "est-ce que ca marche",
pas a "est-ce le bon outil". On confronte donc quatre approches :

  1. zero-shot   : modele de base, prompt nu               (cout nul)
  2. few-shot    : modele de base + 3 exemples dans le prompt   (cout nul,
                   mais rallonge chaque requete)
  3. rag         : modele de base + 3 passages recuperes du corpus
                   d'entrainement par similarite TF-IDF    (cout : un index)
  4. finetuned   : le modele QLoRA                          (cout : entrainement)

Toutes utilisent exactement la meme procedure de decodage (greedy) et le meme
jeu de test, seule la construction du prompt change.

Equite de la comparaison
------------------------
Les quatre approches disposent EXACTEMENT de la meme connaissance : les 85
concepts du jeu d'entrainement. Le fine-tuning les a absorbes dans ses poids,
le RAG y accede par recherche, le few-shot en montre trois. Seul le MECANISME
differe, ce qui est precisement ce que l'on veut comparer.

Consequence du split par groupe : aucun passage du train ne contient la reponse
a une question de test. Le RAG ne peut donc pas "tricher" en retrouvant la
reponse attendue ; il fournit du contexte VOISIN. C'est la comparaison honnete,
mais elle sous-estime ce que donnerait un RAG en production, ou la base
documentaire couvrirait reellement les questions posees.

Note sur le retrieveur : TF-IDF plutot qu'un modele d'embeddings, pour rester
leger et sans telechargement supplementaire. Un vrai systeme utiliserait des
embeddings denses.

Usage :
    python scripts/baselines.py                # les 3 baselines de base
    python scripts/baselines.py --include-ft   # + le modele fine-tune
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import load_baseline  # noqa: E402
from inference import generate_from_messages, load_model  # noqa: E402
from src.config import Config  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.retrieval import TfidfRetriever  # noqa: E402
from train import SYSTEM_PROMPT  # noqa: E402

N_SHOTS = 3
N_RETRIEVED = 3
MAX_NEW_TOKENS = 320


def load_split(cfg: Config, name: str) -> list[dict]:
    path = Path(cfg.data.processed_dir) / f"{name}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
#  Construction des prompts
# ---------------------------------------------------------------------------
def messages_zero_shot(question: str, _train, _retriever) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def messages_few_shot(question: str, shots: list[dict], _retriever) -> list[dict]:
    """Exemples presentes comme de vrais tours de conversation anterieurs."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in shots:
        msgs.append({"role": "user", "content": ex["instruction"]})
        msgs.append({"role": "assistant", "content": ex["output"]})
    msgs.append({"role": "user", "content": question})
    return msgs


def messages_rag(question: str, _shots, retriever) -> list[dict]:
    passages = retriever.query(question, k=N_RETRIEVED)
    context = "\n\n".join(
        f"[Extrait {i}] {p['instruction']}\n{p['output']}"
        for i, p in enumerate(passages, 1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            "Reponds a la question en t'appuyant sur les extraits de "
            "documentation ci-dessous.\n\n"
            f"{context}\n\n"
            f"Question : {question}"},
    ]


# ---------------------------------------------------------------------------
def run_strategy(model, tokenizer, test_data, builder, shots, retriever,
                 label: str) -> dict:
    preds, refs = [], []
    print(f"\n--- {label} ---")
    for i, item in enumerate(test_data, 1):
        msgs = builder(item["instruction"], shots, retriever)
        preds.append(generate_from_messages(
            model, tokenizer, msgs,
            max_new_tokens=MAX_NEW_TOKENS, temperature=0.0))
        refs.append(item["output"])
        if i % 8 == 0 or i == len(test_data):
            print(f"    {i}/{len(test_data)}")
    return {"metrics": compute_metrics(preds, refs), "predictions": preds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-ft", action="store_true",
                        help="evalue aussi le modele fine-tune (comparaison)")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    train_data = load_split(cfg, "train")
    test_data = load_split(cfg, "test")
    if args.n:
        test_data = test_data[:args.n]

    # Exemples few-shot : pris dans le TRAIN, de categories variees, fixes
    # (deterministe). Ils ne doivent evidemment jamais venir du test.
    seen_cats, shots = set(), []
    for item in sorted(train_data, key=lambda d: d["group_id"]):
        if item["category"] not in seen_cats and item.get("variant") != "paraphrase":
            shots.append(item)
            seen_cats.add(item["category"])
        if len(shots) == N_SHOTS:
            break

    retriever = TfidfRetriever(train_data)

    print(f"Test : {len(test_data)} exemples | few-shot : {len(shots)} exemples "
          f"| RAG : top-{N_RETRIEVED} sur {len(train_data)} passages")

    results, dumps = {}, {}

    # --- Modele de base : les 3 strategies de prompting ---
    base_model, base_tok = load_baseline(cfg)
    for key, builder, label in (
        ("zero_shot", messages_zero_shot, "base / zero-shot"),
        ("few_shot", messages_few_shot, f"base / few-shot ({N_SHOTS} exemples)"),
        ("rag", messages_rag, f"base / RAG (top-{N_RETRIEVED} TF-IDF)"),
    ):
        out = run_strategy(base_model, base_tok, test_data, builder,
                           shots, retriever, label)
        results[key], dumps[key] = out["metrics"], out["predictions"]

    # --- Modele fine-tune (optionnel) ---
    if args.include_ft:
        del base_model
        torch.cuda.empty_cache()
        ft_model, ft_tok = load_model(cfg, cfg.train.output_dir, None)
        out = run_strategy(ft_model, ft_tok, test_data, messages_zero_shot,
                           shots, retriever, "fine-tune (LoRA)")
        results["finetuned"], dumps["finetuned"] = out["metrics"], out["predictions"]

    # --- Rapport ---
    print("\n" + "=" * 74)
    print("COMPARAISON DES APPROCHES  (IC 95 % bootstrap)")
    print("=" * 74)
    print(f"{'approche':<14}{'ROUGE-1':>22}{'ROUGE-L':>22}{'BLEU':>16}")
    for key, m in results.items():
        r1 = f"{m['rouge1']:.3f} [{m['ci95']['rouge1'][0]:.3f}-{m['ci95']['rouge1'][1]:.3f}]"
        rl = f"{m['rougeL']:.3f} [{m['ci95']['rougeL'][0]:.3f}-{m['ci95']['rougeL'][1]:.3f}]"
        bl = f"{m['bleu']:.2f} [{m['ci95']['bleu'][0]:.1f}-{m['ci95']['bleu'][1]:.1f}]"
        print(f"{key:<14}{r1:>22}{rl:>22}{bl:>16}")

    out_dir = Path(cfg.train.output_dir)
    (out_dir / "baselines_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "baselines_predictions.json").write_text(
        json.dumps({"questions": [d["instruction"] for d in test_data],
                    "references": [d["output"] for d in test_data], **dumps},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEcrit : {out_dir / 'baselines_metrics.json'}")


if __name__ == "__main__":
    main()
