"""
Analyse qualitative des reponses du modele sur le jeu de test.

Un score agrege dit "combien", jamais "quoi" ni "pourquoi". Ce script exploite
`eval_predictions.json` pour repondre aux questions que les metriques masquent :
sur quels exemples le modele echoue-t-il, dans quelles categories, et selon
quel mode d'echec (reponse tronquee, hors-sujet, trop verbeuse, degeneree) ?

Ne necessite ni GPU ni modele : il relit les predictions deja produites.

Usage :
    python scripts/failure_analysis.py
    python scripts/failure_analysis.py --worst 5 --model finetuned
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rouge_score import rouge_scorer                    # noqa: E402
from src.config import Config                           # noqa: E402


def load_test_categories(cfg: Config) -> list[str]:
    path = Path(cfg.data.processed_dir) / "test.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line)["category"] for line in f]


def diagnose(pred: str, ref: str) -> list[str]:
    """Heuristiques simples de mode d'echec (indicatives, pas normatives)."""
    flags = []
    n_pred, n_ref = len(pred.split()), len(ref.split())

    if n_pred == 0:
        flags.append("vide")
    elif n_pred < 0.4 * n_ref:
        flags.append("trop courte")
    elif n_pred > 2.0 * n_ref:
        flags.append("trop verbeuse")

    # Troncature : pas de ponctuation finale (budget de tokens epuise)
    if pred and pred[-1] not in ".!?:»\"'`)":
        flags.append("tronquee")

    # Degenerescence : une phrase repetee a l'identique
    phrases = [s.strip() for s in re.split(r"[.!?]", pred) if len(s.strip()) > 25]
    if len(phrases) != len(set(phrases)):
        flags.append("repetition")

    return flags or ["-"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="finetuned",
                   help="cle du modele dans le fichier de predictions")
    p.add_argument("--worst", type=int, default=4,
                   help="nombre d'exemples les moins bons a afficher")
    p.add_argument("--file", default=None)
    args = p.parse_args()

    cfg = Config()
    pred_file = Path(args.file) if args.file else (
        Path(cfg.train.output_dir) / "eval_predictions.json")
    if not pred_file.exists():
        raise SystemExit(f"Introuvable : {pred_file}\n"
                         "Lancer d'abord : python evaluate.py --baseline")

    data = json.loads(pred_file.read_text(encoding="utf-8"))
    questions, refs = data["questions"], data["references"]
    preds = data[args.model]
    categories = load_test_categories(cfg)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows = []
    for i, (q, ref, pred) in enumerate(zip(questions, refs, preds)):
        rows.append({
            "i": i,
            "category": categories[i] if i < len(categories) else "?",
            "question": q,
            "reference": ref,
            "prediction": pred,
            "rougeL": scorer.score(ref, pred)["rougeL"].fmeasure,
            "len_ratio": len(pred.split()) / max(len(ref.split()), 1),
            "flags": diagnose(pred, ref),
        })

    print("=" * 78)
    print(f"ANALYSE QUALITATIVE — modele '{args.model}' ({len(rows)} exemples)")
    print("=" * 78)

    # --- Par categorie ---
    per_cat = defaultdict(list)
    for r in rows:
        per_cat[r["category"]].append(r["rougeL"])
    print("\nROUGE-L moyen par categorie :")
    for cat, vals in sorted(per_cat.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        avg = sum(vals) / len(vals)
        bar = "#" * int(avg * 60)
        print(f"  {cat:<20} {avg:.3f}  n={len(vals):<3} {bar}")

    # --- Modes d'echec ---
    flag_counts = defaultdict(int)
    for r in rows:
        for f in r["flags"]:
            flag_counts[f] += 1
    print("\nModes d'echec detectes :")
    for flag, n in sorted(flag_counts.items(), key=lambda kv: -kv[1]):
        label = "aucun probleme detecte" if flag == "-" else flag
        print(f"  {label:<24} {n:>3} / {len(rows)}")

    ratios = sorted(r["len_ratio"] for r in rows)
    med = ratios[len(ratios) // 2]
    print(f"\nRatio de longueur (prediction / reference) : mediane {med:.2f} "
          f"| min {ratios[0]:.2f} | max {ratios[-1]:.2f}")

    # --- Pires exemples ---
    rows.sort(key=lambda r: r["rougeL"])
    print(f"\n{'='*78}\n{args.worst} EXEMPLES LES MOINS BONS\n{'='*78}")
    for r in rows[:args.worst]:
        print(f"\n[{r['category']}] ROUGE-L={r['rougeL']:.3f} "
              f"| flags: {', '.join(r['flags'])}")
        print(f"  Q   : {r['question'][:110]}")
        print(f"  REF : {r['reference'][:190]}...")
        print(f"  GEN : {r['prediction'][:190]}...")

    # --- Meilleur exemple, pour contraste ---
    best = rows[-1]
    print(f"\n{'='*78}\nMEILLEUR EXEMPLE\n{'='*78}")
    print(f"\n[{best['category']}] ROUGE-L={best['rougeL']:.3f}")
    print(f"  Q   : {best['question'][:110]}")
    print(f"  GEN : {best['prediction'][:260]}...")

    out = Path(cfg.train.output_dir) / f"failure_analysis_{args.model}.json"
    out.write_text(json.dumps({
        "model": args.model,
        "per_category_rougeL": {k: sum(v)/len(v) for k, v in per_cat.items()},
        "failure_modes": dict(flag_counts),
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEcrit : {out}")


if __name__ == "__main__":
    main()
