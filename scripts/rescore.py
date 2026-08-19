"""
Recalcule les metriques a partir de predictions DEJA generees.

Interet : changer de protocole de scoring (ici, la normalisation des accents)
sans relancer la generation, qui est la partie couteuse. C'est aussi la raison
pour laquelle `evaluate.py` sauvegarde systematiquement les predictions brutes.

Affiche la comparaison AVEC et SANS normalisation des accents, afin de rendre
l'artefact visible plutot que de le corriger silencieusement.

Usage :
    python scripts/rescore.py
    python scripts/rescore.py --file outputs/.../baselines_predictions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402

NON_MODEL_KEYS = {"questions", "references"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=None)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()

    cfg = Config()
    path = Path(args.file) if args.file else (
        Path(cfg.train.output_dir) / "baselines_predictions.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    refs = data["references"]
    models = [k for k in data if k not in NON_MODEL_KEYS]

    print(f"Fichier : {path.name}  ({len(refs)} exemples)\n")

    results = {}
    print(f"{'approche':<12}{'ROUGE-1 brut':>14}{'ROUGE-1 norm.':>15}"
          f"{'BLEU brut':>11}{'BLEU norm.':>12}")
    for name in models:
        raw = compute_metrics(data[name], refs, n_bootstrap=args.bootstrap,
                              normalize_accents=False)
        norm = compute_metrics(data[name], refs, n_bootstrap=args.bootstrap,
                               normalize_accents=True)
        results[name] = {"raw": raw, "normalized": norm}
        print(f"{name:<12}{raw['rouge1']:>14.4f}{norm['rouge1']:>15.4f}"
              f"{raw['bleu']:>11.2f}{norm['bleu']:>12.2f}")

    # --- Tableau de reference : metriques normalisees avec IC ---
    print(f"\n{'='*74}")
    print("METRIQUES RETENUES (accents normalises, IC 95 % bootstrap)")
    print("=" * 74)
    print(f"{'approche':<12}{'ROUGE-1':>22}{'ROUGE-L':>22}{'BLEU':>16}")
    for name, r in results.items():
        m = r["normalized"]
        r1 = f"{m['rouge1']:.3f} [{m['ci95']['rouge1'][0]:.3f}-{m['ci95']['rouge1'][1]:.3f}]"
        rl = f"{m['rougeL']:.3f} [{m['ci95']['rougeL'][0]:.3f}-{m['ci95']['rougeL'][1]:.3f}]"
        bl = f"{m['bleu']:.2f} [{m['ci95']['bleu'][0]:.1f}-{m['ci95']['bleu'][1]:.1f}]"
        print(f"{name:<12}{r1:>22}{rl:>22}{bl:>16}")

    out = path.parent / f"rescored_{path.stem}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nEcrit : {out}")


if __name__ == "__main__":
    main()
