"""
Comparaison de plusieurs modeles sur le MEME jeu de test.

Repond a la question centrale du projet : la limite observee vient-elle de la
methode d'adaptation, ou du modele de base ? Pour y repondre proprement, une
seule variable change entre les deux colonnes — la taille du modele. Corpus,
decoupage, hyperparametres, protocole d'evaluation et jeu de test sont
identiques.

Lit les `eval_metrics.json` produits par `evaluate.py` et affiche :
  * les scores de chaque systeme avec leurs IC 95 % ;
  * le gain du fine-tuning DANS chaque famille (3B seul, 7B seul) ;
  * si ces gains sont statistiquement etablis (IC disjoints).

Usage :
    python scripts/compare_models.py
    python scripts/compare_models.py --runs outputs/qwen2.5-3b-pyds-lora outputs/qwen2.5-7b-pyds-lora
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import intervals_disjoint  # noqa: E402

METRIQUES = [("rouge1", "ROUGE-1", 3), ("rougeL", "ROUGE-L", 3),
             ("bleu", "BLEU", 2)]


def charger(run_dir: Path) -> dict | None:
    f = run_dir / "eval_metrics.json"
    if not f.exists():
        print(f"  (ignore : {f} absent)")
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def fmt(m: dict, cle: str, dec: int) -> str:
    lo, hi = m["ci95"][cle]
    return f"{m[cle]:.{dec}f} [{lo:.{dec}f}-{hi:.{dec}f}]"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=[
        "outputs/qwen2.5-3b-pyds-lora", "outputs/qwen2.5-7b-pyds-lora"])
    p.add_argument("--labels", nargs="+", default=None)
    args = p.parse_args()

    runs = [PROJECT_ROOT / r for r in args.runs]
    labels = args.labels or [r.name.replace("qwen2.5-", "").replace("-pyds-lora", "")
                             for r in runs]

    donnees = {}
    for label, run in zip(labels, runs, strict=True):
        d = charger(run)
        if d:
            donnees[label] = d

    if not donnees:
        raise SystemExit("Aucun resultat exploitable. Lancer d'abord evaluate.py")

    # --- Tableau principal ---
    largeur = 24
    print("=" * (18 + largeur * len(METRIQUES)))
    print("COMPARAISON DES MODELES  (meme corpus, meme protocole, IC 95 %)")
    print("=" * (18 + largeur * len(METRIQUES)))
    entete = f"{'systeme':<18}" + "".join(f"{nom:>{largeur}}" for _, nom, _ in METRIQUES)
    print(entete)
    print("-" * len(entete))

    for label, d in donnees.items():
        for variante, suffixe in (("baseline", "base"), ("finetuned", "fine-tune")):
            if variante not in d:
                continue
            nom = f"{label} {suffixe}"
            ligne = f"{nom:<18}"
            for cle, _, dec in METRIQUES:
                ligne += f"{fmt(d[variante], cle, dec):>{largeur}}"
            print(ligne)

    # --- Gain du fine-tuning dans chaque famille ---
    print("\n" + "=" * 66)
    print("GAIN DU FINE-TUNING, PAR FAMILLE")
    print("=" * 66)
    for label, d in donnees.items():
        if "baseline" not in d or "finetuned" not in d:
            continue
        print(f"\n{label} :")
        for cle, nom, dec in METRIQUES:
            base, ft = d["baseline"][cle], d["finetuned"][cle]
            gain = (ft - base) / base * 100 if base else 0.0
            etabli = intervals_disjoint(d["baseline"], d["finetuned"], cle)
            verdict = "etabli" if etabli else "NON etabli (IC se recouvrent)"
            print(f"  {nom:<8} {base:.{dec}f} -> {ft:.{dec}f}  ({gain:+.0f} %)  {verdict}")

    # --- Lecture croisee : le modele de base est-il le facteur limitant ? ---
    if len(donnees) >= 2:
        petit, grand = list(donnees)[0], list(donnees)[-1]
        d_p, d_g = donnees[petit], donnees[grand]
        if "baseline" in d_p and "baseline" in d_g:
            print("\n" + "=" * 66)
            print("LE MODELE DE BASE EST-IL LE FACTEUR LIMITANT ?")
            print("=" * 66)
            for cle, nom, dec in METRIQUES:
                ecart_base = d_g["baseline"][cle] - d_p["baseline"][cle]
                ft_p = d_p.get("finetuned", {}).get(cle)
                gain_ft = (ft_p - d_p["baseline"][cle]) if ft_p is not None else None
                print(f"\n  {nom}")
                print(f"    passer de {petit} a {grand} (sans fine-tuning) : {ecart_base:+.{dec}f}")
                if gain_ft is not None:
                    print(f"    fine-tuner le {petit}                        : {gain_ft:+.{dec}f}")
                    if abs(ecart_base) > abs(gain_ft):
                        print("    -> changer de modele de base pese PLUS que le fine-tuning")
                    else:
                        print("    -> le fine-tuning pese plus que le changement de modele")


if __name__ == "__main__":
    main()
