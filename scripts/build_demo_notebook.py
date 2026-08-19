"""
Construit (et exporte) le notebook de demo : notebooks/demo.ipynb

Utilise nbformat pour assembler les cellules, ce qui evite les erreurs
d'echappement d'un .ipynb ecrit a la main. L'execution (pour embarquer les
sorties) est faite separement via nbconvert --execute.

Usage : python scripts/build_demo_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "demo.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(src: str):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------- Titre
md(r"""
# 🤖 Démo — Assistant Python / Data Science / ML (Qwen2.5-3B fine-tuné QLoRA)

Ce notebook présente, de bout en bout, un **LLM fine-tuné en local** sur une
**RTX 4070 Laptop (8 Go)** avec **QLoRA** (LoRA + quantization 4-bit) :

1. Configuration & environnement
2. Le dataset (domaine Python/DS/ML)
3. Courbe d'apprentissage
4. Gain mesuré du fine-tuning (ROUGE / BLEU vs modèle de base)
5. Démonstration en direct

> Modèle de base : `Qwen/Qwen2.5-3B-Instruct` · Méthode : QLoRA (rang 16, 4-bit NF4)
""")

# ---------------------------------------------------------------- 1. Config
md("## 1. Configuration & environnement")
code(r"""
import sys, os, json
from pathlib import Path

# Rendre le projet importable, quel que soit le repertoire d'execution
ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import torch
from src.config import Config

cfg = Config()
print("Modele de base :", cfg.model.model_name)
print("Quantization   :", "4-bit " + cfg.model.bnb_4bit_quant_type if cfg.model.use_4bit else "aucune")
print("LoRA           : rang", cfg.lora.r, "| alpha", cfg.lora.alpha,
      "|", len(cfg.lora.target_modules), "modules cibles")
print("GPU            :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
""")

# ---------------------------------------------------------------- 2. Dataset
md("""
## 2. Le dataset

Un corpus **curé** de paires question/réponse expertes sur Python, NumPy/Pandas,
le ML, le deep learning, les LLMs, l'évaluation et le MLOps.
""")
code(r"""
import matplotlib.pyplot as plt

raw = json.load(open(ROOT / "data" / "raw" / "raw_qa_data.json", encoding="utf-8"))
print(f"{len(raw)} exemples au total\n")

# Repartition par categorie
from collections import Counter
cats = Counter(item["category"] for item in raw)

fig, ax = plt.subplots(figsize=(8, 3.5))
names = [c for c, _ in cats.most_common()]
vals = [cats[c] for c in names]
ax.barh(names[::-1], vals[::-1], color="#4C72B0")
ax.set_xlabel("Nombre d'exemples")
ax.set_title("Répartition du dataset par catégorie")
for i, v in enumerate(vals[::-1]):
    ax.text(v + 0.3, i, str(v), va="center", fontsize=9)
plt.tight_layout(); plt.show()

# Un exemple
ex = raw[0]
print("Exemple —", ex["category"])
print("Q:", ex["instruction"])
print("R:", ex["output"][:280], "...")
""")

# ---------------------------------------------------------------- 3. Loss
md("""
## 3. Courbe d'apprentissage

Loss d'entraînement et de validation au fil des steps (lue depuis l'état du
`Trainer`).

La loss d'entraînement descend continûment, mais celle de **validation atteint
son minimum vers 1 epoch puis remonte** : au-delà, le modèle sur-apprend. C'est
`load_best_model_at_end` qui conserve le meilleur checkpoint, complété par un
early stopping.

> 💡 Ce diagnostic n'était **pas visible** dans la v1 du projet : une fuite de
> données faisait décroître la validation artificiellement. Voir la section
> « Correction méthodologique » du README.
""")
code(r"""
def load_log_history():
    # Cherche le trainer_state.json le plus avance (checkpoints ignores par git,
    # d'ou un repli sur les valeurs du run documente si absent).
    states = sorted((ROOT / "outputs").glob("**/trainer_state.json"),
                    key=lambda p: p.stat().st_mtime) if (ROOT / "outputs").exists() else []
    if states:
        return json.load(open(states[-1]))["log_history"]
    # Repli : valeurs reelles du run v2 (dataset sans fuite)
    return [
        {"step": 5, "loss": 2.669}, {"step": 8, "eval_loss": 2.230},
        {"step": 10, "loss": 2.149}, {"step": 15, "loss": 1.927},
        {"step": 16, "eval_loss": 2.114}, {"step": 20, "loss": 1.669},
        {"step": 24, "eval_loss": 2.194}, {"step": 25, "loss": 1.461},
        {"step": 30, "loss": 1.332}, {"step": 32, "eval_loss": 2.291},
        {"step": 35, "loss": 1.114}, {"step": 40, "loss": 0.912,
                                      "eval_loss": 2.449},
        {"step": 45, "loss": 1.001}, {"step": 48, "eval_loss": 2.433},
    ]

hist = load_log_history()
tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(*zip(*tr), "o-", label="train loss", color="#4C72B0")
ax.plot(*zip(*ev), "s-", label="eval loss", color="#C44E52")
ax.set_xlabel("Step"); ax.set_ylabel("Loss")
ax.set_title("Courbe d'apprentissage (QLoRA, 3 epochs)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"eval loss : {ev[0][1]:.3f} -> {ev[-1][1]:.3f}")
""")

# ---------------------------------------------------------------- 4. Eval
md("""
## 4. Gain mesuré du fine-tuning

Comparaison sur le jeu de test — **strictement disjoint** de l'entraînement —
entre le modèle de base et le modèle fine-tuné.

Les barres d'erreur sont les **intervalles de confiance à 95 %** obtenus par
bootstrap. Sur un jeu de test de 24 exemples, un écart sans mesure d'incertitude
ne prouve rien : ce qui compte est que les intervalles **ne se recouvrent pas**.
""")
code(r"""
metrics = json.load(open(ROOT / "outputs" / "qwen2.5-3b-pyds-lora" / "eval_metrics.json"))
base, ft = metrics["baseline"], metrics["finetuned"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

# --- ROUGE (echelle 0-1) ---
labels, keys = ["ROUGE-1", "ROUGE-2", "ROUGE-L"], ["rouge1", "rouge2", "rougeL"]
x, w = range(len(labels)), 0.35
for offset, m, lab, col in ((-w/2, base, "base", "#B0B0B0"),
                            (w/2, ft, "fine-tuné", "#4C72B0")):
    a1.bar([i + offset for i in x], [m[k] for k in keys], w, label=lab, color=col,
           yerr=[[m[k] - m["ci95"][k][0] for k in keys],
                 [m["ci95"][k][1] - m[k] for k in keys]],
           capsize=4, ecolor="#333333")
a1.set_xticks(list(x)); a1.set_xticklabels(labels)
a1.set_title("ROUGE (IC 95 %)"); a1.legend()

# --- BLEU (echelle 0-100) ---
a2.bar(["base", "fine-tuné"], [base["bleu"], ft["bleu"]],
       color=["#B0B0B0", "#55A868"], capsize=5, ecolor="#333333",
       yerr=[[base["bleu"] - base["ci95"]["bleu"][0], ft["bleu"] - ft["ci95"]["bleu"][0]],
             [base["ci95"]["bleu"][1] - base["bleu"], ft["ci95"]["bleu"][1] - ft["bleu"]]])
a2.set_title("BLEU (IC 95 %)")
plt.tight_layout(); plt.show()

print(f"n = {ft['n_samples']} exemples de test\n")
for k, lab in (("rouge1", "ROUGE-1"), ("rougeL", "ROUGE-L"), ("bleu", "BLEU   ")):
    disjoint = ft["ci95"][k][0] > base["ci95"][k][1]
    print(f"{lab} : {base[k]:.3f} -> {ft[k]:.3f} "
          f"({(ft[k]-base[k])/base[k]*100:+.0f} %)  "
          f"{'IC disjoints' if disjoint else 'IC qui se recouvrent'}")
""")

# ---------------------------------------------------------------- 5. Demo live
md("""
## 5. Démonstration en direct

On charge le modèle de base **quantisé en 4-bit** + l'**adaptateur LoRA**, puis
on génère des réponses (température 0 pour la reproductibilité).
""")
code(r"""
from inference import load_model, generate

model, tokenizer = load_model(cfg, adapter_path=cfg.train.output_dir, merged_path=None)

questions = [
    "Qu'est-ce que la vectorisation en Python ?",
    "Explique LoRA en deux phrases.",
    "Quelle est la différence entre .loc et .iloc dans Pandas ?",
]
for q in questions:
    print("=" * 78)
    print("Q :", q)
    print("-" * 78)
    print(generate(model, tokenizer, q, max_new_tokens=220, temperature=0.0))
    print()
""")

# ---------------------------------------------------------------- Conclusion
md(r"""
## Conclusion

En **~7 min** d'entraînement sur un GPU grand public (RTX 4070, 8 Go), le
fine-tuning QLoRA a :

- **amélioré les 4 métriques**, avec des **intervalles de confiance disjoints** —
  le gain n'est pas un artefact d'échantillonnage (ROUGE-1 **+38 %**,
  BLEU **+132 %**) ;
- adapté le **style et le format** des réponses au domaine cible ;
- produit un **adaptateur de 119 Mo** (vs 6,2 Go pour le modèle complet), en
  n'entraînant que **0,96 %** des paramètres.

### Ce que ce projet montre aussi

Une première version souffrait d'une **fuite de données** : l'augmentation par
paraphrase était appliquée *avant* le découpage, si bien que les réponses de
référence du test avaient déjà été vues à l'entraînement. Elle gonflait les
scores **et masquait un sur-apprentissage**. Le correctif — split par groupe,
stratifié, augmentation réservée au train, contrôle bloquant — est documenté
dans le README.

> Les chiffres ci-dessus sont ceux du protocole **corrigé**. Ils sont plus bas
> en valeur absolue que ceux de la v1, et c'est précisément ce qui les rend
> dignes de confiance.

➡️ Code complet : [`train.py`](../train.py) · [`prepare_dataset.py`](../prepare_dataset.py) ·
[`evaluate.py`](../evaluate.py) · [`README.md`](../README.md)
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print("Notebook ecrit :", OUT, f"({len(cells)} cellules)")
