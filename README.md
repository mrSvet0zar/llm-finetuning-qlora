# 🤖 Fine-Tuning d'un LLM — Assistant expert Python / Data Science / ML

Fine-tuning **QLoRA** de **Qwen2.5-3B-Instruct** sur un dataset spécialisé
Python / Data Science / Machine Learning, entraîné **localement sur une seule
carte grand public (RTX 4070 Laptop, 8 Go de VRAM)**, avec pipeline complet :
préparation des données → entraînement → évaluation → fusion → déploiement (API
FastAPI + Ollama).

> Projet portfolio démontrant la maîtrise du **transfer learning**, du
> **parameter-efficient fine-tuning (PEFT/LoRA)**, de la **quantization 4-bit**,
> et du cycle MLOps complet d'un modèle de langage.

> 📓 **Démo rapide** : [`notebooks/demo.ipynb`](notebooks/demo.ipynb) — parcours
> end-to-end avec graphiques et réponses du modèle déjà exécutés (visibles
> directement sur GitHub, sans rien lancer).

---

## 🎯 En bref

| | |
|---|---|
| **Modèle de base** | `Qwen/Qwen2.5-3B-Instruct` |
| **Méthode** | QLoRA (LoRA + quantization 4-bit NF4) |
| **Domaine** | Python, NumPy/Pandas, ML, Deep Learning, LLMs, MLOps |
| **Matériel** | RTX 4070 Laptop — 8 Go VRAM (Windows) |
| **Framework** | PyTorch + 🤗 Transformers / PEFT / bitsandbytes |
| **Suivi** | TensorBoard |
| **Déploiement** | FastAPI · Ollama (GGUF) · Hugging Face Hub |

### 📊 Résultats

Évaluation sur le jeu de test (8 exemples tenus à l'écart), fine-tuné **vs**
modèle de base non spécialisé (`python evaluate.py --baseline`) :

| Métrique | Modèle de base | Fine-tuné | Gain |
|---|---|---|---|
| ROUGE-1 | 0.277 | **0.349** | **+26 %** |
| ROUGE-2 | 0.039 | **0.058** | **+51 %** |
| ROUGE-L | 0.135 | **0.186** | **+37 %** |
| BLEU | 4.29 | **10.08** | **+135 %** |

**Courbe d'apprentissage** (loss de validation) : 2.19 → 1.90 → 1.74 → 1.68 —
décroissance monotone, **sans overfitting**. Entraînement en **3 min 25 s** sur
la RTX 4070 (~7 Go / 8 Go de VRAM utilisés).

> Le fine-tuning améliore les 4 métriques ; le BLEU plus que double. Le modèle
> adopte le style, le format et le niveau de détail du corpus cible, tout en
> conservant les connaissances du modèle de base.

**Exemple de génération** (modèle fine-tuné, servi via l'API) :

> **Q :** Qu'est-ce que le broadcasting en NumPy ?
> **R :** _Le broadcasting est une fonctionnalité qui permet d'opérer sur deux
> tableaux de format différent sans les réduire à la taille minimale. Les
> conditions sont : dimensions identiques ou 1 dans l'une des deux […]_

### ✅ État du pipeline (validé end-to-end)

| Étape | Statut |
|---|---|
| Génération + préparation du dataset | ✅ 129 exemples, split 109/12/8 |
| Entraînement QLoRA (RTX 4070) | ✅ 3 min 25 s, loss val 2.19 → 1.68 |
| Évaluation ROUGE/BLEU + baseline | ✅ gains sur les 4 métriques |
| Inférence (base 4-bit + adaptateur) | ✅ |
| Fusion LoRA → modèle autonome | ✅ `merge_model.py` |
| Serveur API FastAPI (`/generate`) | ✅ testé (latence ~12 s / 120 tok) |
| Export GGUF + Ollama | 📋 documenté (nécessite llama.cpp) |
| Publication Hugging Face Hub | 📋 documenté (nécessite un token HF) |

> **Note perf** : la latence de `model.generate` non-batché (~12 s) dépasse la
> cible < 2 s du cahier des charges. En production, on passerait par **vLLM**
> (batching continu, PagedAttention) ou le modèle fusionné en fp16 — voir pistes.

---

## 🧠 Choix techniques & adaptations

Le spécification initiale (`CLAUDE.md`) visait **Mistral-7B sur Google Colab**.
Ce projet a été **adapté au matériel local** (RTX 4070, 8 Go) et à Windows, avec
des choix documentés qui reflètent des décisions d'ingénieur réelles :

| Décision | Choix | Justification |
|---|---|---|
| **Modèle** | Qwen2.5-3B au lieu de Mistral-7B | Un 7B en 4-bit sature 8 Go (OOM). Un 3B tient confortablement et Qwen2.5 est plus récent/performant. |
| **Format de prompt** | Chat template ChatML | Qwen utilise ChatML (`<|im_start|>`), pas les balises `[INST]` de Mistral. Utiliser le mauvais format dégrade fortement la qualité. |
| **Attention** | SDPA | `flash_attention_2` ne se compile pas sous Windows ; le SDPA de PyTorch est natif et performant. |
| **Calcul de la loss** | Masquage du prompt | On n'entraîne le modèle **que sur la réponse** (labels `-100` sur le prompt), pas à recopier la question. Meilleure pratique en instruction-tuning. |
| **LoRA target modules** | Toutes projections attn + MLP | Meilleur transfert que `q/v` seuls, pour un surcoût mémoire négligeable en QLoRA. |
| **Mémoire** | 4-bit NF4 + double quant + grad checkpointing + optim paginé | Combinaison qui fait tenir l'entraînement d'un 3B dans 8 Go. |
| **Batch** | 1 × 16 (grad accumulation) | Batch effectif de 16 sans dépasser la VRAM. |
| **Tracking** | TensorBoard | Zéro friction de login (vs Wandb), suffisant pour ce projet. |

### Pourquoi QLoRA ?

Le fine-tuning complet d'un modèle de 3 milliards de paramètres en pleine
précision exigerait ~40-60 Go de VRAM (poids + gradients + états d'optimiseur).
**QLoRA** rend l'opération possible sur 8 Go :

1. Le modèle de base est chargé **quantisé en 4-bit** (NF4) et **gelé** — il ne
   sert que de « socle de connaissances ».
2. On entraîne uniquement de petits **adaptateurs LoRA** (rang 16) insérés dans
   les couches d'attention et MLP — soit **< 1 % des paramètres**.
3. La **double quantization** et l'**optimiseur paginé 8-bit** réduisent encore
   les pics mémoire.

Résultat : un adaptateur de quelques dizaines de Mo, entraîné en local, qui
spécialise le modèle sans toucher à ses poids d'origine.

---

## 📁 Structure du projet

```
finetuning/
├── src/
│   └── config.py            # Configuration centrale (dataclasses)
├── scripts/
│   ├── generate_dataset.py  # Génère le corpus Q&A curé (Python/DS/ML)
│   ├── smoke_test.py        # Test du chat template + masquage (sans GPU lourd)
│   └── test_api.py          # Test de fumée de l'API FastAPI
├── data/
│   ├── raw/                 # raw_qa_data.json (corpus brut)
│   └── processed/           # train/val/test.jsonl
├── notebooks/
│   └── demo.ipynb           # Démo end-to-end (sorties + graphiques embarqués)
├── prepare_dataset.py       # Validation → dédup → split → JSONL
├── train.py                 # Fine-tuning QLoRA (cœur du projet)
├── inference.py             # Génération avec le modèle fine-tuné
├── evaluate.py              # Métriques ROUGE/BLEU (+ comparaison baseline)
├── merge_model.py           # Fusion adaptateur LoRA → modèle autonome
├── api_server.py            # Serveur d'inférence FastAPI
├── Modelfile                # Déploiement Ollama (GGUF)
├── setup.ps1                # Installation environnement (Windows)
├── requirements.txt
└── README.md
```

---

## 🚀 Démarrage rapide

### 1. Installation

```powershell
.\setup.ps1
```

Ce script crée un venv, installe **PyTorch (CUDA 12.4)** puis les dépendances,
et vérifie que le GPU est bien détecté. (Installation manuelle : voir
`requirements.txt`.)

### 2. Données

```bash
python scripts/generate_dataset.py   # corpus curé → data/raw/raw_qa_data.json
python prepare_dataset.py            # validation, dédup, split → data/processed/
```

### 3. Entraînement

```bash
python train.py
```

Suivi en temps réel :

```bash
tensorboard --logdir outputs/qwen2.5-3b-pyds-lora/logs
```

### 4. Inférence

```bash
python inference.py --prompt "Explique la différence entre LoRA et QLoRA"
```

### 5. Évaluation

```bash
python evaluate.py --baseline        # compare fine-tuné vs modèle de base
```

---

## 🔬 Le pipeline en détail

### Phase 1 — Données
Un corpus **curé à la main** de paires Q&A expertes (Python, NumPy/Pandas, ML,
deep learning, LLMs, évaluation, MLOps, data engineering), enrichi par une
**augmentation légère et transparente** (reformulations de questions partageant
la même réponse de référence). Le pipeline de préparation valide (champs requis,
longueurs minimales), **déduplique** (sur l'instruction normalisée) et
**découpe** en train/val/test de façon déterministe (seed fixe).

> **Note d'honnêteté** : le corpus est volontairement compact (~40 paires curées,
> ~130 après augmentation) pour une démo locale reproductible. Le `CLAUDE.md`
> visait 500-2000 paires ; le pipeline est conçu pour **monter en volume
> trivialement** en ajoutant des entrées à `scripts/generate_dataset.py`.

### Phase 2 — Entraînement
QLoRA avec masquage du prompt. La séquence est construite via le chat template
ChatML, puis les tokens du système + de la question sont masqués (`-100`) pour
que **seule la réponse contribue au gradient**. Early-stopping implicite via
`load_best_model_at_end` sur la loss de validation.

### Phase 3 — Évaluation
Métriques lexicales **ROUGE-1/2/L** et **BLEU** sur le jeu de test, avec
comparaison optionnelle au modèle de base pour **quantifier le gain** du
fine-tuning.

> ⚠️ **Limites des métriques** : BLEU et ROUGE sont purement lexicaux et ne
> captent pas la sémantique (une bonne paraphrase peut scorer bas). Elles servent
> de **proxy automatique bon marché** ; une évaluation rigoureuse ajouterait
> BERTScore, un LLM-juge, ou une revue humaine.

### Phase 4 — Fusion & déploiement
`merge_model.py` fusionne l'adaptateur dans le modèle de base (checkpoint
autonome), qui peut ensuite être :
- servi via **FastAPI** (`api_server.py`, endpoint `/generate`),
- converti en **GGUF** et déployé via **Ollama** (`Modelfile`),
- publié sur le **Hugging Face Hub**.

#### Conversion GGUF (pour Ollama)
```bash
git clone https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py outputs/merged-model --outfile outputs/merged-model-gguf/model-f16.gguf
llama.cpp/llama-quantize outputs/merged-model-gguf/model-f16.gguf outputs/merged-model-gguf/model-q4_k_m.gguf Q4_K_M
ollama create qwen-pyds -f Modelfile
```

---

## ⏱️ Temps & empreinte

| | |
|---|---|
| VRAM à l'entraînement | ~7 Go / 8 Go |
| Paramètres entraînés | 29,9 M / 3,12 Md (**0,96 %**) |
| Taille de l'adaptateur | ~119 Mo (vs 6,2 Go pour le modèle fusionné) |
| Durée d'entraînement | 3 min 25 s (3 epochs) |

---

## 🔭 Pistes d'amélioration

- [ ] Étendre le corpus à 1000+ paires (couverture de domaine plus large)
- [ ] Fine-tuning multi-tours (conversations, pas seulement Q&A)
- [ ] Évaluation sémantique (BERTScore, LLM-as-a-judge)
- [ ] Servir avec vLLM (batching continu, meilleur débit)
- [ ] Publier l'adaptateur sur le Hugging Face Hub
- [ ] Dockeriser l'API

---

## 📚 Stack

`PyTorch` · `Transformers` · `PEFT` · `bitsandbytes` · `Accelerate` ·
`Datasets` · `TensorBoard` · `FastAPI` · `Ollama` · `rouge-score` · `sacrebleu`
