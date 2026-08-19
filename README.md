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

Évaluation sur un jeu de test **strictement disjoint** (24 concepts, aucune
réponse partagée avec l'entraînement), fine-tuné **vs** modèle de base
(`python evaluate.py --baseline`). Intervalles de confiance à 95 % obtenus par
**bootstrap** (1000 rééchantillonnages) :

| Métrique | Modèle de base | Fine-tuné | Gain | IC 95 % |
|---|---|---|---|---|
| ROUGE-1 | 0.260 `[0.245–0.276]` | **0.359** `[0.341–0.376]` | **+38 %** | ✅ disjoints |
| ROUGE-2 | 0.030 `[0.024–0.037]` | **0.050** `[0.041–0.059]` | **+65 %** | ✅ disjoints |
| ROUGE-L | 0.127 `[0.117–0.137]` | **0.166** `[0.154–0.177]` | **+31 %** | ✅ disjoints |
| BLEU | 1.92 `[1.19–2.70]` | **4.46** `[3.03–5.90]` | **+132 %** | ✅ disjoints |

Les intervalles du modèle de base et du modèle fine-tuné ne se recouvrent sur
aucune métrique : **le gain est statistiquement établi**, et non un artefact
d'échantillonnage.

> ⚠️ **Portée de cet intervalle** : le bootstrap quantifie le bruit dû à
> l'échantillon de test, **pas** la variance due à l'entraînement lui-même
> (un seul seed). Une validation complète exigerait 3 seeds — voir pistes.

**Courbe d'apprentissage** (loss de validation) :

| epoch | 0.5 | **1.0** | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|
| eval loss | 2.23 | **2.11** ⬅ min | 2.19 | 2.29 | 2.45 | 2.43 |
| train loss | 2.67 | 1.93 | 1.67 | 1.33 | 0.91 | 1.00 |

La loss de validation atteint son minimum à **1 epoch** puis remonte : le modèle
sur-apprend au-delà. `load_best_model_at_end` conserve le meilleur checkpoint et
un **early stopping** (patience 2) interrompt désormais l'entraînement.
Durée : **7 min 25 s** sur la RTX 4070 (~7 Go / 8 Go de VRAM).

### ✅ État du pipeline (validé end-to-end)

| Étape | Statut |
|---|---|
| Corpus curé (7 catégories) | ✅ 126 concepts, `data/corpus/` |
| Préparation (split groupe + stratifié) | ✅ 255 / 17 / 24, **fuite = 0** |
| Entraînement QLoRA (RTX 4070) | ✅ 7 min 25 s, meilleur checkpoint @ 1 epoch |
| Évaluation ROUGE/BLEU + baseline + IC | ✅ gains à IC disjoints |
| Inférence (base 4-bit + adaptateur) | ✅ |
| Fusion LoRA → modèle autonome | ✅ `merge_model.py` |
| Serveur API FastAPI (`/generate`) | ✅ testé (latence ~12 s / 120 tok) |
| Export GGUF + Ollama | 📋 documenté (nécessite llama.cpp) |
| Publication Hugging Face Hub | 📋 documenté (nécessite un token HF) |

> **Note perf** : la latence de `model.generate` non-batché (~12 s) dépasse la
> cible < 2 s du cahier des charges. En production, on passerait par **vLLM**
> (batching continu, PagedAttention) ou le modèle fusionné en fp16 — voir pistes.

---

## 🔬 Correction méthodologique : une fuite de données dans la v1

> Cette section documente un **défaut réel du projet, détecté puis corrigé**.
> Elle est conservée volontairement : savoir auditer son propre protocole
> compte davantage qu'un score flatteur.

### Le défaut

La v1 augmentait le corpus en générant, pour chaque question, deux
reformulations **partageant la même réponse de référence**, puis découpait le
tout **aléatoirement, ligne par ligne**. Conséquence mécanique : une
reformulation pouvait atterrir dans le train pendant qu'une autre allait dans le
test — avec la **réponse cible identique**.

Mesure sur le split v1 :

```
test : 8/8 exemples dont la réponse exacte est présente dans le train
       (7 réponses distinctes seulement sur 43 concepts)
```

Le modèle était donc évalué sur des réponses **vues à l'entraînement** : on
mesurait de la mémorisation, pas de la généralisation.

### Les deux conséquences

1. **Métriques gonflées.** Le BLEU annoncé (10,08) tombe à **4,46** sur un
   protocole propre. Le *gain relatif* du fine-tuning, lui, reste réel et se
   confirme même renforcé (+38 % de ROUGE-1 contre +26 % annoncés).
2. **Overfitting masqué.** En v1, la loss de validation décroissait sagement
   (2,19 → 1,68), suggérant un apprentissage sain. Une fois la fuite éliminée,
   elle **remonte dès la 2ᵉ epoch** : le modèle sur-apprenait, et la fuite le
   dissimulait. C'est ce qui a motivé l'ajout d'un early stopping.

### Le correctif

| Aspect | v1 (défaillante) | v2 (corrigée) |
|---|---|---|
| Ordre des opérations | augmenter **puis** splitter | **splitter puis** augmenter |
| Unité de découpage | la ligne | le **concept** (`group_id`) |
| Portée de l'augmentation | tout le corpus | **train uniquement** |
| Couverture des catégories | aléatoire (1 catégorie absente du test) | **stratifiée**, 7/7 par split |
| Contrôle | aucun | **garde-fou bloquant** |
| Corpus | 43 concepts | **126 concepts** |
| Fuite mesurée | **8/8** | **0/24** |

Le principe appliqué est le **group-aware splitting** (équivalent de
`GroupShuffleSplit`) : toutes les variantes d'un concept restent du même côté du
découpage. Il s'impose dès que les lignes ne sont pas indépendantes — plusieurs
mesures d'un même patient, plusieurs sessions d'un même utilisateur, ou ici
plusieurs formulations d'une même question.

### Le garde-fou

`prepare_dataset.py` **échoue** désormais si un `group_id` ou une réponse de
référence traverse le split :

```
4. Verification anti-fuite
   val   : groupes partages avec train = 0 | reponses deja vues = 0/17
   test  : groupes partages avec train = 0 | reponses deja vues = 0/24
   OK — aucun groupe ni aucune reponse partages.
```

Ce contrôle a été **validé négativement** (on lui a soumis un jeu volontairement
fuyant pour vérifier qu'il bloque bien) — un contrôle jamais vu échouer n'est
pas un contrôle.

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
│   ├── generate_dataset.py  # Assemble le corpus + attribue les group_id
│   ├── smoke_test.py        # Test du chat template + masquage (sans GPU lourd)
│   └── test_api.py          # Test de fumée de l'API FastAPI
├── data/
│   ├── corpus/              # ⭐ Corpus curé, 1 fichier JSON par catégorie
│   ├── raw/                 # raw_qa_data.json (corpus assemblé)
│   └── processed/           # train/val/test.jsonl
├── notebooks/
│   └── demo.ipynb           # Démo end-to-end (sorties + graphiques embarqués)
├── prepare_dataset.py       # Split par groupe → augmentation train → anti-fuite
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
Un corpus **curé à la main** de **126 concepts** répartis en 7 catégories
(Python, NumPy/Pandas, ML, deep learning & LLMs, évaluation, MLOps, data
engineering), un fichier JSON par catégorie dans `data/corpus/`.

**L'ordre des opérations est la garantie méthodologique** :

```
valider/dédupliquer  →  splitter PAR GROUPE (stratifié)  →  augmenter le TRAIN  →  vérifier
```

Chaque concept reçoit un `group_id` : toutes ses reformulations restent du même
côté du découpage. L'augmentation (2 paraphrases par question) n'est appliquée
qu'**après** le split et **uniquement au train**. Un contrôle final **bloque**
l'exécution si une réponse de référence traverse le split (voir
[Correction méthodologique](#-correction-méthodologique--une-fuite-de-données-dans-la-v1)).

> **Note d'honnêteté** : 126 concepts restent modestes face aux 500-2000 paires
> visées par le `CLAUDE.md`. Le corpus est conçu pour **monter en volume
> trivialement** : ajouter un objet JSON dans le fichier de catégorie adéquat
> suffit, les `group_id` étant attribués automatiquement.

### Phase 2 — Entraînement
QLoRA avec masquage du prompt. La séquence est construite via le chat template
ChatML, puis les tokens du système + de la question sont masqués (`-100`) pour
que **seule la réponse contribue au gradient**. Sélection du meilleur checkpoint
via `load_best_model_at_end` sur la loss de validation, et **early stopping**
(patience 2) — indispensable, la validation remontant dès la 2ᵉ epoch.

### Phase 3 — Évaluation
Métriques lexicales **ROUGE-1/2/L** et **BLEU** sur le jeu de test, comparaison
au modèle de base, et **intervalles de confiance à 95 % par bootstrap**
(1000 rééchantillonnages) pour distinguer un gain réel du bruit
d'échantillonnage. Les prédictions brutes sont sauvegardées dans
`eval_predictions.json` pour permettre l'analyse qualitative des échecs.

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
| Durée d'entraînement | 7 min 25 s (3 epochs, meilleur checkpoint @ 1) |

---

## 🔭 Feuille de route

**Rigueur ML**
- [ ] Entraîner sur **3 seeds** et rapporter moyenne ± écart-type (le bootstrap
      actuel ne capture que le bruit du jeu de test, pas celui de l'entraînement)
- [ ] **Sweep d'hyperparamètres** (learning rate × rang LoRA × epochs)
- [ ] Baseline **few-shot prompting** et **RAG** — le fine-tuning était-il le bon
      outil ? Comparer avant de conclure
- [ ] Évaluation sémantique (**BERTScore**, **LLM-as-a-judge**)
- [ ] Analyse qualitative des échecs à partir de `eval_predictions.json`
- [ ] Étendre le corpus à 500+ concepts

**Ingénierie**
- [ ] Suite **pytest** + **GitHub Actions**, dont un test qui échoue si la fuite
      de données réapparaît
- [ ] Dépendances figées (lockfile), `pyproject.toml`, ruff + pre-commit
- [ ] Logging structuré (commit git + seed journalisés par run)
- [ ] Dockerfile

**Serving**
- [ ] Corriger le blocage de la boucle d'événements dans `api_server.py`
      (endpoint `async` appelant une génération synchrone)
- [ ] **vLLM** + streaming SSE, authentification, rate limiting, `/metrics`
- [ ] Publier l'adaptateur sur le Hugging Face Hub + **model card**

---

## 📚 Stack

`PyTorch` · `Transformers` · `PEFT` · `bitsandbytes` · `Accelerate` ·
`Datasets` · `TensorBoard` · `FastAPI` · `Ollama` · `rouge-score` · `sacrebleu`
