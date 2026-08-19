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

Jeu de test **strictement disjoint** de l'entraînement (24 concepts, aucune
réponse partagée). Quatre approches comparées, **toutes avec la même
connaissance disponible** (les 85 concepts d'entraînement) et la même procédure
de décodage — seul le mécanisme change. Intervalles de confiance à 95 % par
**bootstrap** (1000 rééchantillonnages), accents normalisés des deux côtés
(voir [l'artefact de mesure](#️-un-artefact-de-mesure--les-accents)) :

| Approche | ROUGE-1 | ROUGE-L | BLEU |
|---|---|---|---|
| base, zero-shot | 0.304 `[0.283–0.324]` | 0.147 `[0.134–0.160]` | 2.31 `[1.5–3.1]` |
| base, few-shot (3 ex.) | 0.299 `[0.272–0.324]` | 0.153 `[0.139–0.167]` | 2.41 `[1.8–3.0]` |
| base + RAG (top-3) | 0.302 `[0.280–0.324]` | 0.148 `[0.134–0.163]` | 3.14 `[2.1–4.4]` |
| **fine-tuné (QLoRA)** | **0.360** `[0.342–0.377]` | **0.167** `[0.155–0.178]` | **4.46** `[3.0–5.9]` |

**Lecture rigoureuse des intervalles** — c'est là que le résultat devient nuancé :

| Métrique | Verdict |
|---|---|
| ROUGE-1 | fine-tuné **disjoint** du zero-shot → gain **établi** ✅ |
| ROUGE-L | **recouvre** le few-shot → gain **non établi** ⚠️ |
| BLEU | **recouvre** le RAG → gain **non établi** ⚠️ |
| BERTScore | 0.657 → **0.702** (+6,8 %), gain **modeste** |

Sur **3 graines**, le fine-tuné donne ROUGE-1 **0.3632 ± 0.0043** et BLEU
**4.86 ± 0.68** : le gain est robuste à la variabilité d'entraînement
(voir [Rigueur expérimentale](#-rigueur-expérimentale)).

> **Deux sources d'incertitude, deux mesures** : le bootstrap quantifie le bruit
> de l'**échantillon de test** ; l'écart-type multi-graines quantifie celui de
> l'**entraînement**. Les deux sont nécessaires — l'un ne remplace pas l'autre.

> 🎯 **Conclusion honnête** : le fine-tuning apporte un gain **réel mais modeste,
> et concentré sur la forme**. Il n'améliore pas l'exactitude factuelle, limitée
> par le modèle de base de 3 Md. Ni le few-shot ni le RAG ne font mieux. Pour un
> usage exigeant la justesse du contenu, le levier ne serait pas « plus de
> fine-tuning » mais **un modèle de base plus fort** ou **du RAG sur une vraie
> base documentaire**.

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
| Entraînement QLoRA (RTX 4070) | ✅ meilleur checkpoint @ 1 epoch |
| Évaluation ROUGE/BLEU + IC bootstrap | ✅ |
| Sweep d'hyperparamètres (6 configs) | ✅ non concluant — et c'est la conclusion |
| Multi-graines (3 seeds) | ✅ ROUGE-1 0.3632 ± 0.0043 |
| Baselines few-shot / RAG | ✅ aucune ne bat le fine-tuning |
| Évaluation sémantique (BERTScore) | ✅ gain modeste (+6,8 %) |
| Analyse qualitative des échecs | ✅ erreurs factuelles documentées |
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
   protocole propre. Le *gain relatif* du fine-tuning reste réel — mais un
   second artefact, celui des accents, le réduira encore ensuite (de +38 % à
   environ **+18 %** de ROUGE-1). Voir
   [l'artefact de mesure](#️-un-artefact-de-mesure--les-accents).
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

## 📐 Rigueur expérimentale

### Recherche d'hyperparamètres

`python scripts/sweep.py` — grille learning rate × rang LoRA, **6 configurations**,
sélectionnées sur la **loss de validation** (jamais sur le test).

Le nombre d'epochs n'est volontairement **pas** balayé : `load_best_model_at_end`
combiné à l'early stopping le sélectionne déjà automatiquement. Le balayer
reviendrait à optimiser deux fois la même chose.

| config | lr | rang | eval_loss | meilleur step |
|---|---|---|---|---|
| lr2e-4_r16 | 2e-4 | 16 | **2.1142** | 16 |
| lr1e-4_r32 | 1e-4 | 32 | 2.1209 | 16 |
| lr1e-4_r8 | 1e-4 | 8 | 2.1225 | 48 |
| lr2e-4_r8 | 2e-4 | 8 | 2.1241 | 24 |
| lr1e-4_r16 | 1e-4 | 16 | 2.1255 | 32 |
| lr2e-4_r32 | 2e-4 | 32 | 2.1257 | 16 |

**Étendue totale : 0.0115** — soit 0,5 % de la valeur mesurée.

> Observation transverse : `best_step = 16` (≈ 1 epoch) pour la majorité des
> configurations. **Le sur-apprentissage précoce n'est pas un accident de
> réglage, il est structurel** à ce volume de données.

### Variance d'entraînement (3 graines)

`python scripts/multi_seed.py --lr 2e-4 --lora-r 16 --seeds 42 1337 2024`

| seed | eval_loss | ROUGE-1 | ROUGE-L | BLEU |
|---|---|---|---|---|
| 42 | 2.1142 | 0.3594 | 0.1660 | 4.46 |
| 1337 | 2.1200 | 0.3679 | 0.1731 | 5.64 |
| 2024 | 2.1177 | 0.3624 | 0.1689 | 4.47 |
| **moyenne ± σ** | 2.1173 ± 0.0029 | **0.3632 ± 0.0043** | **0.1693 ± 0.0036** | **4.86 ± 0.68** |

*Contrôle de reproductibilité : le seed 42 redonne exactement `2.1142`, valeur
identique au run correspondant du sweep.*

### ⚖️ Ce que la combinaison des deux révèle

| Source de variation | écart-type |
|---|---|
| Entre **configurations** (6 configs, 1 graine chacune) | 0.0043 |
| Entre **graines** (1 config, 3 graines) | 0.0029 |

Les deux sont **du même ordre de grandeur**. L'avantage de la « meilleure »
configuration sur la deuxième (0.0067) ne représente que ~2,3 écarts-types de
bruit de graine.

> **Conclusion honnête : ce sweep ne permet pas de désigner un gagnant.**
> Avec une seule graine par configuration, l'effet des hyperparamètres est
> indiscernable du bruit d'entraînement. Un sweep concluant exigerait plusieurs
> graines par configuration (6 × 3 = 18 runs). Annoncer une « configuration
> optimale » sur cette base reviendrait à sur-apprendre sur du hasard.

En revanche, le **gain du fine-tuning lui-même est massif devant ce bruit** :
ROUGE-1 passe de 0.260 à 0.363, soit **+0.103, environ 24 écarts-types**. Cette
conclusion-là est solide.

---

## 🔎 Ce que les métriques ne disent pas

`python scripts/failure_analysis.py` — analyse des sorties réelles, sans GPU.

### Aucun échec structurel…

| Contrôle | Résultat |
|---|---|
| Réponses tronquées | 0 / 24 |
| Répétitions dégénérées | 0 / 24 |
| Réponses vides ou trop courtes | 0 / 24 |
| Ratio de longueur (généré / référence) | médiane **1.13** |

Le modèle produit systématiquement des réponses bien formées, de longueur
appropriée, dans le style du corpus. **C'est exactement ce que ROUGE et BLEU
récompensent** — et cela explique l'essentiel du gain mesuré.

### …mais des erreurs factuelles réelles

En lisant les générations, le tableau change :

| Génération du modèle | Réalité |
|---|---|
| « RAG (**Relevant Answer Generation**) » | **Retrieval**-Augmented Generation |
| « ROUGE-**L** compare les tokens exacts, ROUGE-**1** les phrases identiques » | Les deux sont inversés |
| « `chunksize` donne une taille maximale d'**entraînement** par chunk » | Aucun rapport |
| « `async`/`await` permettent d'**écrimer** les callbacks » | Mot inexistant |

> **Le modèle a appris le style et le format, pas la maîtrise du fond.**
> C'est le résultat attendu d'un fine-tuning sur 85 concepts : la forme
> s'apprend vite, le savoir bien plus lentement. Et les métriques lexicales y
> sont **structurellement aveugles**, puisqu'elles ne comptent que des mots
> communs.

C'est précisément pourquoi ce projet ajoute des métriques sémantiques et des
baselines : sans elles, on conclurait à tort que le modèle « connaît » le
domaine.

### Évaluation sémantique (BERTScore)

`python scripts/semantic_eval.py`

| Modèle | BERTScore F1 |
|---|---|
| Base | 0.6569 |
| Fine-tuné | **0.7018** |

| Approche | BERTScore F1 |
|---|---|
| base, zero-shot | 0.6569 |
| base, few-shot | 0.6581 |
| base + RAG | 0.6649 |
| **fine-tuné** | **0.7018** |

**+6,8 %** en relatif — à comparer aux **+18 %** de ROUGE-1 et **+93 %** de BLEU
sur les mêmes sorties. **Plus la métrique s'approche du sens et s'éloigne de la
forme de surface, plus le gain rétrécit.** Cet écart *est* le résultat.

*(Attention à l'échelle : BERTScore est compressé et descend rarement sous 0.6
entre textes de même langue ; les pourcentages ne sont pas comparables d'une
métrique à l'autre. C'est la tendance relative qui informe.)*

### Toutes les approches hallucinent

Sur la question « Qu'est-ce que le score ROUGE ? », les expansions produites
pour l'acronyme :

| Approche | Expansion produite |
|---|---|
| Référence | Recall-Oriented Understudy for Gisting Evaluation ✅ |
| zero-shot | « Résumé Rouge » ❌ |
| few-shot | « Résumé-Outil de Vérification Rédactionnellement Optimal » ❌ |
| RAG | « Référentiel de Outil de Qualité pour Langage Naturel » ❌ |
| fine-tuné | *n'invente pas l'acronyme, mais inverse ROUGE-1 et ROUGE-L* ❌ |

> **Le fine-tuning n'a pas introduit ces erreurs : le modèle de base 3B les
> produit déjà.** La limite est la connaissance du modèle de base, pas la
> méthode d'adaptation. Aucune des quatre approches ne la corrige.

Le RAG échoue ici pour une raison **structurelle** : le découpage étant
group-aware, aucun passage du corpus d'entraînement ne contient la réponse à une
question de test. Le RAG ne peut fournir que du contexte *voisin*. C'est ce qui
rend la comparaison équitable, mais cela sous-estime ce que donnerait un RAG en
production, avec une base documentaire couvrant réellement les questions posées.

---

## ⚠️ Un artefact de mesure : les accents

> Second défaut méthodologique détecté et corrigé, après la fuite de données.

Le corpus de référence de ce projet a été écrit **sans accents** (choix initial
de robustesse console). Le modèle fine-tuné a donc appris à ne pas en mettre,
tandis que le modèle de base écrit un français correctement accentué.

ROUGE et BLEU comparant des **tokens exacts**, `métriques` et `metriques`
comptent comme deux mots différents. La baseline était donc pénalisée pour une
raison purement orthographique :

| Approche | ROUGE-1 brut | Accents normalisés | Écart | Taux d'accents |
|---|---|---|---|---|
| zero-shot | 0.2603 | **0.3040** | **+0.044** | 2,70 % |
| few-shot | 0.2511 | **0.2988** | **+0.048** | 2,75 % |
| RAG | 0.2572 | **0.3016** | **+0.044** | 2,70 % |
| fine-tuné | 0.3594 | 0.3601 | +0.001 | **0,06 %** |

**Environ la moitié du gain ROUGE-1 initialement mesuré (+38 %) n'était qu'un
alignement de style d'écriture.** Après correction, le gain réel est d'environ
**+18 %**.

`compute_metrics` normalise désormais les accents **des deux côtés** par défaut.
`scripts/rescore.py` recalcule les métriques depuis les prédictions sauvegardées
et affiche les deux versions, pour rendre l'artefact visible plutôt que de le
corriger en silence.

> Leçon : une décision cosmétique sur le format des données (« retirer les
> accents pour la console ») a contaminé la mesure. **Tout choix de
> prétraitement est un choix méthodologique.**

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

## 🧪 Qualité logicielle

[![CI](https://github.com/mrSvet0zar/llm-finetuning-qlora/actions/workflows/ci.yml/badge.svg)](https://github.com/mrSvet0zar/llm-finetuning-qlora/actions/workflows/ci.yml)

| | |
|---|---|
| **Tests** | 42 tests, **98 % de couverture** sur `src/` |
| **Durée** | **~4 s** — sans GPU ni téléchargement de modèle |
| **Lint** | ruff (pycodestyle, pyflakes, isort, pyupgrade, bugbear) |
| **CI** | GitHub Actions, matrice Python 3.10 / 3.12 |
| **Reproductibilité** | `requirements.lock.txt` (162 versions figées) |

### Une CI qui tourne en secondes, par conception

Le code est séparé en deux couches :

- **`src/metrics.py`, `src/retrieval.py`, `prepare_dataset.py`** — logique pure,
  sans `torch` ni `transformers`
- **`train.py`, `evaluate.py`, `inference.py`** — chargement de modèles, GPU

La CI n'installe que la première (`pip install -e ".[dev]"`, trois dépendances).
Elle valide donc la logique **critique** — découpage des données, métriques,
retrieval — sans jamais toucher un GPU ni le Hugging Face Hub. Les tests
nécessitant le tokenizer sont marqués `slow` et exclus (`pytest -m "not slow"`).

### Le test qui compte

```python
def test_le_garde_fou_detecte_bien_une_fuite():
    """Un controle qu'on n'a jamais vu echouer n'est pas un controle."""
    train = [{"group_id": "g1", "instruction": "variante A", "output": "Reponse X"}]
    test  = [{"group_id": "g1", "instruction": "variante B", "output": "Reponse X"}]
    with pytest.raises(SystemExit, match="FUITE DE DONNEES"):
        assert_no_leakage(train, [], test)
```

`tests/test_split.py` verrouille la correction du Tier 0 : si l'augmentation
repassait un jour avant le découpage, **la CI échouerait**. Un second job rejoue
le pipeline de données complet sur le vrai corpus et vérifie que les splits
n'ont pas bougé (`git diff --exit-code data/processed/`).

> Deux bugs réels ont été trouvés en écrivant ces tests : `int(n × 0.15)` vidait
> la validation sur les petites catégories, et les fichiers sources du corpus
> contenaient des accents résiduels contredisant la convention.

```bash
pytest -m "not slow"      # suite rapide (CI)
pytest -m slow            # tokenizer requis, en local
ruff check .
pre-commit install        # hooks avant commit
```

---

## 📁 Structure du projet

```
finetuning/
├── src/                     # Logique reutilisable
│   ├── config.py            # Configuration centrale (dataclasses)
│   ├── metrics.py           # ⭐ Metriques + bootstrap (SANS torch -> testable en CI)
│   ├── retrieval.py         # ⭐ Retrieveur TF-IDF (SANS torch)
│   └── logging_utils.py     # ⭐ Logging structure + empreinte de run
├── tests/                   # ⭐ 42 tests, 98 % de couverture sur src/
│   ├── test_split.py        #    garde-fou anti-fuite (non-regression Tier 0)
│   ├── test_metrics.py      #    metriques, IC, normalisation des accents
│   ├── test_corpus.py       #    integrite du corpus reel
│   ├── test_retrieval.py    #    deduplication par group_id
│   ├── test_config.py       #    coherence de la configuration
│   └── test_masking.py      #    masquage du prompt (marque `slow`)
├── .github/workflows/ci.yml # ⭐ CI : ruff + pytest + pipeline de donnees
├── scripts/
│   ├── generate_dataset.py  # Assemble le corpus + attribue les group_id
│   ├── sweep.py             # ⭐ Recherche d'hyperparamètres (lr × rang LoRA)
│   ├── multi_seed.py        # ⭐ Entraînement multi-graines (variance réelle)
│   ├── baselines.py         # ⭐ zero-shot / few-shot / RAG vs fine-tuning
│   ├── semantic_eval.py     # ⭐ BERTScore + harnais LLM-as-a-judge
│   ├── failure_analysis.py  # ⭐ Analyse qualitative des échecs (sans GPU)
│   ├── rescore.py           # ⭐ Re-scoring sans regénérer (artefact accents)
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
├── Dockerfile               # ⭐ Image d'inference (multi-etapes, sans poids)
├── pyproject.toml           # ⭐ Paquet + config ruff & pytest
├── .pre-commit-config.yaml  # ⭐ Hooks avant commit
├── setup.ps1                # Installation environnement (Windows)
├── requirements.txt         # Dependances directes
├── requirements.lock.txt    # ⭐ Versions figees (reproductibilite)
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
python evaluate.py --baseline        # fine-tuné vs modèle de base, avec IC 95 %
```

### 6. Protocole expérimental complet

```bash
python scripts/sweep.py                                    # recherche d'hyperparamètres
python scripts/multi_seed.py --lr 2e-4 --lora-r 16         # variance sur 3 graines
python scripts/baselines.py --include-ft                   # few-shot / RAG / fine-tuné
python scripts/semantic_eval.py                            # BERTScore
python scripts/failure_analysis.py                         # analyse qualitative (sans GPU)
python scripts/rescore.py                                  # re-scoring sans regénérer
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
- [x] Entraînement sur **3 graines**, moyenne ± écart-type
- [x] **Sweep d'hyperparamètres** (lr × rang LoRA) — conclusion : non concluant,
      l'effet est du même ordre que le bruit de graine
- [x] Baselines **few-shot** et **RAG** — le fine-tuning était-il le bon outil ?
- [x] Évaluation sémantique **BERTScore**
- [x] Analyse qualitative des échecs
- [x] Correction de l'**artefact des accents** dans le scoring
- [ ] **Réécrire le corpus en français correctement accentué** — la normalisation
      sans accents était un mauvais choix pour un dataset français, et c'est elle
      qui a créé l'artefact de mesure
- [ ] **LLM-as-a-judge** — harnais écrit (`scripts/semantic_eval.py --judge`),
      non exécuté faute de juge fort : un Qwen-3B jugeant ses propres sorties est
      biaisé par auto-préférence. Nécessite une clé API
- [ ] Sweep **avec plusieurs graines par configuration** (6 × 3 = 18 runs), seule
      façon de conclure sur les hyperparamètres
- [ ] Comparer à un **modèle de base plus fort** (7B+) — c'est le levier que
      l'analyse d'échecs désigne pour l'exactitude factuelle
- [ ] RAG sur une **vraie base documentaire** (et non le seul corpus d'entraînement)
- [ ] Étendre le corpus à 500+ concepts

**Ingénierie**
- [x] Suite **pytest** (42 tests) + **GitHub Actions**, dont le test qui échoue
      si la fuite de données réapparaît
- [x] Dépendances figées (`requirements.lock.txt`), `pyproject.toml`, ruff,
      pre-commit
- [x] Logging structuré (commit git, graine et versions journalisés par run)
- [x] Dockerfile multi-étapes (sans poids embarqués)
- [ ] Publier l'image sur un registre + CI de build Docker
- [ ] Tests d'intégration GPU sur un runner dédié (aujourd'hui marqués `slow`)

**Serving**
- [ ] Corriger le blocage de la boucle d'événements dans `api_server.py`
      (endpoint `async` appelant une génération synchrone)
- [ ] **vLLM** + streaming SSE, authentification, rate limiting, `/metrics`
- [ ] Publier l'adaptateur sur le Hugging Face Hub + **model card**

---

## 📚 Stack

`PyTorch` · `Transformers` · `PEFT` · `bitsandbytes` · `Accelerate` ·
`Datasets` · `TensorBoard` · `FastAPI` · `Ollama` · `rouge-score` · `sacrebleu`
