---
license: cc-by-4.0
language:
  - fr
task_categories:
  - question-answering
  - text-generation
tags:
  - python
  - data-science
  - machine-learning
  - instruction-tuning
size_categories:
  - n<1K
---

# Corpus Q&A Python / Data Science / ML (français)

Corpus **écrit à la main** de 126 paires question/réponse en français sur
Python, la data science et le machine learning, conçu pour le fine-tuning
d'instruction d'un LLM.

- **Auteur** : Milan Ganivet
- **Langue** : français
- **Licence** : CC BY 4.0
- **Source** : https://github.com/mrSvet0zar/llm-finetuning-qlora

---

## Composition

| Catégorie | Concepts |
|---|---|
| deep-learning-llm | 24 |
| ml-fundamentals | 20 |
| python-core | 20 |
| numpy-pandas | 18 |
| mlops | 16 |
| data-engineering | 14 |
| evaluation | 14 |
| **Total** | **126** |

| | |
|---|---|
| Longueur des questions | 10 mots en moyenne (6 → 15) |
| Longueur des réponses | 109 mots en moyenne (83 → 125) |
| Volume total | ~13 700 mots de réponses |

### Structure

Un fichier JSON par catégorie dans `data/corpus/` :

```json
{
  "instruction": "Qu'est-ce que le broadcasting en NumPy ?",
  "output": "Le broadcasting est le mecanisme qui permet a NumPy d'appliquer..."
}
```

`scripts/generate_dataset.py` assemble ces fichiers et attribue à chaque
concept un **`group_id` stable** (`{categorie}-{index}`), unité indivisible du
découpage.

---

## Découpage — le point méthodologique central

| Split | Exemples | Concepts | Catégories |
|---|---|---|---|
| train | 255 | 85 | 7/7 |
| validation | 17 | 17 | 7/7 |
| test | 24 | 24 | 7/7 |

Trois garanties, dans cet ordre :

1. **Découpage par groupe** — toutes les variantes d'un concept restent du même
   côté. Chaque concept possède une réponse de référence unique ; les répartir
   entre splits reviendrait à évaluer le modèle sur des réponses vues.
2. **Stratification par catégorie** — les 7 domaines sont représentés dans
   chaque split. Sans cela, une catégorie entière peut disparaître du test.
3. **Augmentation après le découpage, sur le train seul** — deux reformulations
   par question (255 = 85 × 3). Validation et test conservent **une seule
   question canonique par concept**.

Un contrôle **bloquant** vérifie qu'aucun `group_id` ni aucune réponse de
référence ne traverse le découpage, et un test de non-régression
(`tests/test_split.py`) échoue si cette propriété se perd.

> ### Une fuite de données dans la v1
>
> La première version augmentait **avant** de découper, aléatoirement ligne par
> ligne : **8/8** des exemples de test avaient leur réponse exacte présente
> dans le train. Les métriques étaient gonflées, et le sur-apprentissage
> masqué. Ce défaut et sa correction sont documentés dans le
> [README](README.md#-correction-méthodologique--une-fuite-de-données-dans-la-v1).

---

## Constitution

Rédaction manuelle, sans scraping ni génération par un LLM. Chaque réponse vise
la concision (~110 mots), l'exactitude technique, et un style homogène :
définition, mécanisme, exemple ou cas d'usage, puis limite ou piège courant.

Aucune donnée personnelle, aucun contenu sous licence tierce. Aucune annotation
externe : un seul rédacteur, sans processus de relecture croisée.

---

## Limites et biais

| Limite | Détail |
|---|---|
| **Taille** | 126 concepts — un ordre de grandeur sous ce qu'exige un fine-tuning sérieux |
| **Rédacteur unique** | Aucune relecture croisée ; les angles morts d'une seule personne sont reproduits tels quels |
| **Absence d'accents** | Le corpus est écrit **sans accents** (choix initial de robustesse console). Mauvais choix pour un corpus français : le modèle apprend à ne pas en mettre, ce qui a **biaisé les métriques lexicales** (voir ci-dessous) |
| **Un seul tour** | Pas de conversations multi-tours |
| **Pas de cas négatifs** | Aucune question hors domaine appelant un refus ou une réserve |
| **Formulations homogènes** | Questions de style proche ; peu représentatif du langage utilisateur réel |
| **Couverture** | Notions fondamentales uniquement ; ni bibliothèques de niche, ni versions récentes |

### L'artefact des accents

Le modèle de base écrit un français correctement accentué (2,70 % des
caractères), le modèle fine-tuné a appris à ne pas en mettre (0,06 %). ROUGE et
BLEU comparant des **tokens exacts**, `métriques` ≠ `metriques` : la baseline
était pénalisée pour une raison purement orthographique, gonflant le gain
mesuré de ~+0.044 de ROUGE-1 — **environ la moitié du gain annoncé**.

Le scoring normalise désormais les accents des deux côtés. **La correction de
fond serait de réécrire le corpus en français correctement accentué**, ce qui
reste à faire.

> Leçon transférable : un choix de prétraitement présenté comme cosmétique
> (« retirer les accents pour la console ») peut contaminer la mesure.

---

## Usage recommandé

Adapté à : démonstration de pipeline de fine-tuning, recherche sur le PEFT,
base de comparaison entre méthodes d'adaptation.

**Non adapté** à : entraîner un assistant technique destiné à un usage réel
(volume trop faible), ni à servir de référence de vérité factuelle.

---

## Citation

```bibtex
@misc{ganivet2026corpuspyds,
  author = {Ganivet, Milan},
  title  = {Corpus Q&A Python/Data Science/ML en francais},
  year   = {2026},
  url    = {https://github.com/mrSvet0zar/llm-finetuning-qlora}
}
```
