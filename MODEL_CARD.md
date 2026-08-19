---
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
tags:
  - qlora
  - lora
  - peft
  - french
  - python
  - data-science
language:
  - fr
pipeline_tag: text-generation
---

# Qwen2.5-3B — Assistant Python / Data Science / ML (adaptateur QLoRA)

Adaptateur **LoRA** entraîné sur `Qwen/Qwen2.5-3B-Instruct` pour répondre en
français à des questions techniques de Python, data science et machine learning.

- **Développé par** : Milan Ganivet
- **Type** : adaptateur PEFT/LoRA (rang 16), à appliquer sur le modèle de base
- **Langue** : français
- **Licence** : Apache 2.0 (héritée du modèle de base)
- **Code source** : https://github.com/mrSvet0zar/llm-finetuning-qlora
- **Taille** : ~119 Mo (contre 6,2 Go pour le modèle fusionné)

---

## ⚠️ À lire avant tout usage

> **Ce modèle produit des réponses bien formées mais parfois FACTUELLEMENT
> FAUSSES.** Le fine-tuning a amélioré le style, le format et la longueur des
> réponses ; il n'a **pas** amélioré l'exactitude du contenu, limitée par les
> connaissances du modèle de base de 3 milliards de paramètres.
>
> Erreurs réellement observées sur le jeu de test :
> - « RAG (**Relevant Answer Generation**) » au lieu de *Retrieval-Augmented Generation*
> - définitions de **ROUGE-1 et ROUGE-L inversées**
> - `chunksize` décrit comme « une taille maximale d'entraînement »
>
> Ces erreurs sont **plausibles et bien rédigées**, donc difficiles à repérer
> pour un lecteur non expert. C'est précisément ce qui les rend dangereuses.

**Ne pas utiliser** comme source de vérité technique, dans un contexte
pédagogique sans relecture, ni dans une chaîne automatisée sans vérification
humaine.

**Usages appropriés** : démonstration technique, brouillon à relire, recherche
sur le fine-tuning parameter-efficient, base de comparaison.

---

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", device_map="auto", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "mrSvet0zar/qwen2.5-3b-pyds-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

messages = [
    {"role": "system", "content":
     "Tu es un assistant expert en Python, data science et machine learning. "
     "Tu reponds de facon claire, correcte et concise, avec des exemples "
     "pertinents quand c'est utile."},
    {"role": "user", "content": "Qu'est-ce que le broadcasting en NumPy ?"},
]
enc = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                    return_tensors="pt", return_dict=True)
out = model.generate(**enc.to(model.device), max_new_tokens=320, do_sample=False)
print(tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

Le **system prompt ci-dessus fait partie du contrat d'entraînement** : le
modèle a été entraîné avec lui. L'omettre dégrade les résultats.

---

## Données d'entraînement

Corpus **écrit à la main** de 126 concepts en français, répartis en 7 catégories
(Python, NumPy/Pandas, ML, deep learning & LLMs, évaluation, MLOps, data
engineering). Voir la [dataset card](DATASET_CARD.md).

Découpage **par concept** (group-aware) et **stratifié par catégorie** :
85 concepts d'entraînement (augmentés à 255 exemples par reformulation),
17 de validation, 24 de test. **Aucune réponse de référence n'est partagée
entre les splits.**

---

## Procédure d'entraînement

| Paramètre | Valeur |
|---|---|
| Méthode | QLoRA (4-bit NF4 + double quantization) |
| Rang LoRA / alpha | 16 / 32 |
| Modules ciblés | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Paramètres entraînés | 29,9 M / 3,12 Md (**0,96 %**) |
| Learning rate | 2e-4, scheduler cosine, 5 pas de warmup |
| Batch effectif | 16 (1 × 16 accumulations) |
| Epochs | 3 demandées, **meilleur checkpoint à 1 epoch** (early stopping) |
| Longueur max | 1024 tokens |
| Matériel | RTX 4070 Laptop, 8 Go — ~7 min |

**Le prompt est masqué dans le calcul de la loss** (labels `-100`) : seule la
réponse de l'assistant contribue au gradient.

**La validation remonte dès la 2ᵉ epoch** — sur-apprentissage structurel à ce
volume de données, observé sur toutes les configurations testées.

---

## Évaluation

Jeu de test de 24 concepts disjoints. Intervalles de confiance à 95 % par
bootstrap (1000 rééchantillonnages), accents normalisés des deux côtés.

| Approche | ROUGE-1 | ROUGE-L | BLEU | BERTScore |
|---|---|---|---|---|
| base, zero-shot | 0.304 `[0.283–0.324]` | 0.147 | 2.31 | 0.657 |
| base, few-shot | 0.299 `[0.272–0.324]` | 0.153 | 2.41 | 0.658 |
| base + RAG | 0.302 `[0.280–0.324]` | 0.148 | 3.14 | 0.665 |
| **ce modèle** | **0.360** `[0.342–0.377]` | **0.167** | **4.46** | **0.702** |

Sur **3 graines** : ROUGE-1 **0.3632 ± 0.0043**, BLEU **4.86 ± 0.68**.

**Lecture honnête** : le gain n'est **établi que sur ROUGE-1** (intervalles
disjoints). Sur ROUGE-L il recouvre celui du few-shot, sur BLEU celui du RAG.
Le gain BERTScore (+6,8 %) est bien plus modeste que le gain BLEU (+93 %) :
**plus la métrique s'approche du sens, plus l'avantage rétrécit.**

---

## Limites et biais

| Limite | Détail |
|---|---|
| **Exactitude factuelle** | Non améliorée. Erreurs plausibles et bien rédigées (voir avertissement) |
| **Taille du corpus** | 126 concepts — très en deçà d'un fine-tuning sérieux |
| **Taille du test** | 24 exemples ; les IC sont larges |
| **Couverture** | Uniquement les 7 catégories du corpus ; hors domaine, comportement non caractérisé |
| **Langue** | Corpus **sans accents** (choix initial contestable) ; le modèle a appris à ne pas en mettre |
| **Multi-tours** | Entraîné sur des échanges à un seul tour |
| **Sécurité** | Aucun alignement ni filtrage spécifique ajouté ; hérite de ceux du modèle de base |
| **Sur-apprentissage** | Le corpus est petit : risque de restitution quasi littérale d'exemples d'entraînement |

**Biais hérités** : le modèle de base a été entraîné sur des données web ; ce
fine-tuning ne corrige aucun de ses biais. Le corpus reflète en outre les choix
et angles morts d'un seul rédacteur.

---

## Impact environnemental

Entraînement : ~7 min sur une RTX 4070 Laptop (~115 W TDP), soit environ
**0,013 kWh**. Le sweep d'hyperparamètres et le multi-graines portent le total
du projet à environ **1 h de GPU**, soit ~0,12 kWh — négligeable comparé au
pré-entraînement du modèle de base.

---

## Reproduire

```bash
git clone https://github.com/mrSvet0zar/llm-finetuning-qlora
cd llm-finetuning-qlora && ./setup.ps1
python scripts/generate_dataset.py && python prepare_dataset.py
python train.py
python evaluate.py --baseline
```

Chaque run journalise son empreinte (commit git, graine, versions, GPU) dans
`train_metrics.json`.

---

## Citation

```bibtex
@misc{ganivet2026qwenpyds,
  author = {Ganivet, Milan},
  title  = {Qwen2.5-3B Python/DS/ML — adaptateur QLoRA},
  year   = {2026},
  url    = {https://github.com/mrSvet0zar/llm-finetuning-qlora}
}
```
