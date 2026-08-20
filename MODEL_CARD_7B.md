---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
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

# Qwen2.5-7B — Assistant Python / Data Science / ML (adaptateur QLoRA)

Adaptateur **LoRA** entraîné sur `Qwen/Qwen2.5-7B-Instruct`, avec **exactement
le même corpus, le même découpage et les mêmes hyperparamètres** que la
[version 3B](https://huggingface.co/mrSvet0zar/qwen2.5-3b-pyds-lora).

Ces deux adaptateurs forment une **expérience contrôlée** : une seule variable
change, la taille du modèle de base.

- **Développé par** : Milan Ganivet
- **Type** : adaptateur PEFT/LoRA (rang 16)
- **Langue** : français
- **Licence** : Apache 2.0 (héritée du modèle de base)
- **Code source** : https://github.com/mrSvet0zar/llm-finetuning-qlora
- **Taille** : ~162 Mo

---

## ⚠️ À lire avant tout usage

> **Ce modèle produit des réponses bien formées mais parfois FACTUELLEMENT
> FAUSSES.** Passer de 3 à 7 milliards de paramètres améliore l'exactitude sans
> la régler.
>
> Constat direct sur le jeu de test — le 7B **corrige** une erreur du 3B :
>
> | | Réponse sur « Qu'est-ce que ROUGE ? » |
> |---|---|
> | 3B | « ROUGE-**L** = tokens exacts, ROUGE-**1** = phrases » ❌ inversé |
> | **7B** | « rouge-1 = n-grammes, rouge-2 = bigrammes, rouge-L = longueurs communes » ✅ |
>
> …mais en **introduit une autre** : interrogé sur BERTScore, il le décrit comme
> « un modèle de classification fine-tuné » — ce qui est faux.

**Ne pas utiliser** comme source de vérité technique ni dans une chaîne
automatisée sans vérification humaine.

---

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", device_map="auto", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "mrSvet0zar/qwen2.5-7b-pyds-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

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

Le **system prompt ci-dessus fait partie du contrat d'entraînement**. L'omettre
dégrade les résultats.

---

## Résultats — et pourquoi ils sont plus intéressants que « le 7B gagne »

Jeu de test de 24 concepts, strictement disjoint de l'entraînement.
Intervalles de confiance à 95 % par bootstrap.

| Système | ROUGE-1 | ROUGE-L | BLEU | BERTScore |
|---|---|---|---|---|
| 3B base | 0.260 | 0.127 | 1.92 | 0.6569 |
| 3B fine-tuné | 0.359 | 0.166 | 4.46 | 0.7018 |
| 7B base | 0.330 | 0.157 | 3.36 | **0.6572** |
| **7B fine-tuné (ce modèle)** | **0.385** | **0.180** | **6.24** | **0.7119** |

**1. Le gain du fine-tuning rétrécit quand le modèle grandit :**

| | ROUGE-1 | ROUGE-L | BLEU |
|---|---|---|---|
| Gain sur le 3B | +38 % ✅ établi | +31 % ✅ établi | +132 % ✅ établi |
| Gain sur le 7B | +17 % ✅ établi | +14 % ⚠️ **non établi** | +86 % ⚠️ **non établi** |

Sur ce modèle, deux gains sur trois ne sont **plus statistiquement établis** :
les intervalles de confiance se recouvrent.

**2. Fine-tuner un 3B équivaut à peu près à utiliser un 7B brut.**
Le 3B fine-tuné (0.359) fait jeu égal avec le 7B non spécialisé (0.330) —
7 minutes d'entraînement contre 4 Go de VRAM permanents.

**3. Le score sémantique ne bouge pas.**
Le 7B base obtient **0.6572** de BERTScore, le 3B base **0.6569** — écart nul,
alors que leur ROUGE-1 diffère de 27 %. Le modèle plus gros **écrit davantage
dans le style de la référence sans être plus proche du sens.**

---

## Entraînement

| Paramètre | Valeur |
|---|---|
| Méthode | QLoRA (4-bit NF4 + double quantization) |
| Rang LoRA / alpha | 16 / 32 |
| Paramètres entraînés | 40,4 M / 7,66 Md (**0,53 %**) |
| Learning rate | 2e-4, cosine, 5 pas de warmup |
| Batch effectif | 16 (1 × 16 accumulations) |
| Longueur max | 768 tokens |
| Optimiseur | `adamw_8bit` (**non paginé** — voir ci-dessous) |
| Epochs | 3 demandées, arrêt à 2, **meilleur checkpoint à 1** |
| Meilleure eval_loss | **1.775** (contre 2.114 pour le 3B) |
| Matériel | RTX 4070 Laptop 8 Go — **15 min 25 s**, 5,9 Go de VRAM |

> **Note technique** : l'optimiseur `paged_adamw_8bit`, pourtant le défaut
> recommandé de QLoRA, fait planter l'entraînement sous Windows
> (`STATUS_IN_PAGE_ERROR`) — il s'appuie sur la mémoire unifiée CUDA, instable
> avec le pilote WDDM. L'optimiseur non paginé règle le problème pour un
> surcoût mémoire négligeable (~80 Mo).

Le prompt est **masqué dans le calcul de la loss** : seule la réponse contribue
au gradient. La validation remonte dès la 2ᵉ epoch — sur-apprentissage
structurel à ce volume de données, observé aussi sur le 3B.

---

## Données

Corpus **écrit à la main** de 126 concepts en français, 7 catégories.
Découpage **par concept** et **stratifié** : 85 train (augmentés à 255), 17
validation, 24 test. **Aucune réponse de référence partagée entre les splits.**

Voir la [dataset card](https://huggingface.co/datasets/mrSvet0zar/corpus-python-ds-ml-fr).

---

## Limites

| Limite | Détail |
|---|---|
| **Exactitude factuelle** | Améliorée mais non réglée (voir avertissement) |
| **Taille du corpus** | 126 concepts — très en deçà d'un fine-tuning sérieux |
| **Taille du test** | 24 exemples ; intervalles de confiance larges |
| **Gains non établis** | ROUGE-L et BLEU : les IC se recouvrent avec le modèle de base |
| **Langue** | Corpus **sans accents** ; le modèle a appris à ne pas en mettre |
| **Multi-tours** | Entraîné sur des échanges à un seul tour |
| **Couverture** | Hors des 7 catégories, comportement non caractérisé |

**Biais** : hérités du modèle de base, qu'aucun fine-tuning de cette ampleur ne
corrige. Le corpus reflète les choix d'un unique rédacteur.

---

## Impact environnemental

Entraînement : 15 min 25 s sur RTX 4070 Laptop (~115 W), soit environ
**0,03 kWh** — négligeable devant le pré-entraînement du modèle de base.

---

## Citation

```bibtex
@misc{ganivet2026qwen7bpyds,
  author = {Ganivet, Milan},
  title  = {Qwen2.5-7B Python/DS/ML — adaptateur QLoRA},
  year   = {2026},
  url    = {https://github.com/mrSvet0zar/llm-finetuning-qlora}
}
```
