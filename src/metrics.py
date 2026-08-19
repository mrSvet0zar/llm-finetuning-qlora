"""
Metriques d'evaluation — logique PURE, sans torch ni GPU.

Ce module est volontairement isole de `evaluate.py` (qui charge des modeles) :
il n'importe ni torch, ni transformers, ni peft. Consequence pratique, la CI
peut l'installer et le tester en quelques secondes, sans GPU ni telechargement
de poids.
"""
from __future__ import annotations

import random
import unicodedata

import sacrebleu
from rouge_score import rouge_scorer

ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")
ALL_KEYS = ROUGE_KEYS + ("bleu",)


def strip_accents(text: str) -> str:
    """Retire les diacritiques : 'metriques' accentue == 'metriques'."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def per_example_rouge(predictions: list[str], references: list[str]) -> list[dict]:
    """Scores ROUGE par exemple (necessaires pour le bootstrap)."""
    scorer = rouge_scorer.RougeScorer(list(ROUGE_KEYS), use_stemmer=True)
    return [
        {k: scorer.score(ref, pred)[k].fmeasure for k in ROUGE_KEYS}
        for pred, ref in zip(predictions, references, strict=True)
    ]


def compute_metrics(predictions: list[str], references: list[str],
                    n_bootstrap: int = 1000, seed: int = 42,
                    normalize_accents: bool = True) -> dict:
    """Metriques ponctuelles + intervalles de confiance a 95 % par bootstrap.

    Sur un jeu de test de quelques dizaines d'exemples, un score isole n'est
    qu'une estimation bruitee. On reechantillonne le jeu avec remise pour
    estimer la dispersion : si les intervalles de deux modeles se recouvrent
    largement, la difference observee n'est pas etablie.

    `normalize_accents` (actif par defaut) retire les diacritiques des DEUX
    cotes avant comparaison. Sans cela la mesure est biaisee : le corpus de
    reference de ce projet est ecrit sans accents, le modele fine-tune a donc
    appris a ne pas en mettre, tandis que le modele de base ecrit un francais
    correctement accentue. ROUGE comparant des tokens exacts, la baseline etait
    penalisee pour une raison purement orthographique, ce qui gonflait le gain
    d'environ +0.044 de ROUGE-1. La normalisation neutralise cet artefact.
    """
    if not predictions:
        raise ValueError("Aucune prediction a evaluer.")
    if len(predictions) != len(references):
        raise ValueError(
            f"Tailles incoherentes : {len(predictions)} predictions "
            f"pour {len(references)} references.")

    if normalize_accents:
        predictions = [strip_accents(p) for p in predictions]
        references = [strip_accents(r) for r in references]

    n = len(predictions)
    per_ex = per_example_rouge(predictions, references)

    point = {k: sum(e[k] for e in per_ex) / n for k in ROUGE_KEYS}
    point["bleu"] = sacrebleu.corpus_bleu(predictions, [references]).score

    ci = bootstrap_ci(predictions, references, per_ex, n_bootstrap, seed)
    return {**point, "ci95": ci, "n_samples": n}


def bootstrap_ci(predictions: list[str], references: list[str],
                 per_example: list[dict], n_bootstrap: int,
                 seed: int) -> dict[str, list[float]]:
    """Intervalles de confiance a 95 % par reechantillonnage avec remise."""
    n = len(predictions)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {k: [] for k in ALL_KEYS}

    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        for k in ROUGE_KEYS:
            samples[k].append(sum(per_example[i][k] for i in idx) / n)
        samples["bleu"].append(sacrebleu.corpus_bleu(
            [predictions[i] for i in idx],
            [[references[i] for i in idx]]).score)

    ci = {}
    for k, vals in samples.items():
        vals.sort()
        lo = vals[int(0.025 * n_bootstrap)]
        hi = vals[int(0.975 * n_bootstrap) - 1]
        ci[k] = [lo, hi]
    return ci


def intervals_disjoint(a: dict, b: dict, key: str) -> bool:
    """True si les IC 95 % de deux resultats ne se recouvrent pas.

    Heuristique lisible : un recouvrement signifie que la difference observee
    n'est pas etablie par les donnees.
    """
    lo_a, hi_a = a["ci95"][key]
    lo_b, hi_b = b["ci95"][key]
    return hi_a < lo_b or hi_b < lo_a
