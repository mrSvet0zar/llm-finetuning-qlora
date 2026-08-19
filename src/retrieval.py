"""
Recherche documentaire pour la baseline RAG — logique PURE, sans torch.

Isole de `scripts/baselines.py` (qui charge des modeles) afin d'etre testable
en CI sans GPU.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """Recherche lexicale sur un corpus de documents.

    Deux points de conception importants :

    * Analyseur au MOT (1-2 grammes, sublinear_tf) plutot qu'au caractere.
      Compare sur les vraies donnees, l'analyseur `char_wb` produisait des
      similarites numeriquement plus elevees mais moins pertinentes : elles
      etaient dominees par des sequences de caracteres courantes en francais
      plutot que par le vocabulaire technique discriminant.

    * DEDUPLICATION par `group_id`. Le corpus d'entrainement contient les
      reformulations de chaque concept ; sans deduplication, le top-3 renvoyait
      trois variantes du MEME concept, gaspillant deux tiers du contexte fourni
      au modele.
    """

    def __init__(self, docs: list[dict]):
        if not docs:
            raise ValueError("Corpus vide.")
        self.docs = docs
        corpus = [f"{d['instruction']} {d['output']}" for d in docs]
        self.vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def query(self, question: str, k: int = 3) -> list[dict]:
        """Renvoie les k documents les plus proches, un seul par `group_id`."""
        vec = self.vectorizer.transform([question])
        sims = cosine_similarity(vec, self.matrix)[0]

        seen: set[str] = set()
        out: list[dict] = []
        for i in sims.argsort()[::-1]:
            gid = self.docs[i].get("group_id")
            if gid is not None and gid in seen:
                continue
            if gid is not None:
                seen.add(gid)
            out.append(self.docs[i])
            if len(out) == k:
                break
        return out
