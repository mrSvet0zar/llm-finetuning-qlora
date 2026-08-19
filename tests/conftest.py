"""Fixtures partagees par la suite de tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def synthetic_corpus() -> list[dict]:
    """Corpus synthetique : 7 categories x 4 concepts, une reponse par concept.

    Reproduit la structure du vrai corpus (group_id + category) sans dependre
    de son contenu, afin que les tests restent valides quand le corpus evolue.
    """
    categories = ["cat-a", "cat-b", "cat-c", "cat-d", "cat-e", "cat-f", "cat-g"]
    items = []
    for cat in categories:
        for i in range(4):
            items.append({
                "group_id": f"{cat}-{i:03d}",
                "category": cat,
                "instruction": f"Question numero {i} de la categorie {cat} ?",
                "input": "",
                "output": (f"Reponse de reference unique pour le concept {i} "
                           f"de la categorie {cat}, suffisamment longue."),
            })
    return items
