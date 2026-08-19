"""
Assemble le corpus brut Q&A a partir de data/corpus/*.json.

Chaque fichier de `data/corpus/` contient les paires question/reponse CUREES
d'une categorie (le nom du fichier donne la categorie). Ce script les fusionne,
attribue a chaque concept un `group_id` STABLE, et ecrit data/raw/raw_qa_data.json.

IMPORTANT — aucune augmentation n'est faite ici.
Les reformulations de questions sont generees dans `prepare_dataset.py`, APRES
le decoupage train/val/test et UNIQUEMENT sur le jeu d'entrainement. Augmenter
avant le split ferait fuiter la meme reponse de reference des deux cotes
(voir README, section "Correction methodologique").

Usage :
    python scripts/generate_dataset.py
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
OUT_FILE = PROJECT_ROOT / "data" / "raw" / "raw_qa_data.json"


def normalize_text(text: str) -> str:
    """Uniformise le corpus : pas d'accents (coherence + robustesse console).

    Le corpus a ete redige sans accents ; cette normalisation garantit qu'un
    ajout ulterieur reste homogene avec l'existant.
    """
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def load_category(path: Path) -> list[dict]:
    """Charge un fichier de categorie et attribue les group_id."""
    category = path.stem.replace("_", "-")
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    items = []
    for i, entry in enumerate(entries):
        instruction = normalize_text(entry["instruction"].strip())
        output = normalize_text(entry["output"].strip())
        items.append({
            # group_id = unite INDIVISIBLE au moment du split.
            # Toutes les variantes d'un concept partagent ce meme id.
            "group_id": f"{category}-{i:03d}",
            "category": category,
            "instruction": instruction,
            "input": entry.get("input", ""),
            "output": output,
        })
    return items


def main() -> None:
    if not CORPUS_DIR.exists():
        raise SystemExit(f"Dossier corpus introuvable : {CORPUS_DIR}")

    files = sorted(CORPUS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"Aucun fichier .json dans {CORPUS_DIR}")

    all_items: list[dict] = []
    per_category: dict[str, int] = {}

    for path in files:
        items = load_category(path)
        all_items.extend(items)
        per_category[items[0]["category"]] = len(items)

    # Garde-fou : les group_id doivent etre uniques
    ids = [it["group_id"] for it in all_items]
    if len(ids) != len(set(ids)):
        raise SystemExit("group_id dupliques — verifier les fichiers de corpus.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"Concepts cures : {len(all_items)}")
    print(f"Ecrit dans     : {OUT_FILE}\n")
    print("Repartition par categorie :")
    for cat, n in sorted(per_category.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:22s} {n:4d}")


if __name__ == "__main__":
    main()
