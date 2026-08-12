"""
Preparation du dataset : validation -> deduplication -> split -> JSONL.

Entree  : data/raw/raw_qa_data.json  (genere par scripts/generate_dataset.py)
Sortie  : data/processed/{train,val,test}.jsonl

Chaque ligne de sortie conserve les champs STRUCTURES :
    {"instruction": ..., "input": ..., "output": ..., "category": ...}

On NE bake PAS le chat template ici : c'est train.py qui l'applique, afin de
garder la frontiere prompt/reponse et de permettre le masquage du prompt dans
le calcul de la loss. On ecrit tout de meme un apercu formate pour inspection.

Usage :
    python prepare_dataset.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

# Rendre src importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Config  # noqa: E402


def load_raw_data(source_file: str) -> list[dict]:
    with open(source_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    """Normalisation pour la detection de doublons."""
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_and_dedup(data: list[dict], cfg: Config) -> list[dict]:
    """Valide les champs, filtre les exemples trop courts, deduplique."""
    validated: list[dict] = []
    seen: set[str] = set()

    n_missing = n_short = n_dup = 0

    for item in data:
        instruction = (item.get("instruction") or "").strip()
        output = (item.get("output") or "").strip()

        # Champs requis
        if not instruction or not output:
            n_missing += 1
            continue

        # Longueurs minimales
        if (len(instruction) < cfg.data.min_instruction_len
                or len(output) < cfg.data.min_output_len):
            n_short += 1
            continue

        # Deduplication sur l'instruction normalisee
        key = _normalize(instruction)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)

        validated.append({
            "instruction": instruction,
            "input": (item.get("input") or "").strip(),
            "output": output,
            "category": item.get("category", "unknown"),
        })

    print("Validation :")
    print(f"  - entrees brutes        : {len(data)}")
    print(f"  - champs manquants      : {n_missing}")
    print(f"  - trop courtes          : {n_short}")
    print(f"  - doublons              : {n_dup}")
    print(f"  - retenues              : {len(validated)}")
    return validated


def split_data(data: list[dict], cfg: Config) -> tuple[list, list, list]:
    """Decoupe en train/val/test de facon deterministe."""
    rng = random.Random(cfg.data.seed)
    data = data.copy()
    rng.shuffle(data)

    n = len(data)
    n_train = int(n * cfg.data.train_ratio)
    n_val = int(n * cfg.data.val_ratio)

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test


def write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = Config()
    random.seed(cfg.data.seed)

    print(f"Chargement : {cfg.data.raw_file}\n")
    raw = load_raw_data(cfg.data.raw_file)

    validated = validate_and_dedup(raw, cfg)
    train, val, test = split_data(validated, cfg)

    processed = Path(cfg.data.processed_dir)
    write_jsonl(train, processed / "train.jsonl")
    write_jsonl(val, processed / "val.jsonl")
    write_jsonl(test, processed / "test.jsonl")

    total = len(validated)
    print("\nSplit :")
    print(f"  train : {len(train):4d} ({len(train)/total*100:.0f}%)")
    print(f"  val   : {len(val):4d} ({len(val)/total*100:.0f}%)")
    print(f"  test  : {len(test):4d} ({len(test)/total*100:.0f}%)")
    print(f"\nFichiers ecrits dans : {processed}")


if __name__ == "__main__":
    main()
