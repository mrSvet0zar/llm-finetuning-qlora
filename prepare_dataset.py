"""
Preparation du dataset : validation -> split PAR GROUPE -> augmentation du train.

Ordre des operations (l'ordre EST la correction methodologique) :

    1. valider et dedupliquer
    2. decouper train/val/test PAR GROUPE  (un concept = un groupe indivisible)
    3. augmenter le TRAIN uniquement (reformulations de questions)
    4. verifier l'absence de fuite  -> echec bloquant si detectee

Pourquoi : chaque concept possede une reponse de reference unique. Si l'on
augmentait AVANT le decoupage, les reformulations d'une meme question se
retrouveraient de part et d'autre du split, avec la MEME reponse cible : le
modele serait alors evalue sur des reponses vues en entrainement, ce qui
mesure de la memorisation et gonfle les scores. C'est le defaut qui affectait
la v1 de ce projet (8/8 des exemples de test etaient concernes).

Entree  : data/raw/raw_qa_data.json   (genere par scripts/generate_dataset.py)
Sortie  : data/processed/{train,val,test}.jsonl

Usage :
    python prepare_dataset.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import Config  # noqa: E402

# --------------------------------------------------------------------------
#  Reformulations appliquees AU TRAIN UNIQUEMENT
# --------------------------------------------------------------------------
PARAPHRASE_TEMPLATES = [
    "Peux-tu m'expliquer : {q}",
    "J'aimerais comprendre. {q}",
    "Explique simplement : {q}",
    "En quelques phrases, {ql}",
    "Pour un entretien technique : {q}",
]
MAX_PARAPHRASES = 2  # en plus de la question originale


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def _normalize(text: str) -> str:
    """Normalisation pour comparaison (doublons, detection de fuite)."""
    return re.sub(r"\s+", " ", text.strip().lower())


# --------------------------------------------------------------------------
#  1. Validation
# --------------------------------------------------------------------------
def load_raw_data(source_file: str) -> list[dict]:
    with open(source_file, encoding="utf-8") as f:
        return json.load(f)


def validate_and_dedup(data: list[dict], cfg: Config) -> list[dict]:
    validated: list[dict] = []
    seen: set[str] = set()
    n_missing = n_short = n_dup = 0

    for item in data:
        instruction = (item.get("instruction") or "").strip()
        output = (item.get("output") or "").strip()

        if not instruction or not output:
            n_missing += 1
            continue
        if (len(instruction) < cfg.data.min_instruction_len
                or len(output) < cfg.data.min_output_len):
            n_short += 1
            continue

        key = _normalize(instruction)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)

        validated.append({
            "group_id": item.get("group_id") or f"auto-{_normalize(output)[:40]}",
            "category": item.get("category", "unknown"),
            "instruction": instruction,
            "input": (item.get("input") or "").strip(),
            "output": output,
        })

    print("1. Validation")
    print(f"   entrees brutes   : {len(data)}")
    print(f"   champs manquants : {n_missing}")
    print(f"   trop courtes     : {n_short}")
    print(f"   doublons         : {n_dup}")
    print(f"   retenues         : {len(validated)}")
    return validated


# --------------------------------------------------------------------------
#  2. Split PAR GROUPE
# --------------------------------------------------------------------------
def _split_sizes(n: int, cfg: Config) -> tuple[int, int]:
    """Tailles (train, val) pour n concepts d'une meme categorie.

    La troncature de `int(n * ratio)` vide la validation sur les petites
    categories : avec n=4 et un ratio de 0.15, `int(0.6)` vaut 0. On garantit
    donc au moins un concept en validation ET en test des que la categorie en
    compte au moins trois.
    """
    if n <= 1:
        return n, 0                       # tout au train
    if n == 2:
        return 1, 0                       # 1 train, 1 test

    n_val = max(1, int(n * cfg.data.val_ratio))
    n_train = int(n * cfg.data.train_ratio)
    # Laisse au minimum un concept au test
    n_train = min(n_train, n - n_val - 1)
    return max(n_train, 1), n_val


def group_aware_split(data: list[dict], cfg: Config) -> tuple[list, list, list]:
    """Decoupe en train/val/test sans jamais scinder un groupe, ET en
    stratifiant par categorie.

    Deux garanties :
      * GROUPE  : un `group_id` (= un concept et sa reponse de reference) est
        attribue entierement a un seul split (cf. GroupShuffleSplit).
      * STRATIFICATION : le decoupage est applique categorie par categorie, de
        sorte que chaque split couvre les 7 domaines proportionnellement. Sans
        cela, une categorie entiere peut disparaitre du test par hasard et
        l'evaluation n'est plus representative.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        groups[item["group_id"]].append(item)

    # Regroupe les identifiants de groupe par categorie
    by_category: dict[str, list[str]] = defaultdict(list)
    for gid, items in groups.items():
        by_category[items[0]["category"]].append(gid)

    rng = random.Random(cfg.data.seed)
    split_ids: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for category in sorted(by_category):        # tri = determinisme
        gids = sorted(by_category[category])
        rng.shuffle(gids)
        n_train, n_val = _split_sizes(len(gids), cfg)
        split_ids["train"] += gids[:n_train]
        split_ids["val"] += gids[n_train:n_train + n_val]
        split_ids["test"] += gids[n_train + n_val:]

    out = {name: [item for gid in ids for item in groups[gid]]
           for name, ids in split_ids.items()}

    print("\n2. Split par groupe (stratifie par categorie)")
    print(f"   concepts (groupes) : {len(groups)}")
    for name in ("train", "val", "test"):
        n_cats = len({groups[g][0]['category'] for g in split_ids[name]})
        print(f"   {name:5s} : {len(split_ids[name]):3d} concepts "
              f"-> {len(out[name]):3d} exemples | {n_cats}/7 categories")
    return out["train"], out["val"], out["test"]


# --------------------------------------------------------------------------
#  3. Augmentation du TRAIN uniquement
# --------------------------------------------------------------------------
def augment(items: list[dict], cfg: Config) -> list[dict]:
    rng = random.Random(cfg.data.seed)
    augmented: list[dict] = []

    for item in items:
        augmented.append({**item, "variant": "original"})
        templates = rng.sample(PARAPHRASE_TEMPLATES,
                               k=min(MAX_PARAPHRASES, len(PARAPHRASE_TEMPLATES)))
        for tpl in templates:
            augmented.append({
                **item,
                "instruction": tpl.format(q=item["instruction"],
                                          ql=_lower_first(item["instruction"])),
                "variant": "paraphrase",
            })

    rng.shuffle(augmented)
    print("\n3. Augmentation (train uniquement)")
    print(f"   {len(items)} -> {len(augmented)} exemples "
          f"(x{len(augmented)/max(len(items),1):.1f})")
    return augmented


# --------------------------------------------------------------------------
#  4. Verification anti-fuite (bloquante)
# --------------------------------------------------------------------------
def assert_no_leakage(train: list[dict], val: list[dict], test: list[dict]) -> None:
    """Echoue si un groupe ou une reponse de reference traverse le split."""
    print("\n4. Verification anti-fuite")
    problems: list[str] = []

    g_train = {i["group_id"] for i in train}
    o_train = {_normalize(i["output"]) for i in train}

    for name, split in (("val", val), ("test", test)):
        g_overlap = g_train & {i["group_id"] for i in split}
        o_overlap = [i for i in split if _normalize(i["output"]) in o_train]

        print(f"   {name:5s} : groupes partages avec train = {len(g_overlap)} | "
              f"reponses deja vues = {len(o_overlap)}/{len(split)}")
        if g_overlap:
            problems.append(f"{len(g_overlap)} group_id partages entre train et {name}")
        if o_overlap:
            problems.append(
                f"{len(o_overlap)} reponses de {name} presentes dans le train")

    if problems:
        raise SystemExit("FUITE DE DONNEES DETECTEE :\n  - " + "\n  - ".join(problems))
    print("   OK — aucun groupe ni aucune reponse partages.")


# --------------------------------------------------------------------------
def write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = Config()
    print(f"Chargement : {cfg.data.raw_file}\n")
    raw = load_raw_data(cfg.data.raw_file)

    validated = validate_and_dedup(raw, cfg)
    train, val, test = group_aware_split(validated, cfg)

    # L'augmentation vient APRES le split, et seulement sur le train.
    train = augment(train, cfg)

    assert_no_leakage(train, val, test)

    processed = Path(cfg.data.processed_dir)
    write_jsonl(train, processed / "train.jsonl")
    write_jsonl(val, processed / "val.jsonl")
    write_jsonl(test, processed / "test.jsonl")

    print(f"\nFichiers ecrits dans : {processed}")
    print(f"  train : {len(train):4d} exemples")
    print(f"  val   : {len(val):4d} exemples")
    print(f"  test  : {len(test):4d} exemples")


if __name__ == "__main__":
    main()
