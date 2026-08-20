"""
Publication des adaptateurs LoRA et du corpus sur le Hugging Face Hub.

CE SCRIPT NE CONTIENT AUCUN JETON et n'en reclame jamais un en clair. Il
s'appuie sur l'authentification deja etablie sur la machine :

    hf auth login              # une fois, interactivement
    # ou bien : export HF_TOKEN=...

Il refuse de publier si aucune authentification n'est trouvee, plutot que de
demander un secret.

Par defaut il fonctionne en MODE SIMULATION : il affiche ce qui serait publie
sans rien envoyer. La publication reelle exige `--confirm`, car pousser sur le
Hub est une action publique et difficilement reversible.

Usage :
    python scripts/publish_to_hub.py                            # simulation (3B)
    python scripts/publish_to_hub.py --models all --dataset     # simulation complete
    python scripts/publish_to_hub.py --models all --dataset --confirm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_USER = "mrSvet0zar"
DATASET_REPO = f"{DEFAULT_USER}/corpus-python-ds-ml-fr"

# Chaque modele porte SA propre carte : publier un modele avec les chiffres
# d'un autre serait exactement le genre d'erreur que ce projet documente.
MODELES = {
    "3b": {"adapter": "outputs/qwen2.5-3b-pyds-lora",
           "repo": f"{DEFAULT_USER}/qwen2.5-3b-pyds-lora",
           "carte": "MODEL_CARD.md"},
    "7b": {"adapter": "outputs/qwen2.5-7b-pyds-lora",
           "repo": f"{DEFAULT_USER}/qwen2.5-7b-pyds-lora",
           "carte": "MODEL_CARD_7B.md"},
}

# Fichiers de l'adaptateur : petits, suffisants pour recharger le modele.
ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]


def check_auth() -> str | None:
    """Verifie qu'une authentification existe, sans jamais afficher le jeton."""
    try:
        from huggingface_hub import whoami
    except ImportError:
        print("  huggingface_hub absent : pip install huggingface-hub")
        return None
    try:
        return whoami()["name"]
    except Exception:                                          # noqa: BLE001
        return None


def collect_model_files(adapter_dir: Path) -> list[Path]:
    presents = [adapter_dir / f for f in ADAPTER_FILES if (adapter_dir / f).exists()]
    manquants = [f for f in ADAPTER_FILES if not (adapter_dir / f).exists()]
    if manquants:
        print(f"  (absents, ignores : {', '.join(manquants)})")
    return presents


def taille_mo(chemins: list[Path]) -> float:
    return sum(p.stat().st_size for p in chemins) / 1e6


def publier_modele(api, spec: dict, prive: bool) -> str:
    """Cree/met a jour un depot de modele et y envoie l'adaptateur."""
    adapter_dir = PROJECT_ROOT / spec["adapter"]
    carte = PROJECT_ROOT / spec["carte"]

    api.create_repo(spec["repo"], repo_type="model", private=prive, exist_ok=True)
    for f in collect_model_files(adapter_dir):
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=spec["repo"], repo_type="model")
        print(f"    envoye : {f.name}")
    # La carte devient le README du depot (convention du Hub)
    api.upload_file(path_or_fileobj=str(carte), path_in_repo="README.md",
                    repo_id=spec["repo"], repo_type="model")
    print(f"    envoye : {carte.name} -> README.md")
    return f"https://huggingface.co/{spec['repo']}"


def publier_dataset(api, repo: str, prive: bool) -> str:
    api.create_repo(repo, repo_type="dataset", private=prive, exist_ok=True)
    api.upload_folder(folder_path=str(PROJECT_ROOT / "data" / "corpus"),
                      path_in_repo="corpus", repo_id=repo, repo_type="dataset")
    api.upload_folder(folder_path=str(PROJECT_ROOT / "data" / "processed"),
                      path_in_repo="splits", repo_id=repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(PROJECT_ROOT / "DATASET_CARD.md"),
                    path_in_repo="README.md", repo_id=repo, repo_type="dataset")
    print("    envoye : corpus/, splits/, DATASET_CARD.md -> README.md")
    return f"https://huggingface.co/datasets/{repo}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--confirm", action="store_true",
                   help="publie reellement (sinon simulation)")
    p.add_argument("--models", nargs="*", default=["3b"],
                   choices=[*MODELES, "all"],
                   help="modeles a publier (defaut : 3b)")
    p.add_argument("--dataset", action="store_true", help="publie aussi le corpus")
    p.add_argument("--private", action="store_true", help="depots prives")
    p.add_argument("--dataset-repo", default=DATASET_REPO)
    args = p.parse_args()

    cles = list(MODELES) if "all" in args.models else args.models
    visibilite = "  (prive)" if args.private else "  (public)"

    print("=" * 66)
    print("PUBLICATION SUR LE HUGGING FACE HUB")
    print("=" * 66)

    utilisateur = check_auth()
    if utilisateur:
        print(f"  Authentifie en tant que : {utilisateur}")
    else:
        print("  NON AUTHENTIFIE — executer :  hf auth login")

    # --- Inventaire ---
    total, specs = 0.0, []
    for cle in cles:
        spec = MODELES[cle]
        adapter_dir = PROJECT_ROOT / spec["adapter"]
        carte = PROJECT_ROOT / spec["carte"]
        if not adapter_dir.exists():
            print()
            print(f"  (ignore : {adapter_dir} absent)")
            continue
        if not carte.exists():
            raise SystemExit(f"Carte manquante : {carte}")
        specs.append(spec)

        fichiers = collect_model_files(adapter_dir)
        taille = taille_mo(fichiers)
        total += taille
        print()
        print(f"Modele {cle.upper()} -> {spec['repo']}{visibilite}")
        for f in fichiers:
            print(f"    {f.name:<32} {f.stat().st_size / 1e6:8.1f} Mo")
        print(f"    {spec['carte'] + ' -> README.md':<32} "
              f"{carte.stat().st_size / 1e3:8.1f} Ko")
        print(f"    total : {taille:.1f} Mo")

    if args.dataset:
        corpus = sorted((PROJECT_ROOT / "data" / "corpus").glob("*.json"))
        splits = sorted((PROJECT_ROOT / "data" / "processed").glob("*.jsonl"))
        taille = taille_mo(corpus + splits)
        total += taille
        print()
        print(f"Dataset -> {args.dataset_repo}{visibilite}")
        print(f"    {len(corpus)} fichiers de corpus + {len(splits)} splits")
        print(f"    total : {taille:.1f} Mo")

    print()
    print(f"VOLUME TOTAL : {total:.1f} Mo")

    # --- Simulation ---
    if not args.confirm:
        print()
        print("-" * 66)
        print("MODE SIMULATION — rien n'a ete publie.")
        print("Pour publier reellement, ajouter --confirm")
        print("-" * 66)
        return

    if not utilisateur:
        raise SystemExit("Publication annulee : authentification absente.")

    from huggingface_hub import HfApi
    api = HfApi()
    urls = []

    for spec in specs:
        print()
        print(f"Publication de {spec['repo']} ...")
        urls.append(publier_modele(api, spec, args.private))

    if args.dataset:
        print()
        print(f"Publication de {args.dataset_repo} ...")
        urls.append(publier_dataset(api, args.dataset_repo, args.private))

    print()
    print("=" * 66)
    print("PUBLIE :")
    for u in urls:
        print(f"  {u}")


if __name__ == "__main__":
    main()
