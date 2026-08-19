"""
Publication de l'adaptateur LoRA et du dataset sur le Hugging Face Hub.

CE SCRIPT NE CONTIENT AUCUN JETON et n'en demande jamais un en clair. Il
s'appuie sur l'authentification deja etablie sur la machine :

    huggingface-cli login          # une fois, interactivement
    # ou bien : export HF_TOKEN=...

Il refuse de publier si aucune authentification n'est trouvee, plutot que de
reclamer un secret.

Par defaut, il fonctionne en MODE SIMULATION : il affiche ce qui serait
publie, sans rien envoyer. La publication reelle exige `--confirm`, car
pousser sur le Hub est une action publique et difficilement reversible.

Usage :
    python scripts/publish_to_hub.py                       # simulation
    python scripts/publish_to_hub.py --confirm             # publie l'adaptateur
    python scripts/publish_to_hub.py --confirm --dataset   # publie aussi le dataset
    python scripts/publish_to_hub.py --private             # depot prive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402

DEFAULT_USER = "mrSvet0zar"
MODEL_REPO = f"{DEFAULT_USER}/qwen2.5-3b-pyds-lora"
DATASET_REPO = f"{DEFAULT_USER}/corpus-python-ds-ml-fr"

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--confirm", action="store_true",
                   help="publie reellement (sinon simulation)")
    p.add_argument("--dataset", action="store_true",
                   help="publie aussi le corpus")
    p.add_argument("--private", action="store_true", help="depot prive")
    p.add_argument("--model-repo", default=MODEL_REPO)
    p.add_argument("--dataset-repo", default=DATASET_REPO)
    p.add_argument("--adapter", default=None)
    args = p.parse_args()

    cfg = Config()
    adapter_dir = Path(args.adapter or cfg.train.output_dir)
    if not adapter_dir.exists():
        raise SystemExit(f"Adaptateur introuvable : {adapter_dir}\n"
                         "Lancer d'abord : python train.py")

    print("=" * 66)
    print("PUBLICATION SUR LE HUGGING FACE HUB")
    print("=" * 66)

    utilisateur = check_auth()
    if utilisateur:
        print(f"  Authentifie en tant que : {utilisateur}")
    else:
        print("  NON AUTHENTIFIE")
        print("  Executer d'abord :  huggingface-cli login")

    # --- Inventaire ---
    fichiers_modele = collect_model_files(adapter_dir)
    print(f"\nModele  -> {args.model_repo}"
          f"{'  (prive)' if args.private else '  (public)'}")
    for f in fichiers_modele:
        print(f"    {f.name:<32} {f.stat().st_size / 1e6:8.1f} Mo")
    print(f"    {'MODEL_CARD.md -> README.md':<32} "
          f"{(PROJECT_ROOT / 'MODEL_CARD.md').stat().st_size / 1e3:8.1f} Ko")
    print(f"    total : {taille_mo(fichiers_modele):.1f} Mo")

    if args.dataset:
        corpus = sorted((PROJECT_ROOT / "data" / "corpus").glob("*.json"))
        splits = sorted((PROJECT_ROOT / "data" / "processed").glob("*.jsonl"))
        print(f"\nDataset -> {args.dataset_repo}")
        print(f"    {len(corpus)} fichiers de corpus + {len(splits)} splits")
        print(f"    total : {taille_mo(corpus + splits):.1f} Mo")

    # --- Simulation ---
    if not args.confirm:
        print("\n" + "-" * 66)
        print("MODE SIMULATION — rien n'a ete publie.")
        print("Pour publier reellement :")
        print("    python scripts/publish_to_hub.py --confirm"
              + (" --dataset" if args.dataset else ""))
        print("-" * 66)
        return

    if not utilisateur:
        raise SystemExit("\nPublication annulee : authentification absente.\n"
                         "Executer : huggingface-cli login")

    from huggingface_hub import HfApi
    api = HfApi()

    # --- Modele ---
    print(f"\nCreation/mise a jour de {args.model_repo} ...")
    api.create_repo(args.model_repo, repo_type="model",
                    private=args.private, exist_ok=True)
    for f in fichiers_modele:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=args.model_repo, repo_type="model")
        print(f"    envoye : {f.name}")
    # La model card devient le README du depot (convention du Hub)
    api.upload_file(path_or_fileobj=str(PROJECT_ROOT / "MODEL_CARD.md"),
                    path_in_repo="README.md", repo_id=args.model_repo,
                    repo_type="model")
    print("    envoye : MODEL_CARD.md -> README.md")
    print(f"  -> https://huggingface.co/{args.model_repo}")

    # --- Dataset ---
    if args.dataset:
        print(f"\nCreation/mise a jour de {args.dataset_repo} ...")
        api.create_repo(args.dataset_repo, repo_type="dataset",
                        private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(PROJECT_ROOT / "data" / "corpus"),
                          path_in_repo="corpus", repo_id=args.dataset_repo,
                          repo_type="dataset")
        api.upload_folder(folder_path=str(PROJECT_ROOT / "data" / "processed"),
                          path_in_repo="splits", repo_id=args.dataset_repo,
                          repo_type="dataset")
        api.upload_file(path_or_fileobj=str(PROJECT_ROOT / "DATASET_CARD.md"),
                        path_in_repo="README.md", repo_id=args.dataset_repo,
                        repo_type="dataset")
        print(f"  -> https://huggingface.co/datasets/{args.dataset_repo}")

    print("\nPublication terminee.")


if __name__ == "__main__":
    main()
