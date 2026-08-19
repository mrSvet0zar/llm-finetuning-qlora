"""
Recherche d'hyperparametres (grid search) : learning rate x rang LoRA.

Methodologie
------------
* La selection se fait sur la LOSS DE VALIDATION, jamais sur le test. Le test
  reste intact jusqu'a l'evaluation finale du modele retenu.
* Le nombre d'epochs n'est PAS balaye : l'early stopping combine a
  `load_best_model_at_end` le selectionne automatiquement pour chaque config.
  Le balayer reviendrait a optimiser deux fois la meme chose.
* Chaque run s'execute dans un SOUS-PROCESSUS distinct : cela garantit que la
  memoire GPU est integralement liberee entre deux configurations (sinon la
  fragmentation finit par provoquer un OOM sur 8 Go).
* Tous les runs partagent la meme graine, afin que les ecarts observes
  proviennent des hyperparametres et non de l'initialisation.

Sortie : outputs/sweep/results.json + tableau recapitulatif.

Usage :
    python scripts/sweep.py
    python scripts/sweep.py --dry-run     # affiche la grille sans entrainer
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = PROJECT_ROOT / "outputs" / "sweep"
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():                      # environnement non Windows
    PYTHON = Path(sys.executable)

# --- Grille ---------------------------------------------------------------
LEARNING_RATES = [1e-4, 2e-4]
LORA_RANKS = [8, 16, 32]
SEED = 42


def run_one(lr: float, r: int) -> dict:
    """Lance un entrainement en sous-processus et renvoie ses metriques."""
    tag = f"lr{lr:g}_r{r}"
    out_dir = SWEEP_DIR / tag

    cmd = [
        str(PYTHON), str(PROJECT_ROOT / "train.py"),
        "--lr", str(lr),
        "--lora-r", str(r),
        "--seed", str(SEED),
        "--output-dir", str(out_dir),
        "--run-name", f"sweep-{tag}",
    ]

    print(f"\n{'='*70}\n>>> {tag}\n{'='*70}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    duration = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"ECHEC ({tag}) :\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        return {"tag": tag, "lr": lr, "lora_r": r, "status": "failed",
                "duration_s": duration}

    metrics_file = out_dir / "train_metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

    # Step du meilleur checkpoint (early stopping).
    # Tri NUMERIQUE : un tri lexicographique placerait checkpoint-8 apres
    # checkpoint-16.
    state_files = sorted(
        out_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    best_step = None
    if state_files:
        state = json.loads(state_files[-1].read_text(encoding="utf-8"))
        best_step = state.get("best_global_step") or state.get("global_step")

    result = {
        "tag": tag,
        "lr": lr,
        "lora_r": r,
        "status": "ok",
        "eval_loss": metrics.get("eval_loss"),
        "train_loss": metrics.get("train_loss"),
        "best_step": best_step,
        "duration_s": round(duration, 1),
    }
    print(f"<<< {tag} : eval_loss={result['eval_loss']:.4f} "
          f"({duration/60:.1f} min)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grid = list(product(LEARNING_RATES, LORA_RANKS))
    print(f"Grille : {len(LEARNING_RATES)} lr x {len(LORA_RANKS)} rangs "
          f"= {len(grid)} configurations")
    for lr, r in grid:
        print(f"  lr={lr:g}  r={r}  (alpha={2*r})")
    if args.dry_run:
        return

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_one(lr, r) for lr, r in grid]

    ok = [r for r in results if r["status"] == "ok"]
    ok.sort(key=lambda r: r["eval_loss"])

    print(f"\n{'='*70}\nRESULTATS (tries par eval_loss croissante)\n{'='*70}")
    print(f"{'config':<14}{'lr':>8}{'rang':>6}{'eval_loss':>12}"
          f"{'best_step':>11}{'duree':>9}")
    for r in ok:
        print(f"{r['tag']:<14}{r['lr']:>8.0e}{r['lora_r']:>6}"
              f"{r['eval_loss']:>12.4f}{str(r['best_step']):>11}"
              f"{r['duration_s']/60:>8.1f}m")

    if ok:
        best = ok[0]
        print(f"\nMeilleure configuration : {best['tag']} "
              f"(eval_loss = {best['eval_loss']:.4f})")

    payload = {
        "grid": {"learning_rates": LEARNING_RATES, "lora_ranks": LORA_RANKS,
                 "seed": SEED},
        "selection_metric": "eval_loss (jeu de VALIDATION)",
        "results": results,
        "best": ok[0] if ok else None,
    }
    (SWEEP_DIR / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEcrit : {SWEEP_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
