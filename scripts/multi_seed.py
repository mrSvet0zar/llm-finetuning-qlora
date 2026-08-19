"""
Entrainement repete sur plusieurs graines aleatoires.

Pourquoi
--------
Le bootstrap de `evaluate.py` quantifie le bruit du JEU DE TEST : "si j'avais
tire d'autres exemples de test, quel score aurais-je obtenu ?". Il ne dit rien
du bruit de l'ENTRAINEMENT : initialisation des adaptateurs LoRA, ordre des
batchs, dropout. Deux entrainements identiques a la graine pres donnent des
modeles differents.

Rapporter un score issu d'un seul run revient donc a confondre un effet avec
une fluctuation. On reentraine ici la meme configuration avec N graines et on
rapporte moyenne +/- ecart-type.

Chaque etape s'execute en sous-processus (memoire GPU liberee entre les runs).

Usage :
    python scripts/multi_seed.py                       # config par defaut
    python scripts/multi_seed.py --lr 1e-4 --lora-r 16 # config issue du sweep
    python scripts/multi_seed.py --seeds 42 1337 2024
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS_DIR = PROJECT_ROOT / "outputs" / "seeds"
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

METRICS = ("rouge1", "rouge2", "rougeL", "bleu")


def sh(cmd: list[str], label: str) -> bool:
    print(f"    > {label} ...", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(f"    ECHEC ({label}) :\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
        return False
    print(f"      OK ({(time.perf_counter()-t0)/60:.1f} min)")
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    args = p.parse_args()

    SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []

    for seed in args.seeds:
        out_dir = SEEDS_DIR / f"seed{seed}"
        print(f"\n=== seed {seed} ===")

        train_cmd = [str(PYTHON), str(PROJECT_ROOT / "train.py"),
                     "--seed", str(seed), "--output-dir", str(out_dir),
                     "--run-name", f"seed{seed}"]
        if args.lr is not None:
            train_cmd += ["--lr", str(args.lr)]
        if args.lora_r is not None:
            train_cmd += ["--lora-r", str(args.lora_r)]
        if not sh(train_cmd, f"entrainement seed={seed}"):
            continue

        # Evaluation sur le test (sans baseline : elle ne depend pas de la graine)
        if not sh([str(PYTHON), str(PROJECT_ROOT / "evaluate.py"),
                   "--adapter", str(out_dir)], f"evaluation seed={seed}"):
            continue

        metrics = json.loads((out_dir / "eval_metrics.json").read_text(
            encoding="utf-8"))["finetuned"]
        train_m = json.loads((out_dir / "train_metrics.json").read_text(
            encoding="utf-8"))
        runs.append({"seed": seed,
                     "eval_loss": train_m.get("eval_loss"),
                     **{k: metrics[k] for k in METRICS}})

    if not runs:
        raise SystemExit("Aucun run exploitable.")

    # --- Agregation ---
    print("\n" + "=" * 66)
    print(f"RESULTATS SUR {len(runs)} GRAINES")
    print("=" * 66)
    print(f"{'seed':<8}{'eval_loss':>11}" + "".join(f"{m:>12}" for m in METRICS))
    for r in runs:
        print(f"{r['seed']:<8}{r['eval_loss']:>11.4f}"
              + "".join(f"{r[m]:>12.4f}" for m in METRICS))

    summary = {}
    print("-" * 66)
    line_mean = f"{'moyenne':<8}{'':<11}"
    line_std = f"{'ecart-type':<8}{'':<11}"
    for m in METRICS:
        vals = [r[m] for r in runs]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary[m] = {"mean": mean, "std": std,
                      "min": min(vals), "max": max(vals), "values": vals}
        line_mean += f"{mean:>12.4f}"
        line_std += f"{std:>12.4f}"
    print(line_mean)
    print(line_std)

    print("\nLecture : un ecart entre deux modeles doit depasser cet ecart-type")
    print("pour etre attribue a autre chose qu'a la variabilite d'entrainement.")

    payload = {"seeds": args.seeds,
               "config": {"lr": args.lr, "lora_r": args.lora_r},
               "runs": runs, "summary": summary}
    (SEEDS_DIR / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEcrit : {SEEDS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
