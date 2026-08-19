"""
Journalisation structuree des runs.

Un `print` ne dit pas QUAND, ni SUR QUELLE VERSION du code, ni AVEC QUELLE
graine. Or sans ces trois informations, un resultat d'entrainement n'est pas
reproductible : c'est exactement le probleme que le Tier 1 a mis en evidence
(comparer deux runs suppose de savoir ce qui les distingue).

Ce module fournit :
  * `setup_logging()`  : logger console lisible + fichier JSONL par run ;
  * `run_context()`    : empreinte du run (commit git, graine, versions,
                         materiel), a journaliser au demarrage.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonLinesFormatter(logging.Formatter):
    """Une ligne JSON par evenement : indexable et interrogeable."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Champs additionnels passes via `extra={...}`
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(log_dir: str | Path | None = None,
                  name: str = "finetuning",
                  level: int = logging.INFO) -> logging.Logger:
    """Configure un logger : console lisible + fichier JSONL optionnel."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s",
                                           datefmt="%H:%M:%S"))
    logger.addHandler(console)

    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path / "run.jsonl", encoding="utf-8")
        file_handler.setFormatter(JsonLinesFormatter())
        logger.addHandler(file_handler)

    return logger


def git_commit() -> str | None:
    """Hash court du commit courant, ou None hors depot git."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def git_is_dirty() -> bool | None:
    """True si des modifications non commitees existent (run non reproductible)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def run_context(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Empreinte complete du run, a journaliser au demarrage.

    Contient le minimum permettant de rejouer un resultat : version du code,
    versions des bibliotheques determinantes, et materiel.
    """
    ctx: dict[str, Any] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    # Versions des bibliotheques critiques, si presentes.
    for module in ("torch", "transformers", "peft", "bitsandbytes", "datasets"):
        try:
            ctx[f"{module}_version"] = __import__(module).__version__
        except Exception:                       # noqa: BLE001 - module absent
            ctx[f"{module}_version"] = None

    try:
        import torch
        if torch.cuda.is_available():
            ctx["gpu"] = torch.cuda.get_device_name(0)
            ctx["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:                           # noqa: BLE001
        pass

    if extra:
        ctx.update(extra)
    return ctx


def log_run_context(logger: logging.Logger, ctx: dict[str, Any]) -> None:
    """Journalise l'empreinte du run et alerte si le depot est modifie."""
    logger.info("Contexte du run", extra={"ctx_run": ctx})
    if ctx.get("git_dirty"):
        logger.warning(
            "Depot git modifie (non commite) : ce run ne sera pas reproductible "
            "a l'identique depuis le seul hash de commit.")
