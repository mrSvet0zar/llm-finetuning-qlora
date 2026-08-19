"""
Evaluation SEMANTIQUE des predictions : BERTScore + juge LLM.

Motivation
----------
L'analyse qualitative des sorties a montre que le modele fine-tune produit des
reponses au bon FORMAT mais parfois factuellement fausses (par exemple "RAG =
Relevant Answer Generation" au lieu de Retrieval-Augmented Generation). BLEU et
ROUGE sont structurellement incapables de detecter cela : ils comptent des mots
communs, pas du sens.

Deux niveaux de mesure complementaires sont ajoutes ici :

  * BERTScore : compare des embeddings contextuels plutot que des mots exacts.
    Recompense les paraphrases correctes. Ne verifie PAS la veracite.
  * Juge LLM  : note chaque reponse sur des criteres explicites, dont
    l'EXACTITUDE FACTUELLE, seul critere reellement pertinent ici.

Le juge accepte deux backends :
  - API Anthropic si ANTHROPIC_API_KEY est definie (juge fort, recommande) ;
  - a defaut, le modele de base local, avec un avertissement : un modele de 3B
    qui juge un derive de lui-meme est un juge faible et biaise (auto-preference).
    Les resultats sont alors INDICATIFS et signales comme tels.

Usage :
    python scripts/semantic_eval.py                      # BERTScore seul
    python scripts/semantic_eval.py --judge              # + juge LLM
    python scripts/semantic_eval.py --predictions outputs/.../baselines_predictions.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402

# Cles a ignorer dans le fichier de predictions (ce ne sont pas des modeles)
NON_MODEL_KEYS = {"questions", "references"}


# ---------------------------------------------------------------------------
#  BERTScore
# ---------------------------------------------------------------------------
def run_bertscore(preds: list[str], refs: list[str], device: str = "cpu") -> dict:
    from bert_score import score as bert_score
    # lang="fr" selectionne un modele multilingue adapte au francais.
    # device="cpu" par defaut : le volume est faible (quelques dizaines de
    # textes) et cela evite d'entrer en concurrence avec un entrainement en
    # cours sur les 8 Go de VRAM.
    P, R, F1 = bert_score(preds, refs, lang="fr", verbose=False,
                          rescale_with_baseline=False, device=device)
    return {
        "bertscore_p": P.mean().item(),
        "bertscore_r": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
        "per_example_f1": [round(v, 4) for v in F1.tolist()],
    }


# ---------------------------------------------------------------------------
#  Juge LLM
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """Tu evalues la reponse d'un assistant technique a une question \
de Python / data science / machine learning.

QUESTION :
{question}

REPONSE DE REFERENCE (consideree correcte) :
{reference}

REPONSE A EVALUER :
{prediction}

Note la reponse a evaluer sur trois criteres, chacun de 1 a 5 :
- exactitude : les affirmations sont-elles factuellement correctes ? (critere le \
plus important ; une definition erronee ou un acronyme faux doit faire chuter la note)
- pertinence : la reponse traite-t-elle reellement la question posee ?
- clarte : la reponse est-elle structuree et comprehensible ?

Reponds UNIQUEMENT par un objet JSON, sans texte autour :
{{"exactitude": <1-5>, "pertinence": <1-5>, "clarte": <1-5>, "justification": "<une phrase>"}}"""


def judge_via_api(items: list[dict]) -> list[dict] | None:
    """Juge via l'API Anthropic si une cle est disponible."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        print("  (paquet `anthropic` absent : pip install anthropic)")
        return None

    client = anthropic.Anthropic(api_key=api_key)
    out = []
    for i, it in enumerate(items, 1):
        msg = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(**it)}],
        )
        out.append(parse_judge_output(msg.content[0].text))
        print(f"    {i}/{len(items)}")
    return out


def judge_via_local(items: list[dict]) -> list[dict]:
    """Repli : juge avec le modele de base local (faible, biaise)."""
    import torch  # noqa: F401

    from evaluate import load_baseline
    from inference import generate_from_messages

    print("  ATTENTION : juge local (Qwen2.5-3B). Un modele de cette taille qui")
    print("  evalue un derive de lui-meme est un juge FAIBLE et sujet a")
    print("  l'auto-preference. Resultats indicatifs uniquement.")

    cfg = Config()
    model, tok = load_baseline(cfg)
    out = []
    for i, it in enumerate(items, 1):
        txt = generate_from_messages(model, tok, [
            {"role": "system", "content": "Tu es un evaluateur rigoureux. "
                                          "Tu reponds uniquement en JSON."},
            {"role": "user", "content": JUDGE_PROMPT.format(**it)},
        ], max_new_tokens=200, temperature=0.0)
        out.append(parse_judge_output(txt))
        if i % 6 == 0 or i == len(items):
            print(f"    {i}/{len(items)}")
    return out


def parse_judge_output(text: str) -> dict:
    """Extrait le JSON de la sortie du juge, tolerant au texte parasite."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"exactitude": None, "pertinence": None, "clarte": None,
                "parse_error": True}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"exactitude": None, "pertinence": None, "clarte": None,
                "parse_error": True}
    for k in ("exactitude", "pertinence", "clarte"):
        v = data.get(k)
        data[k] = v if isinstance(v, (int, float)) and 1 <= v <= 5 else None
    return data


def summarize_judge(scores: list[dict]) -> dict:
    out = {}
    for crit in ("exactitude", "pertinence", "clarte"):
        vals = [s[crit] for s in scores if s.get(crit) is not None]
        out[crit] = round(sum(vals) / len(vals), 2) if vals else None
        out[f"{crit}_n"] = len(vals)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default=None)
    p.add_argument("--judge", action="store_true", help="active le juge LLM")
    p.add_argument("--models", nargs="*", default=None,
                   help="sous-ensemble de modeles a evaluer")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="peripherique pour BERTScore (defaut cpu)")
    args = p.parse_args()

    cfg = Config()
    pred_file = Path(args.predictions) if args.predictions else (
        Path(cfg.train.output_dir) / "eval_predictions.json")
    if not pred_file.exists():
        raise SystemExit(f"Introuvable : {pred_file}")

    data = json.loads(pred_file.read_text(encoding="utf-8"))
    questions, refs = data["questions"], data["references"]
    models = args.models or [k for k in data if k not in NON_MODEL_KEYS]
    print(f"Fichier : {pred_file.name}")
    print(f"Modeles : {', '.join(models)}  ({len(questions)} exemples)\n")

    results: dict[str, dict] = {}

    for name in models:
        preds = data[name]
        print(f"=== {name} ===")
        print("  BERTScore ...")
        res = run_bertscore(preds, refs, device=args.device)
        print(f"    F1 = {res['bertscore_f1']:.4f}")

        if args.judge:
            print("  Juge LLM ...")
            items = [{"question": q, "reference": r, "prediction": pr}
                     for q, r, pr in zip(questions, refs, preds, strict=True)]
            scores = judge_via_api(items)
            backend = "api"
            if scores is None:
                scores = judge_via_local(items)
                backend = "local (faible)"
            res["judge"] = {"backend": backend, **summarize_judge(scores),
                            "per_example": scores}
            j = res["judge"]
            print(f"    exactitude={j['exactitude']} pertinence={j['pertinence']} "
                  f"clarte={j['clarte']}")
        results[name] = res

    # --- Rapport ---
    print("\n" + "=" * 72)
    print("EVALUATION SEMANTIQUE")
    print("=" * 72)
    header = f"{'modele':<14}{'BERTScore F1':>14}"
    if args.judge:
        header += f"{'exactitude':>13}{'pertinence':>13}{'clarte':>10}"
    print(header)
    for name, r in results.items():
        line = f"{name:<14}{r['bertscore_f1']:>14.4f}"
        if args.judge and "judge" in r:
            j = r["judge"]
            line += (f"{str(j['exactitude']):>13}{str(j['pertinence']):>13}"
                     f"{str(j['clarte']):>10}")
        print(line)

    out = pred_file.parent / f"semantic_{pred_file.stem}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nEcrit : {out}")


if __name__ == "__main__":
    main()
