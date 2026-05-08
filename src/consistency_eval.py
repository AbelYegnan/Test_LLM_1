# src/consistency_eval.py
# ============================================================
# Résultat 1 : évaluation de cohérence sans jugement externe
# Pour chaque indicateur, on compte combien de fois sur les N
# documents le LLM a fourni une réponse jugée correcte.
# ============================================================

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def evaluate_response_correctness(response: str, indicator: str) -> bool:
    """
    Juge si une réponse est correcte via heuristique simple :
    - Contient au moins un chiffre
    - Contient au moins un mot-clé de l'indicateur
    """
    import re
    if not response or len(response.strip()) < 10:
        return False

    has_numbers = bool(re.search(r"\d+([.,]\d+)?", response))
    keywords = set(
        indicator.lower()
        .replace("-", " ").replace("/", " ")
        .replace("(", " ").replace(")", " ")
        .split()
    )
    stopwords = {"de", "du", "le", "la", "les", "en", "à", "au",
                 "des", "un", "une", "l", "par", "sur"}
    keywords -= stopwords
    response_lower = response.lower()
    keyword_hits = sum(1 for kw in keywords if kw in response_lower)

    return has_numbers and keyword_hits >= 1


def compute_consistency(
    responses_per_doc: Dict[str, Dict[str, str]],
    indicators: List[str],
) -> Dict[str, Dict]:
    """
    Calcule pour chaque indicateur le nombre de documents
    où la réponse est correcte.

    Parameters
    ----------
    responses_per_doc : {doc_key: {indicator: response_text}}
    indicators : liste des indicateurs

    Retourne :
    {indicator: {"correct": int, "total": int, "pct": float,
                 "details": {doc_key: bool}}}
    """
    results = {}
    n_docs = len(responses_per_doc)

    for indicator in indicators:
        correct_count = 0
        details = {}
        for doc_key, responses in responses_per_doc.items():
            response = responses.get(indicator, "")
            is_correct = evaluate_response_correctness(response, indicator)
            details[doc_key] = is_correct
            if is_correct:
                correct_count += 1

        results[indicator] = {
            "correct": correct_count,
            "total": n_docs,
            "pct": round(100 * correct_count / n_docs, 1) if n_docs > 0 else 0.0,
            "details": details,
        }

    return results


def save_consistency_report(
    consistency: Dict[str, Dict],
    llm_label: str,
    output_dir: str,
    method: str = "SELF",
) -> None:
    """
    Sauvegarde le rapport de cohérence en TXT et JSON.
    """
    out_path = Path(output_dir) / "consistency"
    out_path.mkdir(parents=True, exist_ok=True)

    # --- TXT lisible ---
    txt_path = out_path / f"consistency_{llm_label}_{method}.txt"
    lines = [
        f"Rapport de cohérence — LLM : {llm_label} | Méthode : {method}",
        "=" * 60,
        f"{'Indicateur':<40} {'Correct/Total':>15} {'%':>8}",
        "-" * 60,
    ]
    for indicator, data in consistency.items():
        line = (
            f"{indicator:<40} "
            f"{data['correct']}/{data['total']:>12} "
            f"{data['pct']:>7.1f}%"
        )
        lines.append(line)

    lines.append("-" * 60)
    mean_pct = np.mean([d["pct"] for d in consistency.values()])
    lines.append(f"{'MOYENNE GLOBALE':<40} {mean_pct:>22.1f}%")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    Cohérence TXT sauvegardée : {txt_path}")

    # --- JSON complet ---
    json_path = out_path / f"consistency_{llm_label}_{method}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(consistency, f, ensure_ascii=False, indent=2)
    print(f"    Cohérence JSON sauvegardée : {json_path}")


def print_consistency_summary(consistency: Dict[str, Dict], llm_label: str) -> None:
    """Affiche un résumé dans le terminal."""
    print(f"\n  Résultat sans jugement — {llm_label}")
    print(f"  {'Indicateur':<40} {'Résultat':>20}")
    print(f"  {'-'*62}")
    for indicator, data in consistency.items():
        print(
            f"  {indicator:<40} "
            f"{data['correct']}/{data['total']} documents "
            f"({data['pct']}%)"
        )