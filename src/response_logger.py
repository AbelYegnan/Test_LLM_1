# src/response_logger.py
# ============================================================
# Sauvegarde et affichage des réponses générées par le LLM
# pour chaque indicateur, chaque document et chaque méthode.
#
# Structure de all_responses :
# {
#   llm_label: {
#     doc_key: {
#       indicator: {
#         method: {
#           "score": float,
#           "reponse": str   ← vraie réponse textuelle du LLM
#         }
#       }
#     }
#   }
# }
# ============================================================

import json
from pathlib import Path
from typing import Dict


def save_responses(
    responses: Dict,
    output_dir: str,
) -> None:
    """
    Sauvegarde toutes les réponses LLM en JSON et TXT lisible.

    Pour chaque LLM → document → indicateur → méthode,
    on sauvegarde le score numérique ET la réponse textuelle.
    """
    out_path = Path(output_dir) / "responses"
    out_path.mkdir(parents=True, exist_ok=True)

    for llm_label, docs_data in responses.items():

        # ── JSON complet (utile pour analyse ultérieure) ──────────────
        json_path = out_path / f"responses_{llm_label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        # ── TXT lisible par un humain ─────────────────────────────────
        txt_path = out_path / f"responses_{llm_label}.txt"
        lines = [
            f"Réponses LLM — {llm_label}",
            "=" * 70,
        ]

        for doc_key, indicators_data in docs_data.items():
            lines.append(f"\n{'─'*70}")
            lines.append(f"  Document : {doc_key}")
            lines.append(f"{'─'*70}")

            for indicator, methods_data in indicators_data.items():
                lines.append(f"\n  Indicateur : {indicator}")
                lines.append(f"  {'-'*60}")

                for method, data in methods_data.items():
                    score = data.get("score", "N/A")
                    reponse = data.get("reponse", "(aucune réponse)")

                    # Tronquer si trop long pour le TXT
                    reponse_short = (
                        reponse[:400] + "..."
                        if len(reponse) > 400 else reponse
                    )

                    lines.append(f"    [{method}]")
                    lines.append(f"      Score   : {score}")
                    lines.append(f"      Réponse : {reponse_short}")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"    Réponses sauvegardées : {txt_path}")
        print(f"    Réponses JSON         : {json_path}")


def print_responses_for_indicator(
    responses: Dict,
    indicator: str,
) -> None:
    """
    Affiche dans le terminal toutes les réponses pour UN indicateur donné.

    Pour chaque LLM et chaque document, on voit :
      - Le score de chaque méthode
      - La réponse textuelle générée par le LLM
    """
    print(f"\n  {'='*65}")
    print(f"  Réponses pour l'indicateur : '{indicator}'")
    print(f"  {'='*65}")

    for llm_label, docs_data in responses.items():
        print(f"\n  LLM : {llm_label}")
        print(f"  {'─'*60}")

        for doc_key, indicators_data in docs_data.items():
            print(f"\n    Document : {doc_key}")

            methods_data = indicators_data.get(indicator, {})
            if not methods_data:
                print("      (aucune réponse enregistrée)")
                continue

            for method, data in methods_data.items():
                score = data.get("score", "N/A")
                reponse = data.get("reponse", "(aucune réponse)")
                reponse_short = (
                    reponse[:200] + "..."
                    if len(reponse) > 200 else reponse
                )
                print(f"\n      [{method}]")
                print(f"        Score   : {score}")
                print(f"        Réponse : {reponse_short}")


def print_all_responses_summary(responses: Dict) -> None:
    """
    Affiche un résumé global dans le terminal :
    pour chaque indicateur, combien de méthodes ont répondu
    et quel est le score moyen.
    """
    print(f"\n  {'='*65}")
    print(f"  Résumé global des réponses")
    print(f"  {'='*65}")

    for llm_label, docs_data in responses.items():
        print(f"\n  LLM : {llm_label}")

        for doc_key, indicators_data in docs_data.items():
            print(f"\n    Document : {doc_key}")
            print(f"    {'─'*55}")
            print(f"    {'Indicateur':<35} {'EPR':>8} {'WPR':>8} {'SELF':>8}")
            print(f"    {'─'*55}")

            for indicator, methods_data in indicators_data.items():
                epr_score  = methods_data.get("EPR",  {}).get("score", float("nan"))
                wpr_score  = methods_data.get("WPR",  {}).get("score", float("nan"))
                self_score = methods_data.get("SELF", {}).get("score", float("nan"))

                def fmt(v):
                    return f"{v:.3f}" if isinstance(v, float) and not __import__('math').isnan(v) else " N/A"

                ind_short = indicator[:34]
                print(f"    {ind_short:<35} {fmt(epr_score):>8} {fmt(wpr_score):>8} {fmt(self_score):>8}")