# src/scoring_methods.py
# ============================================================
# Trois méthodes de scoring pour évaluer la qualité d'une
# réponse LLM sur un indicateur extrait d'un document.
#
# EPR  : Entropy of log-Probabilities (moyenne des log-probs)
#        → Plus le score est élevé (moins négatif), plus le LLM
#          est "confiant" dans sa réponse. Score sans supervision.
#
# WPR  : Weighted Probability scoring (EPR pondéré/supervisé)
#        → Comme EPR mais pondéré par la position du token :
#          les premiers tokens (souvent les plus informatifs)
#          reçoivent un poids plus élevé.
#
# SELF : Auto-évaluation — le LLM produit une réponse ET un score
#        de confiance de 0 à 1 qu'il juge lui-même.
# ============================================================

import re
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
import numpy as np
from typing import Tuple

from .llm_loader import LLMWrapper


# ------------------------------------------------------------------
# Prompts système partagés
# ------------------------------------------------------------------
SYSTEM_EXTRACTION = (
    "Tu es un expert en analyse de documents. "
    "Réponds de façon précise, factuelle et concise en français."
)

SYSTEM_SELF_EVAL = (
    "Tu es un expert en analyse de documents. "
    "Tu dois fournir une réponse et évaluer honnêtement ta propre confiance. "
    "Réponds UNIQUEMENT au format JSON strict : "
    '{"reponse": "<ta réponse>", "confiance": <float entre 0 et 1>}'
)


def build_extraction_prompt(indicator: str, context_chunks: list[str]) -> str:
    """
    Construit le prompt utilisateur pour extraire un indicateur depuis des chunks.
    """
    context = "\n\n---\n\n".join(context_chunks)
    return (
        f"Voici des extraits d'un document :\n\n{context}\n\n"
        f"Question : Quelle est la valeur ou l'information principale "
        f"concernant l'indicateur '{indicator}' dans ce document ? "
        f"Sois précis et cite les chiffres si disponibles."
    )


# ------------------------------------------------------------------
# Méthode 1 : EPR (Entropy of log-Probabilities)
# ------------------------------------------------------------------
def score_epr(
    llm: LLMWrapper,
    indicator: str,
    context_chunks: list[str],
) -> float:
    """
    Calcule le score EPR : moyenne des log-probabilités des tokens générés.
    
    Intuition : un LLM confiant assigne des probabilités élevées aux tokens
    qu'il choisit → log-probs proches de 0. Un LLM incertain → log-probs
    très négatives. On normalise en [0, 1] via sigmoid.

    Retourne un score en [0, 1].
    """
    user_prompt = build_extraction_prompt(indicator, context_chunks)

    try:
        log_probs = llm.get_log_probs(SYSTEM_EXTRACTION, user_prompt)
        if len(log_probs) == 0:
            return 0.5  # valeur neutre si pas de tokens

        mean_log_prob = log_probs.mean().item()

        # Normalisation : log-probs sont dans (-∞, 0].
        # On applique sigmoid(mean_log_prob * scale) pour mapper en (0, 0.5].
        # Scale empirique : 0.5 donne une bonne dispersion.
        score = float(torch.sigmoid(torch.tensor(mean_log_prob * 0.5)))
        return score

    except Exception as e:
        print(f"    ⚠️  EPR erreur pour '{indicator}': {e}")
        return 0.5


# ------------------------------------------------------------------
# Méthode 2 : WPR (Weighted Probability scoring — EPR supervisé)
# ------------------------------------------------------------------
def score_wpr(
    llm: LLMWrapper,
    indicator: str,
    context_chunks: list[str],
) -> float:
    """
    Calcule le score WPR : log-probs pondérées par position décroissante.
    
    Intuition : les premiers tokens d'une réponse (ex: le chiffre clé)
    sont souvent les plus informatifs. On leur donne plus de poids.
    
    Poids : w_i = 1 / (i + 1)  (décroissance harmonique)

    Retourne un score en [0, 1].
    """
    user_prompt = build_extraction_prompt(indicator, context_chunks)

    try:
        log_probs = llm.get_log_probs(SYSTEM_EXTRACTION, user_prompt)
        if len(log_probs) == 0:
            return 0.5

        n = len(log_probs)
        weights = torch.tensor([1.0 / (i + 1) for i in range(n)])
        weights = weights / weights.sum()  # normalisation

        weighted_mean = (log_probs * weights).sum().item()
        score = float(torch.sigmoid(torch.tensor(weighted_mean * 0.5)))
        return score

    except Exception as e:
        print(f"    ⚠️  WPR erreur pour '{indicator}': {e}")
        return 0.5


# ------------------------------------------------------------------
# Méthode 3 : SELF (Auto-évaluation LLM)
# ------------------------------------------------------------------
def score_self(
    llm: LLMWrapper,
    indicator: str,
    context_chunks: list[str],
) -> Tuple[float, str]:
    """
    Demande au LLM de générer sa réponse ET d'évaluer sa propre confiance.
    Le LLM retourne un JSON : {"reponse": "...", "confiance": 0.85}
    
    Retourne (score_confiance: float, reponse: str).
    """
    user_prompt = build_extraction_prompt(indicator, context_chunks)

    try:
        raw = llm.generate(
            system=SYSTEM_SELF_EVAL,
            user=user_prompt,
            max_new_tokens=256,
            temperature=0.1,
        )

        # Parser le JSON retourné
        score, response_text = _parse_self_eval_response(raw)
        return score, response_text

    except Exception as e:
        print(f"    ⚠️  SELF erreur pour '{indicator}': {e}")
        return 0.5, ""


def _parse_self_eval_response(raw: str) -> Tuple[float, str]:
    """
    Extrait la confiance et la réponse depuis le JSON brut du LLM.
    Tolère les formats imparfaits via regex fallback.
    """
    import json

    # Nettoyage : extraire le premier bloc JSON
    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            confiance = float(data.get("confiance", 0.5))
            confiance = max(0.0, min(1.0, confiance))  # clamp [0,1]
            reponse = str(data.get("reponse", ""))
            return confiance, reponse
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback : chercher un float après "confiance"
    conf_match = re.search(r"confiance[\":\s]+([0-9.]+)", raw, re.IGNORECASE)
    if conf_match:
        try:
            confiance = float(conf_match.group(1))
            confiance = max(0.0, min(1.0, confiance))
            return confiance, raw
        except ValueError:
            pass

    return 0.5, raw
