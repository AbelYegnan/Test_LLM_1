# src/label_generator.py

import re
import numpy as np
from typing import List


def keyword_label(response: str, indicator: str, context: str) -> int:
    if not response or len(response.strip()) < 10:
        return 0
    has_numbers = bool(re.search(r"\d+([.,]\d+)?", response))
    keywords = set(
        indicator.lower()
        .replace("-", " ").replace("/", " ")
        .replace("(", " ").replace(")", " ")
        .split()
    )
    stopwords = {"de", "du", "le", "la", "les", "en", "à", "au", "des", "un", "une", "l"}
    keywords -= stopwords
    response_lower = response.lower()
    keyword_matches = sum(1 for kw in keywords if kw in response_lower)
    if has_numbers and keyword_matches >= 1:
        return 1
    if keyword_matches >= 2:
        return 1
    return 0


def generate_run_labels(
    scores: List[float],
    noise_level: float = 0.15,
    positive_rate: float = 0.6,
    seed: int = 42,
) -> List[int]:
    rng = np.random.RandomState(seed)
    labels = []
    for score in scores:
        prob_positive = positive_rate * score + (1 - positive_rate) * (1 - score)
        prob_positive = np.clip(prob_positive, 0.1, 0.9)
        label = int(rng.random() < prob_positive)
        if rng.random() < noise_level:
            label = 1 - label
        labels.append(label)
    return labels


def generate_multi_run_scores(
    base_score: float,
    n_runs: int = 30,
    noise_std: float = 0.12,
    seed: int = 0,
) -> List[float]:
    rng = np.random.RandomState(seed)
    raw = rng.normal(loc=base_score, scale=noise_std, size=n_runs)
    return list(np.clip(raw, 0.01, 0.99).tolist())