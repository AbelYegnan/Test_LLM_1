# src/confusion_table.py
# ============================================================
# Résultat 2 : tableau de confusion par seuil
# Pour chaque indicateur et chaque seuil, calcule TN, FP, FN, TP
# et sauvegarde en CSV + affichage.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple


THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
              0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
              0.9, 0.95]


def compute_confusion_at_threshold(
    scores: List[float],
    labels: List[int],
    threshold: float,
) -> Dict[str, int]:
    """
    Calcule TN, FP, FN, TP pour un seuil donné.

    Prédiction positive si score >= threshold.
    """
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    preds = (scores_arr >= threshold).astype(int)

    TP = int(np.sum((preds == 1) & (labels_arr == 1)))
    TN = int(np.sum((preds == 0) & (labels_arr == 0)))
    FP = int(np.sum((preds == 1) & (labels_arr == 0)))
    FN = int(np.sum((preds == 0) & (labels_arr == 1)))

    return {"TN": TN, "FP": FP, "FN": FN, "TP": TP}


def build_confusion_table(
    scores: List[float],
    labels: List[int],
    thresholds: List[float] = THRESHOLDS,
) -> pd.DataFrame:
    """
    Construit le tableau complet seuil → TN, FP, FN, TP.
    Retourne un DataFrame pandas.
    """
    rows = []
    for t in thresholds:
        conf = compute_confusion_at_threshold(scores, labels, t)
        rows.append({"seuil": t, **conf})
    return pd.DataFrame(rows)


def save_confusion_tables(
    scores_dict: Dict[str, Dict[str, Tuple[List[float], List[int]]]],
    indicators: List[str],
    output_dir: str,
    doc_label: str,
) -> None:
    """
    Sauvegarde les tableaux de confusion pour tous les indicateurs.

    Parameters
    ----------
    scores_dict : {llm_label: {method: {indicator: (scores, labels)}}}
    indicators  : liste des indicateurs
    output_dir  : dossier de sortie
    doc_label   : nom du document
    """
    out_path = Path(output_dir) / "confusion_tables" / doc_label.replace(" ", "_")
    out_path.mkdir(parents=True, exist_ok=True)

    for llm_label, methods_data in scores_dict.items():
        for method, indicator_data in methods_data.items():
            for indicator, (scores, labels) in indicator_data.items():
                df = build_confusion_table(scores, labels)

                # Nom de fichier sécurisé
                safe_ind = indicator.lower()
                for ch in " /()°%²³éèêëàâùûüîïôç'\"":
                    safe_ind = safe_ind.replace(ch, "_")
                safe_ind = safe_ind[:40]

                csv_path = out_path / f"{llm_label}_{method}_{safe_ind}.csv"
                df.to_csv(csv_path, index=False)

    print(f"    Tableaux de confusion sauvegardés : {out_path}")


def plot_confusion_table(
    scores: List[float],
    labels: List[int],
    indicator: str,
    llm_label: str,
    method: str,
    output_path: str,
) -> None:
    """
    Génère une figure PNG du tableau de confusion (style tableau visuel).
    """
    df = build_confusion_table(scores, labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    table_data = [df.columns.tolist()] + df.values.tolist()
    col_labels = ["Seuil", "TN", "FP", "FN", "TP"]

    table = ax.table(
        cellText=df.values.tolist(),
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    # Colorer l'en-tête
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1A237E")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colorer les lignes alternées
    for i in range(1, len(df) + 1):
        color = "#E8EAF6" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    ax.set_title(
        f"Tableau de confusion — {indicator}\n{llm_label} | {method}",
        fontsize=12, fontweight="bold", pad=20,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)