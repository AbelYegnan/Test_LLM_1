# src/roc_plotter.py
# ============================================================
# Tracé des courbes ROC et calcul des AUC
# ============================================================
# Contexte :
#   Pour chaque indicateur et chaque (LLM, méthode), on dispose
#   d'un score de confiance (float [0,1]) et d'une étiquette binaire
#   (1 = information correctement extraite, 0 = extraction incorrecte
#    ou information absente).
#
# La courbe ROC représente TPR vs FPR à différents seuils de score.
# L'AUC (Area Under Curve) résume la performance en un seul chiffre.
# ============================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")   # mode non-interactif pour sauvegarde PNG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple


# Palette de couleurs et styles par méthode
METHOD_STYLES = {
    "EPR":  {"color": "#2196F3", "linestyle": "-",  "marker": "o"},
    "WPR":  {"color": "#FF5722", "linestyle": "--", "marker": "s"},
    "SELF": {"color": "#4CAF50", "linestyle": "-.", "marker": "^"},
}

LLM_COLORS = [
    "#1A237E",   # bleu marine
    "#B71C1C",   # rouge foncé
    "#1B5E20",   # vert foncé
    "#4A148C",   # violet foncé
    "#E65100",   # orange foncé
]


def plot_roc_for_indicator(
    indicator: str,
    scores_dict: Dict[str, Dict[str, Tuple[List[float], List[int]]]],
    output_path: str,
    doc_label: str = "",
) -> None:
    """
    Trace une figure avec les courbes ROC pour UN indicateur.
    
    Parameters
    ----------
    indicator : str
        Nom de l'indicateur (titre du graphe).
    scores_dict : dict
        Structure : {llm_label: {method: (scores_list, labels_list)}}
        ex: {"Mistral-7B": {"EPR": ([0.8, 0.3, ...], [1, 0, ...]), ...}}
    output_path : str
        Chemin de sauvegarde du PNG.
    doc_label : str
        Nom du document pour le titre.
    """
    # Une colonne par LLM
    n_llms = len(scores_dict)
    fig, axes = plt.subplots(
        1, n_llms,
        figsize=(6 * n_llms, 5),
        squeeze=False,
    )

    fig.suptitle(
        f"Courbes ROC — {doc_label}\nIndicateur : {indicator}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for col_idx, (llm_label, methods_data) in enumerate(scores_dict.items()):
        ax = axes[0][col_idx]
        ax.set_title(llm_label, fontsize=11, fontweight="bold")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Aléatoire (AUC=0.50)")

        for method, (scores, labels) in methods_data.items():
            style = METHOD_STYLES.get(method, {"color": "gray", "linestyle": "-"})

            labels_arr = np.array(labels)
            scores_arr = np.array(scores)

            # Cas dégénéré : une seule classe présente
            if len(np.unique(labels_arr)) < 2:
                ax.text(
                    0.5, 0.5,
                    f"{method}\n(données insuffisantes)",
                    ha="center", va="center",
                    fontsize=9, color=style["color"],
                    transform=ax.transAxes,
                )
                continue

            fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr, tpr,
                color=style["color"],
                linestyle=style["linestyle"],
                lw=2,
                label=f"{method} (AUC={roc_auc:.3f})",
            )

        ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=10)
        ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=10)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    💾 Sauvegardé : {output_path}")


def plot_summary_auc(
    all_results: Dict,
    output_path: str,
) -> None:
    """
    Trace un graphe de synthèse : AUC moyen par (LLM, méthode, document).
    
    Parameters
    ----------
    all_results : dict
        Structure complète des résultats.
    output_path : str
        Chemin de sauvegarde.
    """
    # Agréger les AUC moyens par (doc, llm, method)
    summary = {}  # {doc_label: {llm_label: {method: [auc_values]}}}

    for doc_key, doc_data in all_results.items():
        doc_label = doc_data["label"]
        summary[doc_label] = {}
        for llm_label, llm_data in doc_data["llm_results"].items():
            summary[doc_label][llm_label] = {}
            for method in llm_data:
                aucs = [
                    llm_data[method]["auc_per_indicator"].get(ind, np.nan)
                    for ind in doc_data["indicators"]
                ]
                valid_aucs = [a for a in aucs if not np.isnan(a)]
                mean_auc = np.mean(valid_aucs) if valid_aucs else np.nan
                summary[doc_label][llm_label][method] = mean_auc

    n_docs = len(summary)
    fig, axes = plt.subplots(1, n_docs, figsize=(7 * n_docs, 5), squeeze=False)
    fig.suptitle("AUC Moyen par Document, LLM et Méthode", fontsize=14, fontweight="bold")

    methods = ["EPR", "WPR", "SELF"]
    x = np.arange(len(methods))
    bar_width = 0.2

    for col_idx, (doc_label, llm_dict) in enumerate(summary.items()):
        ax = axes[0][col_idx]
        ax.set_title(doc_label, fontsize=10)

        llm_labels = list(llm_dict.keys())
        for i, (llm_label, method_dict) in enumerate(llm_dict.items()):
            auc_values = [method_dict.get(m, 0) for m in methods]
            offset = (i - len(llm_labels) / 2) * bar_width + bar_width / 2
            bars = ax.bar(
                x + offset,
                auc_values,
                bar_width,
                label=llm_label,
                color=LLM_COLORS[i % len(LLM_COLORS)],
                alpha=0.8,
            )
            # Afficher la valeur sur chaque barre
            for bar, val in zip(bars, auc_values):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel("AUC Moyen")
        ax.set_ylim([0, 1.15])
        ax.axhline(y=0.5, color="gray", linestyle="--", lw=1, alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Synthèse AUC sauvegardée : {output_path}")
