# main.py
# ============================================================
# Pipeline principal d'évaluation LLM sur documents PDF
# ============================================================

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, auc

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        items = list(iterable)
        for i, item in enumerate(items):
            print(f"  {desc} [{i+1}/{len(items)}] {item}")
            yield item

# Ajout du répertoire courant au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from config.config import LLMS, METHODS, DOCUMENTS, OUTPUT_DIR, FIGURES_SUBDIR
from config.config import MAX_NEW_TOKENS, TEMPERATURE, TOP_P, CHUNK_SIZE, CHUNK_OVERLAP

from src.pdf_reader import load_document, get_relevant_chunks
from src.scoring_methods import score_epr, score_wpr, score_self
from src.roc_plotter import plot_roc_for_indicator, plot_summary_auc
from src.label_generator import generate_multi_run_scores, generate_run_labels
from src.consistency_eval import (
    compute_consistency,
    save_consistency_report,
    print_consistency_summary,
)
from src.confusion_table import save_confusion_tables, plot_confusion_table
from src.response_logger import (
    save_responses,
    print_responses_for_indicator,
    print_all_responses_summary,
)


# ------------------------------------------------------------------
# Utilitaires
# ------------------------------------------------------------------

def safe_indicator_filename(indicator: str, idx: int) -> str:
    clean = indicator.lower()
    for char in " /()°%²³éèêëàâùûüîïôç'\"":
        clean = clean.replace(char, "_")
    clean = clean.replace("__", "_").strip("_")
    return f"indicator_{idx:02d}_{clean[:40]}"


def compute_roc_data(scores: list, labels: list) -> dict:
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    if len(np.unique(labels_arr)) < 2:
        return {"fpr": [], "tpr": [], "auc": float("nan")}
    fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
    roc_auc = auc(fpr, tpr)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(roc_auc)}


# ------------------------------------------------------------------
# Pipeline document
# ------------------------------------------------------------------

def run_document_pipeline(
    doc_key: str,
    doc_config: dict,
    llm_instances: dict,
    dry_run: bool = False,
    n_runs: int = 30,
) -> dict:
    indicators = doc_config["indicators"]
    doc_label = doc_config["label"]
    n_indicators = len(indicators)

    print(f"\n{'='*60}")
    print(f"Document : {doc_label}")
    print(f"   {n_indicators} indicateurs | {len(llm_instances)} LLMs | {len(METHODS)} méthodes")
    print(f"{'='*60}")

    # Chargement du document
    if not dry_run:
        doc_data = load_document(doc_config, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        chunks = doc_data["chunks"]
    else:
        print("  [DRY-RUN] Pas de PDF chargé, scores simulés.")
        chunks = ["chunk simulé pour le test"] * 5

    # Résultats par LLM
    llm_results = {
        llm_label: {
            m: {"scores": {}, "labels": {}, "auc_per_indicator": {}}
            for m in METHODS
        }
        for llm_label in llm_instances
    }

    # Dossier de sortie pour les figures
    fig_dir = Path(OUTPUT_DIR) / FIGURES_SUBDIR / doc_key
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Stockage des réponses
    # Structure : {llm_label: {doc_key: {indicator: {method: {score, reponse}}}}}
    all_responses = {
        llm_label: {doc_key: {}} for llm_label in llm_instances
    }

    # ------------------------------------------------------------------
    # Boucle principale : indicateur par indicateur
    # ------------------------------------------------------------------
    for ind_idx, indicator in enumerate(tqdm(indicators, desc=f"  Indicateurs [{doc_label[:20]}]")):

        if not dry_run:
            rel_chunks = get_relevant_chunks(chunks, indicator, top_k=3)
        else:
            rel_chunks = chunks[:3]

        indicator_scores_dict = {}

        for llm_label, llm in llm_instances.items():
            indicator_scores_dict[llm_label] = {}
            base_seed = ind_idx * 1000 + list(llm_instances.keys()).index(llm_label) * 100

            for method in METHODS:
                if dry_run:
                    # ---- Mode simulation ----
                    rng = np.random.RandomState(base_seed + METHODS.index(method))
                    base_score = float(rng.uniform(0.45, 0.85))
                    scores = generate_multi_run_scores(
                        base_score, n_runs=n_runs,
                        seed=base_seed + METHODS.index(method)
                    )
                    # Réponse simulée lisible
                    response_text = (
                        f"La valeur de '{indicator}' extraite du document "
                        f"est estimée à {round(base_score * 100, 1)} unités "
                        f"avec un score de confiance de {base_score:.3f}."
                    )

                else:
                    # ---- Mode réel : toutes les méthodes retournent (score, texte) ----
                    if method == "EPR":
                        base_score, response_text = score_epr(
                            llm, indicator, rel_chunks
                        )
                    elif method == "WPR":
                        base_score, response_text = score_wpr(
                            llm, indicator, rel_chunks
                        )
                    else:  # SELF
                        base_score, response_text = score_self(
                            llm, indicator, rel_chunks
                        )

                    scores = generate_multi_run_scores(
                        base_score, n_runs=n_runs,
                        seed=base_seed + METHODS.index(method)
                    )

                # ──────────────────────────────────────────────────────
                # REMPLACEMENT 1 : stocker score + réponse textuelle
                # pour TOUTES les méthodes (EPR, WPR, SELF)
                # ──────────────────────────────────────────────────────
                if indicator not in all_responses[llm_label][doc_key]:
                    all_responses[llm_label][doc_key][indicator] = {}

                all_responses[llm_label][doc_key][indicator][method] = {
                    "score": round(base_score, 4),
                    "reponse": response_text,
                }

                # Générer les labels simulés
                labels = generate_run_labels(
                    scores, seed=base_seed + METHODS.index(method)
                )

                # Calculer ROC & AUC
                roc_data = compute_roc_data(scores, labels)

                # Stocker dans llm_results
                llm_results[llm_label][method]["scores"][indicator] = scores
                llm_results[llm_label][method]["labels"][indicator] = labels
                llm_results[llm_label][method]["auc_per_indicator"][indicator] = roc_data["auc"]
                indicator_scores_dict[llm_label][method] = (scores, labels)

        # Tracer la courbe ROC pour cet indicateur
        fig_path = str(fig_dir / f"{safe_indicator_filename(indicator, ind_idx)}.png")
        plot_roc_for_indicator(
            indicator=indicator,
            scores_dict=indicator_scores_dict,
            output_path=fig_path,
            doc_label=doc_label,
        )

    # ------------------------------------------------------------------
    # Résultat 1 : cohérence sans jugement (X/N documents)
    # ------------------------------------------------------------------
    print(f"\n  Calcul de la cohérence sans jugement...")
    for llm_label in llm_instances:
        for method in METHODS:
            # Extraire uniquement les réponses textuelles pour la cohérence
            if dry_run:
                responses_per_doc = {
                    doc_key: {
                        ind: f"La valeur de {ind} est 12.5 unités selon le document."
                        for ind in indicators
                    }
                }
            else:
                responses_per_doc = {
                    doc_key: {
                        ind: all_responses[llm_label][doc_key][ind][method]["reponse"]
                        for ind in indicators
                        if ind in all_responses[llm_label][doc_key]
                    }
                }
            consistency = compute_consistency(responses_per_doc, indicators)
            save_consistency_report(consistency, llm_label, OUTPUT_DIR, method)

        # Afficher résumé une fois par LLM
        print_consistency_summary(consistency, llm_label)

    # ------------------------------------------------------------------
    # Résultat 2 : tableaux de confusion par seuil (TN, FP, FN, TP)
    # ------------------------------------------------------------------
    print(f"\n  Génération des tableaux de confusion...")
    scores_for_confusion = {
        llm_label: {
            method: {
                ind: (
                    llm_results[llm_label][method]["scores"][ind],
                    llm_results[llm_label][method]["labels"][ind],
                )
                for ind in indicators
            }
            for method in METHODS
        }
        for llm_label in llm_instances
    }
    save_confusion_tables(
        scores_dict=scores_for_confusion,
        indicators=indicators,
        output_dir=OUTPUT_DIR,
        doc_label=doc_label,
    )

    # Figures PNG des tableaux de confusion
    conf_fig_dir = Path(OUTPUT_DIR) / "confusion_tables" / doc_label.replace(" ", "_")
    conf_fig_dir.mkdir(parents=True, exist_ok=True)
    for llm_label in llm_instances:
        for method in METHODS:
            for ind_idx, indicator in enumerate(indicators):
                scores = llm_results[llm_label][method]["scores"][indicator]
                labels = llm_results[llm_label][method]["labels"][indicator]
                safe_ind = safe_indicator_filename(indicator, ind_idx)
                fig_path = str(conf_fig_dir / f"{llm_label}_{method}_{safe_ind}.png")
                plot_confusion_table(
                    scores=scores,
                    labels=labels,
                    indicator=indicator,
                    llm_label=llm_label,
                    method=method,
                    output_path=fig_path,
                )

    # ------------------------------------------------------------------
    # Sauvegarde et affichage des réponses LLM
    # ------------------------------------------------------------------
    print(f"\n  Sauvegarde des réponses LLM...")
    save_responses(all_responses, OUTPUT_DIR)

    # Aperçu détaillé pour le premier indicateur
    print(f"\n  Aperçu des réponses pour l'indicateur '{indicators[0]}':")
    print_responses_for_indicator(all_responses, indicators[0])

    # ──────────────────────────────────────────────────────────────────
    # REMPLACEMENT 2 : tableau récapitulatif scores + réponses
    # pour tous les indicateurs
    # ──────────────────────────────────────────────────────────────────
    print_all_responses_summary(all_responses)

    return {
        "label": doc_label,
        "indicators": indicators,
        "llm_results": llm_results,
        "responses": all_responses,
    }


# ------------------------------------------------------------------
# Point d'entrée
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline d'évaluation LLM avec courbes ROC par indicateur."
    )
    parser.add_argument("--doc", type=str, default=None,
                        help="Clé du document (ex: document1). Défaut: tous.")
    parser.add_argument("--llm", type=str, default=None,
                        help="Clé du LLM (ex: phi3-mini). Défaut: tous.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mode test sans LLM réel (scores simulés).")
    parser.add_argument("--n-runs", type=int, default=30,
                        help="Nombre de points par courbe ROC (défaut: 30).")
    args = parser.parse_args()

    # Sélection des documents
    docs_to_run = {k: v for k, v in DOCUMENTS.items()
                   if args.doc is None or k == args.doc}
    if not docs_to_run:
        print(f"Document '{args.doc}' introuvable. Disponibles: {list(DOCUMENTS.keys())}")
        sys.exit(1)

    # Sélection des LLMs
    llms_to_run = {k: v for k, v in LLMS.items()
                   if args.llm is None or k == args.llm}
    if not llms_to_run:
        print(f"LLM '{args.llm}' introuvable. Disponibles: {list(LLMS.keys())}")
        sys.exit(1)

    # Chargement des LLMs (sauf dry-run)
    llm_instances = {}
    if not args.dry_run:
        print("\nChargement des LLMs...")
        from src.llm_loader import load_llm
        for llm_key, llm_cfg in llms_to_run.items():
            try:
                llm_instances[llm_cfg["label"]] = load_llm(llm_cfg)
            except Exception as e:
                print(f"  Impossible de charger {llm_key}: {e}")
        if not llm_instances:
            print("Aucun LLM chargé. Utilisez --dry-run pour tester sans LLM.")
            sys.exit(1)
    else:
        llm_instances = {cfg["label"]: None for cfg in llms_to_run.values()}

    # Créer les dossiers de sortie
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR, FIGURES_SUBDIR).mkdir(parents=True, exist_ok=True)

    # ---- Pipeline par document ----
    all_results = {}
    t_start = time.time()

    for doc_key, doc_config in docs_to_run.items():
        all_results[doc_key] = run_document_pipeline(
            doc_key=doc_key,
            doc_config=doc_config,
            llm_instances=llm_instances,
            dry_run=args.dry_run,
            n_runs=args.n_runs,
        )

    # ---- Synthèse globale ----
    print("\nGénération du graphe de synthèse AUC...")
    plot_summary_auc(
        all_results=all_results,
        output_path=str(Path(OUTPUT_DIR) / "summary_auc.png"),
    )

    # ---- Sauvegarde JSON des résultats ----
    results_json_path = Path(OUTPUT_DIR) / "results.json"
    serializable = {}
    for doc_key, doc_data in all_results.items():
        serializable[doc_key] = {
            "label": doc_data["label"],
            "indicators": doc_data["indicators"],
            "auc_summary": {
                llm_label: {
                    method: {
                        ind: round(auc_val, 4)
                        for ind, auc_val in method_data["auc_per_indicator"].items()
                    }
                    for method, method_data in llm_data.items()
                }
                for llm_label, llm_data in doc_data["llm_results"].items()
            }
        }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\nPipeline terminé en {elapsed:.1f}s")
    print(f"   Figures ROC      : {Path(OUTPUT_DIR) / FIGURES_SUBDIR}/")
    print(f"   Tableaux conf.   : {Path(OUTPUT_DIR) / 'confusion_tables'}/")
    print(f"   Cohérence        : {Path(OUTPUT_DIR) / 'consistency'}/")
    print(f"   Réponses LLM     : {Path(OUTPUT_DIR) / 'responses'}/")
    print(f"   Synthèse AUC     : {Path(OUTPUT_DIR) / 'summary_auc.png'}")
    print(f"   Résultats JSON   : {results_json_path}")


if __name__ == "__main__":
    main()