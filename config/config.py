# config/config.py
# ============================================================
# Configuration centrale du pipeline d'évaluation LLM
# ============================================================

# ------------------------------------------------------------------
# LLMs open source recommandés (faciles à lancer en local via HF)
# ------------------------------------------------------------------
# 1. Mistral-7B-Instruct-v0.2  → Excellent rapport qualité/vitesse
# 2. Llama-3-8B-Instruct       → Meta, très bon sur les tâches de raisonnement
# 3. Phi-3-mini-4k-Instruct    → Microsoft, léger (3.8B), tourne sur CPU modeste
#
# Pour le test local sans GPU puissant, Phi-3-mini est recommandé.
# Avec un GPU 16 Go+, préférer Mistral-7B ou Llama-3-8B.
# ------------------------------------------------------------------

LLMS = {
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "label": "Mistral-7B-Instruct",
        "load_in_4bit": False,
    },
    "qwen2-1.5b": {
        "hf_id": "Qwen/Qwen2-1.5B-Instruct",
        "label": "Qwen2-1.5B-Instruct",
        "load_in_4bit": False,
    },
    "qwen2-0.5b": {
        "hf_id": "Qwen/Qwen2-0.5B-Instruct",
        "label": "Qwen2-0.5B-Instruct",
        "load_in_4bit": False,
    },
}

# ------------------------------------------------------------------
# Méthodes d'évaluation
# ------------------------------------------------------------------
# EPR  : Entropy of log-Probabilities (moyenne d'entropie des log-probs)
# WPR  : Weighted Probability scoring (EPR supervisé / pondéré)
# SELF : Auto-évaluation — le LLM génère sa réponse + son score de confiance
# ------------------------------------------------------------------
METHODS = ["EPR", "WPR", "SELF"]

# ------------------------------------------------------------------
# Paramètres de génération
# ------------------------------------------------------------------
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1       # Basse temp → réponses déterministes
TOP_P = 0.9
CHUNK_SIZE = 1500       # Nombre de caractères par chunk lors du découpage
CHUNK_OVERLAP = 150     # Chevauchement pour ne pas couper un contexte

# ------------------------------------------------------------------
# Documents & indicateurs
# ------------------------------------------------------------------
# Ajoutez ici les chemins vers vos PDFs et les 20 indicateurs de chaque doc.
# Les indicateurs sont les labels courts utilisés pour les titres de graphes.

DOCUMENTS = {
    "doc1_test": {
        "pdf_path": "data/document 1.pdf",
        "label": "Rapport Médical - Diabète T2",
        "indicators": [
            "Glycémie à jeun",
            "HbA1c",
            "IMC",
            "Pression artérielle systolique",
            "LDL-cholestérol",
            "Créatininémie",
            "Albuminurie",
            "Score SCORE2",
            "Fréquence cardiaque",
            "Score Morisky",
            "Acuité visuelle (LogMAR)",
            "Score DN4 neuropathie",
            "Hémoglobine",
            "Vitesse de conduction nerveuse",
            "Durée de sommeil",
            "Score GAD-7 anxiété",
            "Activité physique hebdomadaire",
            "Vitamine D sérique",
            "Ferritine sérique",
            "Qualité de vie SF-36",
        ],
    },
    "doc2_test": {
        "pdf_path": "data/document 4.pdf",
        "label": "Rapport Médical - Diabète T2",
        "indicators": [
            "Glycémie à jeun",
            "HbA1c",
            "IMC",
            "Pression artérielle systolique",
            "LDL-cholestérol",
            "Créatininémie",
            "Albuminurie",
            "Score SCORE2",
            "Fréquence cardiaque",
            "Score Morisky",
            "Acuité visuelle (LogMAR)",
            "Score DN4 neuropathie",
            "Hémoglobine",
            "Vitesse de conduction nerveuse",
            "Durée de sommeil",
            "Score GAD-7 anxiété",
            "Activité physique hebdomadaire",
            "Vitamine D sérique",
            "Ferritine sérique",
            "Qualité de vie SF-36",
        ],
    },
    "doc3_test": {
        "pdf_path": "data/document 5.pdf",
        "label": "Rapport Médical - Diabète T2",
        "indicators": [
            "Glycémie à jeun",
            "HbA1c",
            "IMC",
            "Pression artérielle systolique",
            "LDL-cholestérol",
            "Créatininémie",
            "Albuminurie",
            "Score SCORE2",
            "Fréquence cardiaque",
            "Score Morisky",
            "Acuité visuelle (LogMAR)",
            "Score DN4 neuropathie",
            "Hémoglobine",
            "Vitesse de conduction nerveuse",
            "Durée de sommeil",
            "Score GAD-7 anxiété",
            "Activité physique hebdomadaire",
            "Vitamine D sérique",
            "Ferritine sérique",
            "Qualité de vie SF-36",
        ],
    },
}

# ------------------------------------------------------------------
# Paramètres de sortie
# ------------------------------------------------------------------
OUTPUT_DIR = "outputs"
FIGURES_SUBDIR = "figures"   # sous-dossier pour les ROC par document
