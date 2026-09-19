"""Configuration centrale du projet (chemins, paramètres par défaut)."""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
REFERENCE_DIR = os.path.join(DATA_DIR, "reference")
KPI_CSV_PATH = os.path.join(DATA_DIR, "kpi_timeseries.csv")
INDEX_DIR = os.path.join(DATA_DIR, "index")

# --- Chunking ---
CHUNK_SIZE_CHARS = 700
CHUNK_OVERLAP_CHARS = 120

# --- Retrieval ---
TOP_K = 4

# --- LLM backend ---
# "extractive" fonctionne sans aucune clé API (par défaut).
# "openai" utilise l'API OpenAI si OPENAI_API_KEY est définie dans l'environnement.
LLM_BACKEND = os.environ.get("RAG_LLM_BACKEND", "extractive")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# --- Forecast ---
FORECAST_METRICS = {
    "cout_par_tonne_usd": "coût par tonne (USD/tonne)",
    "volume_transporte_tonnes": "volume transporté (tonnes)",
    "delai_livraison_jours": "délai de livraison (jours)",
    "taux_disponibilite_flotte_pct": "taux de disponibilité de la flotte (%)",
    "incidents_hse": "incidents HSE (nombre)",
    "stock_intermediaire_tonnes": "stock intermédiaire (tonnes)",
}
DEFAULT_FORECAST_HORIZON = 3  # mois
