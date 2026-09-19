"""Configuration centrale du projet (chemins, paramètres par défaut)."""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

DAILY_KPI_CSV = os.path.join(DATA_DIR, "daily_kpi.csv")
FLEET_CSV = os.path.join(DATA_DIR, "fleet_equipment.csv")
EQUIPMENT_DAILY_CSV = os.path.join(DATA_DIR, "equipment_daily.csv")

STAGES = ["forage", "sautage", "chargement", "transport", "energie"]

STAGE_LABELS = {
    "forage": "Forage",
    "sautage": "Sautage",
    "chargement": "Chargement",
    "transport": "Transport",
    "energie": "Énergie",
}

COST_COLUMNS = {
    "forage": "cout_forage_usd_t",
    "sautage": "cout_sautage_usd_t",
    "chargement": "cout_chargement_usd_t",
    "transport": "cout_transport_usd_t",
    "energie": "cout_energie_usd_t",
}

# Débit indicatif (tonnes-équivalent/heure) par étape, utilisé par le module
# d'optimisation pour convertir des heures d'exploitation en tonnage traité.
# Valeurs stylisées, calibrées pour que la capacité de flotte simulée
# corresponde à la production quotidienne moyenne du jeu de données — ce ne
# sont pas des ratios d'ingénierie minière réels (voir README, limites).
STAGE_THROUGHPUT_T_PER_H = {
    "forage": 165,
    "chargement": 130,
    "transport": 54,
    "energie": 230,  # rarement la contrainte limitante, comme en réalité
}

# Le sautage (explosifs/tir) n'est pas modélisé comme une flotte d'équipement
# avec des heures d'exploitation : c'est traité en dehors de l'optimisation
# flotte propre / sous-traitance, comme un coût variable au tonnage (voir
# src/optimization.py).
OPT_STAGES = ["forage", "chargement", "transport", "energie"]
