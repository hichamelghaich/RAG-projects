"""
Génération de données synthétiques — Optimiseur de coûts opérationnels miniers.

⚠️ IMPORTANT : toutes les données produites par ce script sont 100% FICTIVES.
Elles simulent la structure de coûts et de disponibilité d'une chaîne
opérationnelle minière (forage -> sautage -> chargement -> transport ->
énergie) à des fins de démonstration technique uniquement. Aucune donnée
réelle d'aucun site, équipement ou entreprise n'est utilisée ou représentée.

Sortie :
    data/daily_kpi.csv           -> KPI quotidiens (production, coûts par
                                     étape, disponibilité flotte, HSE)
    data/fleet_equipment.csv     -> référentiel des équipements simulés
    data/equipment_daily.csv     -> panel quotidien par équipement
                                     (heures opérées, disponibilité)
"""

import csv
import os
import random
from datetime import date, timedelta

random.seed(7)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAILY_KPI_PATH = os.path.join(BASE_DIR, "daily_kpi.csv")
FLEET_PATH = os.path.join(BASE_DIR, "fleet_equipment.csv")
EQUIPMENT_DAILY_PATH = os.path.join(BASE_DIR, "equipment_daily.csv")

N_DAYS = 540  # ~18 mois
START_DATE = date(2024, 1, 1)

STAGES = ["forage", "sautage", "chargement", "transport", "energie"]

# coût de base (USD/tonne) et variabilité par étape de la chaîne opérationnelle
STAGE_BASE_COST = {
    "forage": 3.8,
    "sautage": 2.6,
    "chargement": 4.5,
    "transport": 7.2,
    "energie": 2.1,
}

BASE_TONNAGE = 9500  # tonnes/jour


def daterange(n_days, start):
    return [start + timedelta(days=i) for i in range(n_days)]


# --- 1. Flotte d'équipements simulée ---

FLEET_SPEC = [
    ("FOR", "Foreuse", "forage", 4),
    ("PEL", "Pelle hydraulique", "chargement", 5),
    ("CAM", "Camion minier", "transport", 12),
    ("GEN", "Groupe électrogène", "energie", 3),
]


def build_fleet():
    fleet = []
    eq_id = 1
    for prefix, type_name, stage, count in FLEET_SPEC:
        for i in range(count):
            commission_offset = random.randint(0, 900)  # âge variable (jours avant le début de la période)
            fleet.append({
                "equipment_id": f"{prefix}-{i+1:02d}",
                "type": type_name,
                "etape": stage,
                "date_mise_en_service": (START_DATE - timedelta(days=commission_offset)).isoformat(),
            })
            eq_id += 1
    return fleet


def write_fleet_csv(fleet):
    with open(FLEET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fleet[0].keys()))
        writer.writeheader()
        writer.writerows(fleet)
    print(f"[ok] {FLEET_PATH} ({len(fleet)} équipements)")


# --- 2. Panel quotidien par équipement (heures opérées, disponibilité) ---
#
# Chaque équipement a une probabilité quotidienne d'indisponibilité (arrêt
# opérationnel : incident, aléa logistique, immobilisation ponctuelle...).
# C'est uniquement ce qui fait varier la disponibilité de flotte et, via
# elle, le coût par tonne — au même titre qu'un aléa météo ou logistique
# dans la réalité.

STAGE_DOWNTIME_RATE = {
    "forage": 0.045,
    "chargement": 0.035,
    "transport": 0.035,
    "energie": 0.02,
}


def build_equipment_daily(fleet, days):
    rows = []
    for d in days:
        seasonal_bump = 0.015 if d.month in (7, 8) else 0.0  # aléas plus fréquents en été (chaleur, effectifs réduits)
        for eq in fleet:
            stage = eq["etape"]
            downtime_rate = STAGE_DOWNTIME_RATE[stage] + seasonal_bump
            indisponible = 1 if random.random() < downtime_rate else 0

            if indisponible:
                heures_operees = 0.0
            else:
                heures_operees = max(0.0, random.gauss(18, 3)) if stage != "energie" else max(0.0, random.gauss(22, 2))
                heures_operees = min(24.0, heures_operees)

            rows.append({
                "date": d.isoformat(),
                "equipment_id": eq["equipment_id"],
                "etape": stage,
                "heures_operees": round(heures_operees, 2),
                "disponible": 0 if indisponible else 1,
            })

    return rows


def write_equipment_daily_csv(rows):
    fieldnames = list(rows[0].keys())
    with open(EQUIPMENT_DAILY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {EQUIPMENT_DAILY_PATH} ({len(rows)} lignes)")


# --- 3. KPI quotidiens agrégés (production, coûts par étape, disponibilité, HSE) ---

def build_daily_kpi(days, equipment_rows, fleet):
    from collections import defaultdict
    eq_by_date = defaultdict(list)
    for r in equipment_rows:
        eq_by_date[r["date"]].append(r)

    fleet_by_stage = defaultdict(list)
    for eq in fleet:
        fleet_by_stage[eq["etape"]].append(eq["equipment_id"])

    rows = []
    for i, d in enumerate(days):
        date_str = d.isoformat()
        day_eq_rows = eq_by_date[date_str]

        # saisonnalité + légère tendance d'amélioration de la production dans le temps
        seasonality = 1.0 + 0.06 * ((d.month in (10, 11, 12)) - (d.month in (7, 8)))
        trend = 1.0 + 0.0006 * i
        tonnes = BASE_TONNAGE * seasonality * trend * random.uniform(0.95, 1.05)

        cout_par_etape = {}
        dispo_par_etape = {}
        for stage in STAGES:
            stage_eq_rows = [r for r in day_eq_rows if r["etape"] == stage]
            total_eq = len(fleet_by_stage[stage])
            indisponibles = sum(1 for r in stage_eq_rows if r["disponible"] == 0)
            dispo = 1 - (indisponibles / total_eq if total_eq else 0)
            dispo_par_etape[stage] = dispo

            # coût par tonne de l'étape : hausse si disponibilité faible (recours à des
            # solutions de secours plus coûteuses) + bruit
            base = STAGE_BASE_COST[stage]
            cost = base * (1 + 0.35 * (1 - dispo)) * random.uniform(0.93, 1.07)
            cout_par_etape[stage] = round(cost, 3)

        cout_total = round(sum(cout_par_etape.values()), 3)
        dispo_flotte_globale = round(100 * sum(dispo_par_etape.values()) / len(STAGES), 1)

        incidents_hse = max(0, round(random.gauss(0.35, 0.6)))
        if dispo_flotte_globale < 80:
            incidents_hse += random.choice([0, 1])  # dégradation opérationnelle -> risque HSE accru

        rows.append({
            "date": date_str,
            "tonnes_produites": round(tonnes),
            "cout_forage_usd_t": cout_par_etape["forage"],
            "cout_sautage_usd_t": cout_par_etape["sautage"],
            "cout_chargement_usd_t": cout_par_etape["chargement"],
            "cout_transport_usd_t": cout_par_etape["transport"],
            "cout_energie_usd_t": cout_par_etape["energie"],
            "cout_total_usd_t": cout_total,
            "disponibilite_flotte_pct": dispo_flotte_globale,
            "incidents_hse": incidents_hse,
        })

    return rows


def write_daily_kpi_csv(rows):
    fieldnames = list(rows[0].keys())
    with open(DAILY_KPI_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {DAILY_KPI_PATH} ({len(rows)} lignes)")


def main():
    days = daterange(N_DAYS, START_DATE)
    fleet = build_fleet()
    write_fleet_csv(fleet)

    equipment_rows = build_equipment_daily(fleet, days)
    write_equipment_daily_csv(equipment_rows)

    daily_kpi_rows = build_daily_kpi(days, equipment_rows, fleet)
    write_daily_kpi_csv(daily_kpi_rows)


if __name__ == "__main__":
    main()
