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
                                     (heures, maintenance, pannes)
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
                "maintenance_interval_heures": {
                    "forage": 250, "chargement": 300, "transport": 400, "energie": 500,
                }[stage],
            })
            eq_id += 1
    return fleet


def write_fleet_csv(fleet):
    with open(FLEET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fleet[0].keys()))
        writer.writeheader()
        writer.writerows(fleet)
    print(f"[ok] {FLEET_PATH} ({len(fleet)} équipements)")


# --- 2. Panel quotidien par équipement (heures, maintenance, pannes) ---

def build_equipment_daily(fleet, days):
    rows = []
    state = {
        eq["equipment_id"]: {"heures_cumulees": random.uniform(200, 3000),
                              "heures_depuis_maintenance": random.uniform(0, eq["maintenance_interval_heures"])}
        for eq in fleet
    }

    for d in days:
        for eq in fleet:
            eq_id = eq["equipment_id"]
            interval = eq["maintenance_interval_heures"]
            s = state[eq_id]

            # risque de panne = fonction de type "hasard croissant" (reliability
            # engineering) : croît fortement une fois l'intervalle de
            # maintenance nominal dépassé (ratio_usure > 1).
            usure_ratio = s["heures_depuis_maintenance"] / interval
            proba_panne = min(0.45, 0.004 + 0.05 * usure_ratio ** 3)
            panne = 1 if random.random() < proba_panne else 0

            # maintenance préventive programmée : une fois l'intervalle nominal
            # dépassé, une intervention a de bonnes chances d'être déclenchée
            # avant la panne -- mais pas toujours (retards opérationnels réalistes).
            preventive = (not panne) and usure_ratio >= 1.0 and random.random() < 0.20

            if panne or preventive:
                heures_operees = 0.0
                s["heures_depuis_maintenance"] = 0.0  # panne ou maintenance -> reset
            else:
                heures_operees = max(0.0, random.gauss(18, 3)) if eq["etape"] != "energie" else max(0.0, random.gauss(22, 2))
                heures_operees = min(24.0, heures_operees)
                s["heures_cumulees"] += heures_operees
                s["heures_depuis_maintenance"] += heures_operees

            rows.append({
                "date": d.isoformat(),
                "equipment_id": eq_id,
                "etape": eq["etape"],
                "heures_operees": round(heures_operees, 2),
                "heures_cumulees": round(s["heures_cumulees"], 1),
                "heures_depuis_maintenance": round(s["heures_depuis_maintenance"], 1),
                "ratio_usure": round(min(3.0, (s["heures_depuis_maintenance"] / interval)), 3),
                "panne": panne,
            })

    return rows


def add_forward_labels(rows, horizon_days=7):
    """Ajoute une étiquette binaire 'panne_sous_7j' = 1 si l'équipement tombe
    en panne dans les `horizon_days` jours suivants (panel triée par équipement/date)."""
    from collections import defaultdict
    by_eq = defaultdict(list)
    for r in rows:
        by_eq[r["equipment_id"]].append(r)

    for eq_id, eq_rows in by_eq.items():
        eq_rows.sort(key=lambda r: r["date"])
        n = len(eq_rows)
        for i in range(n):
            window = eq_rows[i+1:i+1+horizon_days]
            eq_rows[i]["panne_sous_7j"] = 1 if any(w["panne"] == 1 for w in window) else 0
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
            en_panne = sum(1 for r in stage_eq_rows if r["panne"] == 1)
            dispo = 1 - (en_panne / total_eq if total_eq else 0)
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
    equipment_rows = add_forward_labels(equipment_rows, horizon_days=7)
    write_equipment_daily_csv(equipment_rows)

    daily_kpi_rows = build_daily_kpi(days, equipment_rows, fleet)
    write_daily_kpi_csv(daily_kpi_rows)


if __name__ == "__main__":
    main()
