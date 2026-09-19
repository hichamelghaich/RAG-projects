"""
Génération de données synthétiques — Assistant RAG + Forecast.

⚠️ IMPORTANT : toutes les données produites par ce script sont 100% FICTIVES.
Elles simulent la structure d'indicateurs logistiques/miniers (coût/tonne,
volumes, délais, incidents HSE) à des fins de démonstration technique
uniquement. Aucune donnée réelle d'aucune entreprise n'est utilisée ou
représentée.

Sortie :
    data/kpi_timeseries.csv        -> série mensuelle de KPI (36 mois)
    data/reports/*.md              -> rapports mensuels + rapports d'incident
    data/reference/*.md            -> documents de référence (procédures)
"""

import csv
import os
import random
from datetime import date

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
REFERENCE_DIR = os.path.join(BASE_DIR, "reference")
CSV_PATH = os.path.join(BASE_DIR, "kpi_timeseries.csv")

N_MONTHS = 36
START_YEAR, START_MONTH = 2023, 1

CORRIDOR_NAME = "Corridor Logistique Sud (SIMULÉ)"
SITE_NAME = "Site Minier Fictif — Terminal Portuaire Fictif"

MONTHS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


def month_sequence(n_months, start_year, start_month):
    seq = []
    y, m = start_year, start_month
    for _ in range(n_months):
        seq.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return seq


def generate_kpi_series(months):
    """Génère des séries corrélées avec tendance, saisonnalité et bruit,
    plus quelques chocs ponctuels (incident, pic de coût) réutilisés
    dans les rapports texte pour que le RAG puisse "expliquer" les
    anomalies détectées par le module de forecast."""
    rows = []
    events = {}  # index -> description d'événement notable

    base_volume = 180_000  # tonnes/mois
    base_cost = 28.5  # USD/tonne
    base_delay = 3.2  # jours
    base_availability = 91.0  # %
    base_stock = 45_000  # tonnes

    # quelques événements notables prédéfinis (indices dans la série)
    incident_months = sorted(random.sample(range(3, N_MONTHS - 1), 6))
    cost_shock_months = sorted(random.sample(
        [i for i in range(N_MONTHS) if i not in incident_months], 4
    ))

    for i, d in enumerate(months):
        month_idx_in_year = d.month - 1
        # saisonnalité annuelle (pic d'activité en fin d'année, creux en été)
        seasonality = 1.0 + 0.08 * (
            (month_idx_in_year in (9, 10, 11)) - (month_idx_in_year in (6, 7))
        )
        trend = 1.0 + 0.0025 * i  # légère croissance du volume dans le temps

        volume = base_volume * seasonality * trend * random.uniform(0.96, 1.04)
        cost = base_cost * random.uniform(0.97, 1.03) - 0.01 * i  # gains d'efficience lents
        delay = base_delay * random.uniform(0.85, 1.15)
        availability = base_availability + random.uniform(-1.5, 1.5)
        hse_incidents = max(0, round(random.gauss(1.1, 0.9)))
        stock = base_stock * random.uniform(0.9, 1.1)

        note = ""
        if i in incident_months:
            hse_incidents += random.choice([2, 3])
            delay *= random.uniform(1.25, 1.6)
            availability -= random.uniform(3, 7)
            note = "incident_hse_majeur"
            events[i] = note
        if i in cost_shock_months:
            cost *= random.uniform(1.08, 1.18)
            note = (note + "+choc_cout") if note else "choc_cout"
            events[i] = note

        rows.append({
            "date": d.isoformat(),
            "annee": d.year,
            "mois": d.month,
            "mois_nom": MONTHS_FR[month_idx_in_year],
            "volume_transporte_tonnes": round(volume),
            "cout_par_tonne_usd": round(cost, 2),
            "delai_livraison_jours": round(delay, 2),
            "taux_disponibilite_flotte_pct": round(max(70, min(99, availability)), 1),
            "incidents_hse": hse_incidents,
            "stock_intermediaire_tonnes": round(stock),
            "evenement_notable": note,
        })

    return rows, events


def write_csv(rows):
    fieldnames = list(rows[0].keys())
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {CSV_PATH} ({len(rows)} lignes)")


MONTHLY_TEMPLATE = """# Rapport mensuel — {mois_nom} {annee}
**{corridor} — {site}**
*Document simulé à des fins de démonstration technique (aucune donnée réelle)*

## Résumé du mois
Le volume transporté s'est établi à **{volume:,} tonnes**, pour un coût moyen de
**{cout} USD/tonne**. Le délai de livraison moyen constaté est de **{delai} jours**,
avec un taux de disponibilité de la flotte de **{dispo}%**.

## Faits marquants
{faits_marquants}

## Indicateurs HSE
{hse_section}

## Stock intermédiaire
Le stock tampon en fin de mois s'élève à **{stock:,} tonnes**, {stock_commentaire}.

## Points de vigilance pour le mois suivant
{vigilance}
"""

INCIDENT_TEMPLATE = """# Rapport d'incident HSE — {mois_nom} {annee}
**{corridor} — {site}**
*Document simulé à des fins de démonstration technique (aucune donnée réelle)*

## Nature de l'incident
{nature}

## Impact opérationnel
- Délai de livraison additionnel estimé : {delai_impact} jours
- Disponibilité flotte impactée : -{dispo_impact} points
- Nombre d'incidents HSE comptabilisés ce mois : {n_incidents}

## Actions correctives engagées
{actions}

## Statut
{statut}
"""

REFERENCE_DOCS = {
    "procedure_gestion_incidents_hse.md": """# Procédure de gestion des incidents HSE
*Document de référence simulé*

## Objectif
Décrire le processus standard de déclaration, d'investigation et de clôture
des incidents Hygiène, Sécurité, Environnement (HSE) sur le corridor logistique.

## Étapes
1. **Déclaration immédiate** : tout incident est déclaré dans les 2 heures via
   le canal HSE dédié, avec classification de gravité (mineur / modéré / majeur).
2. **Investigation** : une analyse de cause racine (5 Pourquoi / Ishikawa) est
   menée dans les 5 jours ouvrés pour les incidents modérés et majeurs.
3. **Plan d'action correctif** : chaque incident majeur donne lieu à un plan
   d'action avec responsable et échéance, suivi en comité mensuel.
4. **Clôture** : l'incident n'est clôturé qu'après vérification de l'efficacité
   des actions correctives.

## Seuils d'alerte
- Plus de 2 incidents HSE majeurs sur un mois glissant : escalade automatique
  au comité de direction logistique.
- Toute baisse de disponibilité flotte supérieure à 5 points liée à un incident
  déclenche une revue opérationnelle dédiée.
""",
    "politique_gestion_stock_tampon.md": """# Politique de gestion du stock intermédiaire (stock tampon)
*Document de référence simulé*

## Principe
Le stock tampon a pour but d'absorber les variations de flux entre l'amont
(production/extraction) et l'aval (transport/export), afin de sécuriser la
continuité d'activité en cas d'aléa logistique (incident, maintenance,
conditions météo).

## Seuils cibles
- Stock cible : entre 35 000 et 55 000 tonnes selon la saisonnalité.
- En dessous de 30 000 tonnes : risque de rupture, priorité donnée à
  l'approvisionnement.
- Au-dessus de 60 000 tonnes : risque de saturation des capacités de stockage,
  arbitrage à faire sur les cadences amont.

## Lien avec les indicateurs de performance
Le stock tampon est suivi conjointement avec le délai de livraison et le taux
de disponibilité de la flotte : une hausse simultanée du stock et du délai de
livraison est un signal d'alerte sur la capacité de transport aval.
""",
    "methodologie_kpi_couts.md": """# Méthodologie de suivi des KPI coûts et performance
*Document de référence simulé*

## Définitions
- **Coût par tonne (USD/tonne)** : coût logistique total (transport + manutention
  + pertes) rapporté au tonnage effectivement livré sur le mois.
- **Délai de livraison (jours)** : délai moyen entre la mise à disposition du
  produit et sa réception au point de livraison contractuel.
- **Taux de disponibilité flotte (%)** : part du temps où les actifs de
  transport (trains, camions, convoyeurs) sont opérationnels par rapport au
  temps total planifié.

## Fréquence de revue
Ces indicateurs sont consolidés mensuellement et présentés en revue de
performance. Toute variation de plus de 5% du coût par tonne d'un mois sur
l'autre doit être expliquée et documentée dans le rapport mensuel.
""",
}


def build_report_texts(rows, events):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    faits_pool = [
        "Aucun événement majeur à signaler, activité conforme au plan.",
        "Une opération de maintenance préventive a été menée sur le tronçon "
        "central du corridor, sans impact significatif sur les livraisons.",
        "Des conditions météorologiques défavorables ont ponctuellement "
        "ralenti les opérations de chargement en début de mois.",
        "Une revue conjointe avec les équipes opérationnelles a permis "
        "d'identifier deux leviers d'amélioration sur la cadence de "
        "chargement.",
    ]
    vigilance_pool = [
        "Poursuivre le suivi rapproché du taux de disponibilité de la flotte.",
        "Confirmer la disponibilité des équipes de maintenance pour le mois "
        "prochain.",
        "Anticiper une hausse saisonnière de la demande et ajuster le "
        "dimensionnement du stock tampon en conséquence.",
        "Aucun point de vigilance particulier identifié à ce stade.",
    ]

    for i, row in enumerate(rows):
        mois_nom, annee = row["mois_nom"], row["annee"]
        event = events.get(i, "")

        if "incident_hse_majeur" in event:
            faits = (
                f"Un incident HSE majeur a été enregistré ce mois-ci, "
                f"entraînant une hausse des indicateurs d'incidents "
                f"({row['incidents_hse']} au total sur le mois) et un impact "
                f"sur la disponibilité de la flotte. Voir rapport d'incident "
                f"dédié pour le détail."
            )
        else:
            faits = random.choice(faits_pool)

        if "choc_cout" in event:
            faits += (
                f" Le coût par tonne a par ailleurs augmenté sur le mois "
                f"(effet ponts/surcoûts carburant), atteignant "
                f"{row['cout_par_tonne_usd']} USD/tonne."
            )

        if row["incidents_hse"] == 0:
            hse_section = "Aucun incident HSE enregistré ce mois-ci — mois sans incident."
        elif row["incidents_hse"] <= 1:
            hse_section = (
                f"{row['incidents_hse']} incident HSE mineur enregistré, sans "
                f"impact opérationnel significatif."
            )
        else:
            hse_section = (
                f"{row['incidents_hse']} incidents HSE enregistrés ce mois-ci, "
                f"dont au moins un de gravité modérée à majeure. Voir rapport "
                f"d'incident associé."
            )

        stock_commentaire = (
            "dans la fourchette cible" if 35_000 <= row["stock_intermediaire_tonnes"] <= 55_000
            else "en dehors de la fourchette cible, à surveiller"
        )

        text = MONTHLY_TEMPLATE.format(
            mois_nom=mois_nom, annee=annee, corridor=CORRIDOR_NAME, site=SITE_NAME,
            volume=row["volume_transporte_tonnes"], cout=row["cout_par_tonne_usd"],
            delai=row["delai_livraison_jours"], dispo=row["taux_disponibilite_flotte_pct"],
            faits_marquants=faits, hse_section=hse_section,
            stock=row["stock_intermediaire_tonnes"], stock_commentaire=stock_commentaire,
            vigilance=random.choice(vigilance_pool),
        )

        fname = f"rapport_mensuel_{row['date']}.md"
        with open(os.path.join(REPORTS_DIR, fname), "w", encoding="utf-8") as f:
            f.write(text)

        if "incident_hse_majeur" in event:
            natures = [
                "Arrêt non planifié d'un convoyeur suite à une défaillance "
                "mécanique, ayant nécessité une intervention de maintenance "
                "corrective en urgence.",
                "Quasi-accident lors d'une opération de chargement, "
                "impliquant un écart au mode opératoire standard. Aucun "
                "blessé, mais arrêt de la ligne le temps de l'investigation.",
                "Déversement mineur de minerai lors du transbordement, "
                "entraînant une intervention environnementale et un "
                "nettoyage de la zone concernée.",
            ]
            actions_pool = [
                "Renforcement des contrôles de maintenance préventive sur "
                "les équipements concernés ; sensibilisation des équipes "
                "terrain rappelée en briefing sécurité.",
                "Revue du mode opératoire standard avec les équipes "
                "concernées et mise à jour de la fiche réflexe associée.",
                "Audit HSE ciblé programmé sur la zone concernée dans les "
                "30 jours suivant l'incident.",
            ]
            inc_text = INCIDENT_TEMPLATE.format(
                mois_nom=mois_nom, annee=annee, corridor=CORRIDOR_NAME, site=SITE_NAME,
                nature=random.choice(natures),
                delai_impact=round(row["delai_livraison_jours"] * 0.4, 1),
                dispo_impact=round(random.uniform(3, 7), 1),
                n_incidents=row["incidents_hse"],
                actions=random.choice(actions_pool),
                statut=random.choice([
                    "Incident clôturé après vérification de l'efficacité des "
                    "actions correctives.",
                    "Incident en cours de clôture, plan d'action en suivi au "
                    "comité mensuel.",
                ]),
            )
            inc_fname = f"rapport_incident_{row['date']}.md"
            with open(os.path.join(REPORTS_DIR, inc_fname), "w", encoding="utf-8") as f:
                f.write(inc_text)

    for fname, content in REFERENCE_DOCS.items():
        with open(os.path.join(REFERENCE_DIR, fname), "w", encoding="utf-8") as f:
            f.write(content)

    n_reports = len([n for n in os.listdir(REPORTS_DIR) if n.endswith(".md")])
    n_ref = len(REFERENCE_DOCS)
    print(f"[ok] {n_reports} rapports texte générés dans {REPORTS_DIR}")
    print(f"[ok] {n_ref} documents de référence générés dans {REFERENCE_DIR}")


def main():
    months = month_sequence(N_MONTHS, START_YEAR, START_MONTH)
    rows, events = generate_kpi_series(months)
    write_csv(rows)
    build_report_texts(rows, events)


if __name__ == "__main__":
    main()
