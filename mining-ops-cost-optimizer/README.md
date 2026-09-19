# Optimiseur de coûts opérationnels miniers (démonstrateur)

Projet de modélisation appliqué à une chaîne opérationnelle minière —
**forage → sautage → chargement → transport → énergie** — combinant
**décomposition et suivi des coûts**, **simulation Monte Carlo** (risque de
dépassement de budget / d'objectif de tonnage) et **optimisation**
(recherche opérationnelle : arbitrage flotte propre / sous-traitance pour
atteindre un objectif de production au moindre coût).

> ⚠️ **Données 100% synthétiques.** Ce dépôt est un démonstrateur technique.
> Le jeu de données (KPI quotidiens, flotte d'équipements, disponibilité
> journalière) est entièrement généré par script
> (`data/generate_synthetic_data.py`, graine aléatoire fixe) et ne
> représente **aucun site, équipement ou entreprise réel**. Les débits
> horaires par étape sont des paramètres stylisés (voir *Limites*), pas des
> ratios d'ingénierie minière réels.

## Pourquoi ce projet

Ce projet reproduit, en miniature, le type de problème traité au quotidien
en exploitation minière : piloter le coût à la tonne sur l'ensemble de la
chaîne opérationnelle, quantifier le risque de dépasser un budget ou de
manquer un objectif de production, et arbitrer entre moyens propres et
sous-traitance pour tenir cet objectif. Il complète un premier projet
([`rag-logistics-forecast-assistant`](../rag-logistics-forecast-assistant))
centré sur RAG/LLM/IA agentique : celui-ci se concentre sur la
**modélisation statistique, la simulation et la recherche opérationnelle
appliquées à une problématique opérationnelle concrète**, directement dans
la continuité d'une expérience terrain en exploitation minière (pilotage
d'opérations, optimisation des coûts forage-sautage-chargement-transport-
énergie, suivi HSE et disponibilité de flotte).

## Architecture

```
data/generate_synthetic_data.py
        │
        ▼
┌───────────────────┐   ┌────────────────────┐   ┌──────────────────────┐
│ daily_kpi.csv      │   │ fleet_equipment.csv │   │ equipment_daily.csv   │
│ (coût/tonne par    │   │ (référentiel flotte) │   │ (heures opérées,      │
│  étape, dispo, HSE)│   │                      │   │  disponibilité/jour)  │
└─────────┬──────────┘   └──────────┬───────────┘   └───────────┬───────────┘
          │                          │                            │
          ▼                          └──────────────┬─────────────┘
 src/cost_model.py                                   ▼
 (décomposition, tendance,                 src/simulation.py
  alertes coût)                            (Monte Carlo par
          │                                 ré-échantillonnage historique)
          │                                            │
          └───────────────────┬────────────────────────┘
                               ▼
                    src/optimization.py
           (programme linéaire — scipy.optimize.linprog
            arbitrage flotte propre / sous-traitance
            pour un objectif de production donné)
                               │
                               ▼
                     app.py (dashboard Streamlit)
                     notebooks/demo.ipynb
```

## Stack technique

| Brique | Choix | Détail |
|---|---|---|
| Données | génération synthétique (Python `csv`/`random`) | KPI quotidiens 18 mois + panel équipement 24 machines |
| Modèle de coûts | pandas | décomposition par étape, tendance hebdomadaire, alerte mois/mois |
| Simulation | NumPy (bootstrap / Monte Carlo) | risque de dépassement de budget ou de manquer un objectif de tonnage |
| Optimisation | scipy.optimize (`linprog`, méthode HiGHS) | arbitrage flotte propre / sous-traitance au moindre coût |
| Interface | Streamlit (`app.py`) | 4 onglets : vue d'ensemble, coûts, simulation, optimisation |
| Notebook | Jupyter (`notebooks/demo.ipynb`) | walkthrough complet |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate  # optionnel mais recommandé
pip install -r requirements.txt
```

## Utilisation

### 1. Générer les données synthétiques

```bash
python data/generate_synthetic_data.py
```

Génère :
- `data/daily_kpi.csv` — 540 jours de KPI (production, coût par étape,
  disponibilité flotte, incidents HSE)
- `data/fleet_equipment.csv` — 24 équipements simulés (foreuses, pelles,
  camions, groupes électrogènes)
- `data/equipment_daily.csv` — panel quotidien par équipement (heures
  opérées, disponibilité)

### 2. Lancer le dashboard

```bash
streamlit run app.py
```

### 3. Ou explorer via le notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

### 4. Lancer les tests

```bash
pytest -q
```

## Ce que chaque module démontre

**`src/cost_model.py`** — décomposition du coût total par étape (waterfall),
tendance hebdomadaire, détection d'écart mois-sur-mois. Équivalent direct
d'un suivi de KPI / revue de performance en contrôle de gestion opérationnel.

**`src/simulation.py`** — plutôt que de supposer une loi de probabilité
théorique, le modèle tire au hasard, avec remise (bootstrap), des journées
réellement observées dans l'historique pour composer des milliers de
scénarios plausibles d'un horizon futur (7 à 60 jours). Il en tire des
distributions (P10/P50/P90) du coût à la tonne et du tonnage, et des
probabilités concrètes de dépasser un budget ou de manquer un objectif de
production — une approche simple, robuste et sans hypothèse forte, standard
en gestion des risques opérationnels.

**`src/optimization.py`** — pour un objectif de production sur 7 jours,
détermine la répartition d'heures la moins coûteuse entre flotte propre
(capacité limitée, calibrée sur la disponibilité récente) et sous-traitance
(coût plus élevé mais capacité non limitée), par étape de la chaîne. Le
scénario "flotte propre seule" sert de référence pour chiffrer le **coût
marginal de la sous-traitance** nécessaire pour combler l'écart de
production — un vrai résultat de recherche opérationnelle, pas une simple
règle de trois.

## Exemple de résultat (optimisation, objectif = +10% de production)

| Étape | Heures flotte propre | Heures sous-traitance | Coût flotte propre (USD) | Coût sous-traitance (USD) |
|---|---|---|---|---|
| Forage | 473.5 (100% capacité) | 112.5 | 300 005 | 99 804 |
| Chargement | 593.7 (100% capacité) | 150.1 | 349 292 | 123 621 |
| Transport | 1 426.4 (100% capacité) | 364.2 | 558 150 | 199 516 |
| Énergie | 420.4 (95% capacité) | 0.0 | 204 963 | 0 |

Coût total optimisé : **2 085 290 USD** sur 7 jours. Un scénario "flotte
propre seule" sous-produirait de **19 667 tonnes** ; combler cet écart via
la sous-traitance coûte environ **21 USD/tonne** de coût marginal — c'est ce
type de chiffrage qui permet d'arbitrer objectivement entre investir dans
la flotte propre ou sous-traiter ponctuellement.

## Limites connues

- **Débits horaires stylisés** : les ratios tonnes/heure par étape sont
  calibrés pour que la capacité de flotte simulée corresponde à la
  production moyenne du jeu de données synthétique — ce ne sont pas des
  valeurs d'ingénierie minière réelles (elles varient énormément selon la
  géologie, la méthode d'exploitation, la distance de transport, etc.).
- **Sautage hors optimisation de flotte** : dans ce modèle, le sautage n'a
  pas de flotte d'équipement propre (souvent externalisé/évènementiel dans
  la réalité) ; il est traité comme un coût variable au tonnage, en dehors
  de l'arbitrage flotte propre / sous-traitance.
- **Simulation par bootstrap historique** : simple et robuste, mais suppose
  que les 180 derniers jours restent représentatifs du futur proche ; une
  version plus avancée pondérerait les tirages ou modéliserait explicitement
  une tendance/saisonnalité.
- **Optimisation déterministe** : les capacités utilisées sont des moyennes
  historiques ; une version plus avancée traiterait la disponibilité comme
  une variable aléatoire (optimisation robuste / sous incertitude) — un pont
  naturel avec le module de simulation.

## Structure du dépôt

```
.
├── app.py                          # dashboard Streamlit
├── data/
│   ├── generate_synthetic_data.py  # génération du jeu de données synthétique
│   ├── daily_kpi.csv               # (généré) KPI quotidiens
│   ├── fleet_equipment.csv         # (généré) référentiel flotte
│   └── equipment_daily.csv         # (généré) panel quotidien équipement/disponibilité
├── notebooks/
│   └── demo.ipynb                  # walkthrough complet de l'architecture
├── src/
│   ├── config.py                   # chemins, étapes, débits, paramètres
│   ├── cost_model.py                # décomposition et tendance des coûts
│   ├── simulation.py                # simulation Monte Carlo (bootstrap historique)
│   └── optimization.py              # arbitrage flotte propre / sous-traitance (scipy)
├── tests/
│   └── test_pipeline.py            # tests de fumée (pytest)
└── requirements.txt
```

---

*Projet réalisé dans le cadre d'une préparation de candidature autour des
sujets modélisation / forecast / optimisation / simulation appliqués aux
opérations minières et logistiques.*
