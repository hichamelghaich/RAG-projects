"""Dashboard Streamlit — Optimiseur de coûts opérationnels miniers.

Lancer avec :
    streamlit run app.py

⚠️ Toutes les données affichées sont synthétiques (voir data/generate_synthetic_data.py).
"""

import pandas as pd
import streamlit as st

from src import config
from src.cost_model import load_daily_kpi, cost_breakdown, cost_trend, month_over_month_delta, summary_kpis
from src.predictive_maintenance import load_equipment_daily, train_and_evaluate, current_risk_scores
from src.optimization import run_scenario

st.set_page_config(page_title="Optimiseur de coûts opérationnels miniers", page_icon="⛏️", layout="wide")

st.title("⛏️ Optimiseur de coûts opérationnels miniers")
st.caption(
    "⚠️ Démonstrateur technique — toutes les données (production, coûts, flotte, "
    "pannes) sont **100% synthétiques**, générées par `data/generate_synthetic_data.py`. "
    "Aucune donnée réelle d'aucun site minier n'est utilisée."
)


@st.cache_data(show_spinner="Chargement des données...")
def get_data():
    return load_daily_kpi(), load_equipment_daily()


@st.cache_resource(show_spinner="Entraînement du modèle de maintenance prédictive...")
def get_maintenance_model():
    return train_and_evaluate()


kpi_df, eq_df = get_data()

tab_overview, tab_cost, tab_maintenance, tab_optim = st.tabs(
    ["📊 Vue d'ensemble", "💰 Coûts par étape", "🔧 Maintenance prédictive", "🧮 Optimisation flotte/sous-traitance"]
)

with tab_overview:
    kpis = summary_kpis(kpi_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Production moy. (t/j, 30j)", f"{kpis['tonnes_produites_moy_j']:,}")
    c2.metric("Coût total moy. (USD/t, 30j)", f"{kpis['cout_total_usd_t_moy']}")
    c3.metric("Disponibilité flotte moy. (%)", f"{kpis['disponibilite_flotte_pct_moy']}")
    c4.metric("Incidents HSE (30j)", kpis['incidents_hse_total_30j'])

    delta = month_over_month_delta(kpi_df)
    st.info(
        f"Coût total : **{delta['cout_recent_usd_t']} USD/t** sur les 30 derniers jours "
        f"vs **{delta['cout_precedent_usd_t']} USD/t** les 30 jours précédents "
        f"({'+' if delta['delta_pct'] >= 0 else ''}{delta['delta_pct']}%)."
    )

    st.subheader("Tendance hebdomadaire")
    trend = cost_trend(kpi_df, freq="W")
    metric = st.selectbox("Indicateur :", trend.columns, index=list(trend.columns).index("cout_total_usd_t"))
    st.line_chart(trend[metric])

with tab_cost:
    st.subheader("Décomposition du coût par étape (USD/tonne)")
    period = st.radio("Période :", ["all", "last_90d", "last_30d"], horizontal=True,
                       format_func=lambda p: {"all": "Toute la période", "last_90d": "90 derniers jours",
                                               "last_30d": "30 derniers jours"}[p])
    breakdown = cost_breakdown(kpi_df, period=period)
    st.bar_chart(breakdown)
    st.dataframe(breakdown.rename("USD/tonne").to_frame(), use_container_width=True)

with tab_maintenance:
    st.subheader("Modèle de risque de panne à 7 jours")
    result = get_maintenance_model()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC AUC (test)", result["roc_auc"])
    c2.metric("Précision", result["precision"])
    c3.metric("Rappel", result["recall"])
    c4.metric("Taux de base (test)", result["base_rate_test"])

    st.caption(
        f"Entraîné sur {result['n_train']} observations, évalué sur {result['n_test']} "
        "(split temporel — le modèle n'est jamais entraîné sur le futur)."
    )

    st.subheader("Équipements à risque (score courant)")
    risk_df = current_risk_scores(result["pipeline"], eq_df)
    risk_df = risk_df.rename(columns={
        "equipment_id": "Équipement", "etape": "Étape",
        "heures_depuis_maintenance": "Heures depuis maintenance",
        "ratio_usure": "Ratio d'usure", "risque_panne_7j": "Risque de panne (7j)",
    })
    st.dataframe(
        risk_df.style.format({"Risque de panne (7j)": "{:.1%}", "Ratio d'usure": "{:.2f}"}),
        use_container_width=True,
    )

with tab_optim:
    st.subheader("Arbitrage flotte propre / sous-traitance")
    st.markdown(
        "Pour un objectif de production sur les 7 prochains jours, le modèle "
        "détermine la répartition d'heures la moins coûteuse entre flotte "
        "propre (capacité limitée par la disponibilité récente) et "
        "sous-traitance (coût plus élevé, capacité non limitée), étape par "
        "étape de la chaîne opérationnelle."
    )

    target_pct = st.slider("Objectif de production vs. moyenne récente (%)", 90, 150, 110, step=5)
    scenario = run_scenario(target_multiplier=target_pct / 100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tonnage cible (7 jours)", f"{scenario.target_tonnage:,}")
    c2.metric("Coût total optimisé (USD)", f"{scenario.total_cost_usd:,}")
    c3.metric("Manque à gagner flotte propre seule (t)", f"{scenario.baseline_shortfall_tonnes:,}")

    if scenario.cout_marginal_usd_par_tonne is not None:
        st.warning(
            f"Atteindre cet objectif nécessite de sous-traiter une partie de la "
            f"chaîne : surcoût de **{scenario.cout_additionnel_pour_cible_usd:,} USD** "
            f"par rapport à un scénario flotte propre seule (qui, lui, sous-produirait "
            f"de {scenario.baseline_shortfall_tonnes:,} t) — soit environ "
            f"**{scenario.cout_marginal_usd_par_tonne} USD/tonne** de coût marginal "
            f"pour combler l'écart."
        )
    else:
        st.success("La flotte propre suffit à atteindre cet objectif sans sous-traitance.")

    st.subheader("Allocation détaillée par étape")
    st.dataframe(scenario.stage_allocation, use_container_width=True)
    st.bar_chart(
        scenario.stage_allocation.set_index("etape")[["heures_flotte_propre", "heures_sous_traitance"]]
    )
