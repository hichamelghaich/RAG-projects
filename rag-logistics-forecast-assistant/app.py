"""Interface Streamlit — Assistant RAG + Forecast pour KPI logistiques.

Lancer avec :
    streamlit run app.py

⚠️ Toutes les données affichées sont synthétiques (voir data/generate_synthetic_data.py).
"""

import pandas as pd
import streamlit as st

from src.agent import LogisticsAgent
from src import config

st.set_page_config(page_title="Assistant RAG + Forecast — Logistique", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner="Initialisation de l'agent (index RAG + moteur de forecast)...")
def get_agent():
    agent = LogisticsAgent()
    agent._ensure_index()
    return agent


st.title("📦 Assistant RAG + Forecast — KPI Logistiques")
st.caption(
    "⚠️ Démonstrateur technique — toutes les données (rapports, incidents, "
    "KPI) sont **100% synthétiques**, générées par `data/generate_synthetic_data.py`. "
    "Aucune donnée réelle d'aucune entreprise n'est utilisée."
)

agent = get_agent()

tab_chat, tab_data = st.tabs(["💬 Assistant", "📊 Données"])

with tab_chat:
    st.markdown(
        "Posez une question en langage naturel. L'agent choisit automatiquement "
        "d'interroger la base documentaire (RAG), le module de prévision "
        "(forecast), ou les deux, selon l'intention détectée dans la question."
    )

    examples = [
        "Quelle sera l'évolution du coût par tonne dans les 3 prochains mois ?",
        "Pourquoi le taux de disponibilité de la flotte a-t-il baissé en novembre 2023 ?",
        "Que dit la procédure de gestion des incidents HSE en cas d'incident majeur ?",
        "Quels incidents HSE ont eu lieu récemment et comment va évoluer le délai de livraison dans les 2 prochains mois ?",
    ]
    cols = st.columns(len(examples))
    picked = None
    for c, ex in zip(cols, examples):
        if c.button(ex, use_container_width=True):
            picked = ex

    question = st.text_input("Votre question :", value=picked or "")

    if st.button("Envoyer", type="primary") and question.strip():
        with st.spinner("Analyse de la question et récupération des informations..."):
            response = agent.ask(question)

        st.subheader("Réponse")
        st.info(f"🧭 Outil(s) déclenché(s) par l'agent : **{', '.join(response.tools_used)}** "
                f"(routage : `{response.routing_method}`)")
        st.markdown(response.answer)

        if response.forecast_data:
            fdata = response.forecast_data
            hist_df = pd.DataFrame({
                "date": fdata["history_dates"][-12:],
                "valeur": fdata["history_values"][-12:],
                "type": "historique",
            })
            fut_df = pd.DataFrame({
                "date": fdata["forecast_dates"],
                "valeur": fdata["forecast_values"],
                "type": "prévision",
            })
            chart_df = pd.concat([hist_df, fut_df]).set_index("date")
            st.line_chart(chart_df.pivot_table(index=chart_df.index, columns="type", values="valeur"))

        if response.sources:
            with st.expander("📄 Sources documentaires utilisées"):
                for s in response.sources:
                    st.write(f"- `{s}`")

with tab_data:
    st.subheader("Série temporelle des KPI (synthétique)")
    df = pd.read_csv(config.KPI_CSV_PATH)
    st.dataframe(df, use_container_width=True)
    metric = st.selectbox("Visualiser un indicateur :", list(config.FORECAST_METRICS.keys()),
                           format_func=lambda m: config.FORECAST_METRICS[m])
    st.line_chart(df.set_index("date")[metric])
