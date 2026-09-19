# Assistant RAG + Forecast — KPI Logistiques & Miniers (démonstrateur)

Assistant conversationnel qui combine **recherche documentaire augmentée
(RAG)**, **prévision de séries temporelles (forecast)** et un **agent léger
à function calling** pour répondre à des questions mêlant faits qualitatifs
(rapports, incidents, procédures) et indicateurs numériques (coûts, volumes,
délais) sur un corridor logistique/minier.

> ⚠️ **Données 100% synthétiques.** Ce dépôt est un démonstrateur technique.
> Le jeu de données (KPI mensuels + rapports opérationnels + documents de
> référence) est entièrement généré par script (`data/generate_synthetic_data.py`,
> graine aléatoire fixe) et ne représente **aucune donnée réelle d'aucune
> entreprise**. Aucune clé API payante n'est requise pour faire tourner le
> projet.

---

## Pourquoi ce projet

Ce projet a été construit pour illustrer, sur un cas d'usage logistique
concret, les briques techniques suivantes :

- **RAG** : chunking, embeddings, indexation vectorielle (FAISS), retrieval,
  génération de réponse avec citation des sources.
- **Forecast** : modélisation de séries temporelles (tendance + saisonnalité)
  avec repli automatique en cas d'échec du modèle principal.
- **IA agentique** : un agent qui décide, à partir du langage naturel, quel
  outil interne appeler (recherche documentaire, prévision, ou les deux) —
  pattern *function calling*, avec un mode optionnel utilisant l'API
  function-calling d'OpenAI.

## Architecture

```
Question utilisateur
        │
        ▼
 ┌─────────────────┐        décide quel(s) outil(s) utiliser
 │  Agent (router)  │  ───── (règles par défaut, ou OpenAI
 └───────┬──────────┘        function calling si clé API dispo)
         │
   ┌─────┴─────────────────────────┐
   ▼                                ▼
┌───────────────┐           ┌──────────────────┐
│ search_reports │          │   forecast_kpi     │
│ (pipeline RAG) │          │ (moteur forecast)  │
└───────┬────────┘          └─────────┬──────────┘
        │                             │
  chunking → embeddings         Holt-Winters (statsmodels)
  → index FAISS → retrieval     → repli régression linéaire
  → réponse + sources           → valeurs futures + explication
        │                             │
        └───────────────┬─────────────┘
                         ▼
              Réponse finale composée
              (texte + sources + graphe)
```

## Stack technique

| Brique | Choix par défaut | Alternative / extension prévue |
|---|---|---|
| Chunking | découpage par fenêtre glissante avec chevauchement (`src/chunking.py`) | — |
| Embeddings | TF-IDF + réduction SVD (scikit-learn), aucune dépendance lourde | `sentence-transformers` (local) ou API OpenAI — voir `src/embeddings.py` |
| Vector store | FAISS (`IndexFlatIP`, similarité cosinus) | index approximatif (IVF/HNSW) pour un corpus plus large |
| Génération | synthèse extractive (sans LLM, sans clé API) | backend OpenAI (Chat Completions) — voir `src/llm_backend.py` |
| Forecast | Holt-Winters (statsmodels), repli régression linéaire (scikit-learn) | ARIMA/Prophet possibles en extension |
| Agent / routage | classifieur à base de règles (mots-clés + regex) | function calling OpenAI si `OPENAI_API_KEY` défini |
| Interface | Streamlit (`app.py`) | Notebook Jupyter (`notebooks/demo.ipynb`) |

Le principe directeur : **tout le pipeline fonctionne sans aucune clé API**,
avec une architecture pensée pour brancher un LLM ou des embeddings denses
en production sans réécrire le reste du code (interfaces `BaseEmbeddings`
et `BaseLLMBackend`).

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
- `data/kpi_timeseries.csv` — 36 mois de KPI simulés (coût/tonne, volume,
  délai, disponibilité flotte, incidents HSE, stock intermédiaire)
- `data/reports/*.md` — rapports mensuels + rapports d'incident fictifs
- `data/reference/*.md` — documents de référence (procédures, politiques)

### 2. Lancer l'interface Streamlit

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

## Exemples de questions

| Question | Outil(s) déclenché(s) | Ce que ça montre |
|---|---|---|
| *« Quelle sera l'évolution du coût par tonne dans les 3 prochains mois ? »* | `forecast_kpi` | routage vers le forecast, extraction automatique de la métrique et de l'horizon |
| *« Pourquoi le taux de disponibilité de la flotte a-t-il baissé en novembre 2023 ? »* | `search_reports` | retrieval RAG + réponse sourcée sur un rapport mensuel précis |
| *« Que dit la procédure de gestion des incidents HSE en cas d'incident majeur ? »* | `search_reports` | retrieval sur un document de référence (procédure), pas un rapport mensuel |
| *« Quels incidents HSE ont eu lieu récemment et comment va évoluer le délai de livraison dans les 2 prochains mois ? »* | `forecast_kpi` **+** `search_reports` | question hybride : l'agent combine les deux outils et compose une réponse unique |

## Activer les backends optionnels (LLM / embeddings denses)

Par défaut, tout tourne sans clé API. Pour brancher un LLM en génération et
en routage agentique :

```bash
export OPENAI_API_KEY=sk-...
export RAG_LLM_BACKEND=openai
pip install openai
```

L'agent bascule alors automatiquement en mode `openai_function_calling`
(`src/agent.py`) et le pipeline RAG utilise `OpenAIBackend` pour la
génération (`src/llm_backend.py`). En cas d'erreur (quota, clé invalide),
le code retombe automatiquement sur les backends par défaut plutôt que de
planter.

## Limites connues

- Corpus et KPI **synthétiques**, de taille volontairement réduite (36 mois,
  ~45 documents) — adapté à une démonstration, pas à une évaluation de
  performance à l'échelle.
- Le mode de génération par défaut est **extractif** (pas de reformulation
  par un LLM) afin de ne dépendre d'aucune clé API ; la qualité rédactionnelle
  des réponses est donc volontairement modeste dans ce mode.
- Le routage par règles (mots-clés) est simple par construction ; il est
  documenté comme point d'amélioration naturel vers un routage 100% appris
  (function calling LLM, déjà implémenté en option).
- Les prévisions Holt-Winters sur 36 points mensuels restent indicatives ;
  aucune validation croisée temporelle poussée n'a été mise en place (hors
  périmètre d'un démonstrateur).

## Structure du dépôt

```
.
├── app.py                          # interface Streamlit
├── data/
│   ├── generate_synthetic_data.py  # génération du jeu de données synthétique
│   ├── kpi_timeseries.csv          # (généré) série de KPI mensuels
│   ├── reports/                    # (généré) rapports mensuels + incidents
│   └── reference/                  # (généré) procédures / politiques
├── notebooks/
│   └── demo.ipynb                  # walkthrough complet de l'architecture
├── src/
│   ├── config.py                   # chemins et paramètres
│   ├── chunking.py                 # découpage de texte
│   ├── embeddings.py               # backends d'embedding (TF-IDF par défaut)
│   ├── vector_store.py             # wrapper FAISS
│   ├── llm_backend.py              # backends de génération (extractif / OpenAI)
│   ├── rag_pipeline.py             # orchestration RAG de bout en bout
│   ├── forecast.py                 # moteur de forecast + extraction d'intention
│   └── agent.py                    # agent / routage function-calling
├── tests/
│   └── test_pipeline.py            # tests de fumée (pytest)
└── requirements.txt
```

---

*Projet réalisé dans le cadre d'une préparation de candidature autour des
sujets LLM / RAG / IA agentique appliqués à la logistique et au forecast.*
