"""Maintenance prédictive : estime le risque de panne d'un équipement dans
les 7 prochains jours à partir de son historique d'usage (heures cumulées,
heures depuis la dernière maintenance, type d'équipement).

Modèle volontairement simple (régression logistique) : l'objectif du
projet est de démontrer une approche de bout en bout (features -> modèle
-> évaluation -> score exploitable), pas de maximiser la performance sur
des données synthétiques.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src import config

NUMERIC_FEATURES = ["heures_operees", "heures_cumulees", "heures_depuis_maintenance", "ratio_usure"]
CATEGORICAL_FEATURES = ["etape"]
TARGET = "panne_sous_7j"


def load_equipment_daily() -> pd.DataFrame:
    df = pd.read_csv(config.EQUIPMENT_DAILY_CSV, parse_dates=["date"])
    return df.sort_values(["equipment_id", "date"]).reset_index(drop=True)


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("prep", preprocessor), ("clf", model)])


def train_and_evaluate(df: pd.DataFrame = None, test_size: float = 0.25, random_state: int = 42) -> dict:
    df = df if df is not None else load_equipment_daily()
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET]

    # split temporel simple (train sur les 3/4 premiers mois, test sur le reste)
    # plutôt qu'un split aléatoire, pour rester réaliste vis-à-vis d'un usage
    # de prévision (on n'entraîne jamais sur le futur).
    split_date = df["date"].quantile(1 - test_size, interpolation="nearest")
    train_mask = df["date"] <= split_date
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )

    return {
        "pipeline": pipeline,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "roc_auc": round(float(auc), 3) if auc == auc else None,  # NaN check
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
        "base_rate_test": round(float(y_test.mean()), 3),
    }


def current_risk_scores(pipeline: Pipeline, df: pd.DataFrame = None) -> pd.DataFrame:
    """Retourne, pour chaque équipement, le score de risque de panne à
    7 jours calculé sur sa dernière observation disponible (utilisé par le
    dashboard pour prioriser la maintenance préventive)."""
    df = df if df is not None else load_equipment_daily()
    latest = df.sort_values("date").groupby("equipment_id").tail(1).copy()
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    latest["risque_panne_7j"] = pipeline.predict_proba(latest[features])[:, 1]
    return latest[["equipment_id", "etape", "heures_depuis_maintenance", "ratio_usure", "risque_panne_7j"]] \
        .sort_values("risque_panne_7j", ascending=False)
