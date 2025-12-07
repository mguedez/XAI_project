from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.config import RANDOM_STATE
from src.data_loading.tiresia_loader import load_raw_tiresia, clean_smiles_and_labels
from src.preprocessing.tiresia_descriptors import compute_rdkit_descriptors
from src.explainers.lime_explainer import build_lime_classifier_explainer
from src.explainers.shap_explainer import build_shap_classifier_explainer
from src.models.random_forest import build_rf_classifier
from src.preprocessing.descriptor_cleaning import clean_descriptors

@dataclass
class TiresiaResult:
    model: object
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    lime_explainer: object
    shap_explainer: object
    y_pred: np.ndarray
    accuracy: float
    report: str

def run_tiresia_pipeline(txt_path: str) -> TiresiaResult:
    # 1) carga y limpieza de smiles/labels
    df_raw = load_raw_tiresia(txt_path)
    df_clean = clean_smiles_and_labels(df_raw)

    # 2) calcular descriptores RDKit
    desc_df = compute_rdkit_descriptors(df_clean, smiles_col="SMILES", label_col="LABEL")

    # 2.1) limpieza de descriptores: eliminar columnas con nulos, constantes y muy correladas
    desc_df = clean_descriptors(
        desc_df, smiles_col="SMILES", label_col="LABEL", corr_threshold=0.95
    )

    feature_cols = [c for c in desc_df.columns if c not in ("SMILES", "LABEL")]
    X = desc_df[feature_cols].copy()
    y = desc_df["LABEL"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    rf = build_rf_classifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    lime_exp = build_lime_classifier_explainer(
        X_train=X_train,
        feature_names=feature_cols,
        class_names=["non-toxic", "toxic"],
    )

    shap_exp = build_shap_classifier_explainer(
        model=rf,
        X_train=X_train,
    )

    return TiresiaResult(
        model=rf,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        lime_explainer=lime_exp,
        shap_explainer=shap_exp,
        y_pred=y_pred,
        accuracy=acc,
        report=report,
    )
