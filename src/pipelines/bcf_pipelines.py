from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.config import RANDOM_STATE
from src.data_loading.bcf_loader import load_bcf_dataset
from src.models.random_forest import build_rf_classifier, build_rf_regressor
from src.explainers.lime_explainer import (
    build_lime_classifier_explainer,
    build_lime_regressor_explainer,
)
from src.explainers.shap_explainer import (
    build_shap_classifier_explainer,
    build_shap_regressor_explainer,
)

BCF_COL = "logBCF"
CLASS_COL = "Class"

DESCRIPTOR_COLS = [
    "nHM", "piPC09", "PCD", "X2Av", "MLOGP",
    "ON1V", "N-072", "B02[C-N]", "F04[C-O]",
]

@dataclass
class BCFClassificationResult:
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

@dataclass
class BCFRegressionResult:
    model: object
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    lime_explainer: object
    shap_explainer: object
    y_pred: np.ndarray
    mse: float
    mae: float
    r2: float


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = df[DESCRIPTOR_COLS].copy()
    y_reg = df[BCF_COL].copy()
    y_clf = df[CLASS_COL].copy()
    return X, y_reg, y_clf


def run_bcf_classification(csv_path: str) -> BCFClassificationResult:
    df = load_bcf_dataset(csv_path)
    X, _, y_clf = _prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_clf,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_clf,
    )

    clf = build_rf_classifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    feature_names = DESCRIPTOR_COLS
    class_names = sorted(y_clf.unique().tolist())
    class_names_str = [str(c) for c in class_names]

    lime_exp = build_lime_classifier_explainer(
        X_train=X_train,
        feature_names=feature_names,
        class_names=class_names_str,
    )

    shap_exp = build_shap_classifier_explainer(
        model=clf,
        X_train=X_train,
    )

    return BCFClassificationResult(
        model=clf,
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


def run_bcf_regression(csv_path: str) -> BCFRegressionResult:
    df = load_bcf_dataset(csv_path)
    X, y_reg, _ = _prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_reg,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    reg = build_rf_regressor()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    lime_exp = build_lime_regressor_explainer(
        X_train=X_train,
        feature_names=DESCRIPTOR_COLS,
    )

    shap_exp = build_shap_regressor_explainer(
        model=reg,
        X_train=X_train,
    )

    return BCFRegressionResult(
        model=reg,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        lime_explainer=lime_exp,
        shap_explainer=shap_exp,
        y_pred=y_pred,
        mse=mse,
        mae=mae,
        r2=r2,
    )
