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
from src.llm.gemini_explainer import explicar_con_gemini

BCF_COL = "logBCF"
CLASS_COL = "Class"

DESCRIPTOR_COLS = [
    "nHM", "piPC09", "PCD", "X2Av", "MLOGP",
    "ON1V", "N-072", "B02[C-N]", "F04[C-O]",
]


def _generate_bcf_llm_explanation(
    shap_explainer, 
    lime_explainer,
    X_test, 
    y_pred, 
    feature_names, 
    task_type, 
    model,
    accuracy_val=None
) -> str | None:
    """
    Genera explicación LLM global para el modelo BCF.
    Extrae información real de SHAP y LIME.
    Si hay error (ej: API key no configurada), retorna None.
    """
    # Calcular valores SHAP globales (importancia promedio)
    shap_values = shap_explainer(X_test[:10])  # Usar subset para velocidad
    feature_importance = shap_values.values.mean(axis=0)
    feature_importance = feature_importance.ravel()  # Asegurar que sea 1D
    
    # Crear resumen de SHAP global
    shap_global_summary = "\n".join([
        f"- {fname}: {imp:.4f}" 
        for fname, imp in sorted(zip(feature_names, feature_importance), key=lambda x: abs(x[1]), reverse=True)[:5]
    ])
    
    # Extraer información real de LIME para algunas instancias
    lime_explanations = []
    num_instances = min(3, len(X_test))  # Explicar hasta 3 instancias
    
    for idx in range(num_instances):
        instance = X_test.iloc[idx].values
        pred = y_pred[idx]
        
        # Generar explicación LIME para esta instancia
        # Detectar si es clasificación o regresión
        if hasattr(model, 'predict_proba'):
            predict_fn = model.predict_proba  # Clasificación
        else:
            predict_fn = model.predict  # Regresión
        
        lime_exp_obj = lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=5,
        )
        
        # Extraer features y pesos
        lime_list = lime_exp_obj.as_list()
        lime_text = f"Instancia {idx+1} (Predicción: {pred:.4f}):\n"
        for feature_desc, weight in lime_list:
            lime_text += f"  • {feature_desc}: {weight:.4f}\n"
        lime_explanations.append(lime_text)
    
    lime_global_summary = "\n".join(lime_explanations)
    
    # Info del modelo
    model_info = f"Predicciones realizadas sobre {len(y_pred)} instancias de prueba"
    if accuracy_val is not None:
        metric_name = "Accuracy" if task_type.endswith("Classification") else "R²"
        model_info += f" ({metric_name}: {accuracy_val:.4f})"
    
    # Generar explicación con LLM
    explanation = explicar_con_gemini(
        shap_local=model_info,
        shap_global=f"Descriptores más importantes según SHAP:\n{shap_global_summary}",
        lime_exp=f"Análisis LIME local (interpretación lineal de contribuciones):\n{lime_global_summary}",
        pred=str(y_pred.mean()) if hasattr(y_pred, 'mean') else str(y_pred),
        task_type=task_type,
        expertise_level="beginner",
        domain="AI"
    )
    return explanation



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
    feature_names: list[str]
    llm_explanation: str | None = None

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
    feature_names: list[str]
    llm_explanation: str | None = None


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

    # Generar explicación LLM global (opcional)
    llm_explanation = _generate_bcf_llm_explanation(
        shap_explainer=shap_exp,
        lime_explainer=lime_exp,
        X_test=X_test,
        y_pred=y_pred,
        feature_names=feature_names,
        task_type="BCF Classification",
        model=clf,
        accuracy_val=acc,
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
        feature_names=feature_names,
        llm_explanation=llm_explanation,
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

    # Generar explicación LLM global (opcional)
    llm_explanation = _generate_bcf_llm_explanation(
        shap_explainer=shap_exp,
        lime_explainer=lime_exp,
        X_test=X_test,
        y_pred=y_pred,
        feature_names=DESCRIPTOR_COLS,
        task_type="BCF Regression",
        model=reg,
        accuracy_val=r2,
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
        feature_names=DESCRIPTOR_COLS,
        llm_explanation=llm_explanation,
    )
    
result = run_bcf_classification("data\BCF\Grisoni_et_al_2016_EnvInt88.csv")
print("Reporte de clasificación:\n", result.llm_explanation)
