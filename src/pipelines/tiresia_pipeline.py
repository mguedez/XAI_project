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
from src.llm.gemini_explainer import explicar_con_gemini

def _generate_tiresia_llm_explanation(
    shap_explainer, 
    lime_explainer,
    X_test, 
    y_pred, 
    feature_names, 
    model, 
    task_type="Classification"
) -> str | None:
    """
    Genera explicación LLM global para el modelo TIRESIA.
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
        lime_exp_obj = lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=5,
        )
        
        # Extraer features y pesos
        lime_list = lime_exp_obj.as_list()
        lima_text = f"Instancia {idx+1} (Predicción: {pred}):\n"
        for feature_desc, weight in lime_list:
            lima_text += f"  • {feature_desc}: {weight:.4f}\n"
        lime_explanations.append(lima_text)
    
    lime_global_summary = "\n".join(lime_explanations)
    
    # Generar explicación con LLM
    explanation = explicar_con_gemini(
        shap_local=f"Predicciones realizadas sobre {len(y_pred)} instancias de prueba",
        shap_global=f"Descriptores más importantes según SHAP:\n{shap_global_summary}",
        lime_exp=f"Análisis LIME local (interpretación lineal de contribuciones):\n{lime_global_summary}",
        pred=f"Accuracy del modelo",
        task_type=task_type,
        expertise_level="beginner",
        domain="IA",
        descriptor_names=feature_names,
    )
    return explanation

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
    feature_names: list[str]
    llm_explanation: str | None = None

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

    # Generar explicación LLM global (opcional)
    llm_explanation = _generate_tiresia_llm_explanation(
        shap_explainer=shap_exp,
        lime_explainer=lime_exp,
        X_test=X_test,
        y_pred=y_pred,
        feature_names=feature_cols,
        model=rf,
        task_type="Toxicity Classification",
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
        feature_names=feature_cols,
        llm_explanation=llm_explanation,
    )

result = run_tiresia_pipeline("data/TIRESIA/ci2c01126_si_002.txt")
print("Reporte de clasificación:\n", result.llm_explanation)