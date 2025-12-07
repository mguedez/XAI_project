import pandas as pd
import shap
from src.config import RANDOM_STATE

def build_shap_classifier_explainer(model, X_train: pd.DataFrame):
    """
    Crea un explainer agnóstico para clasificación, usando predict_proba.
    """
    X_background = shap.utils.sample(X_train, 100, random_state=RANDOM_STATE)
    explainer = shap.Explainer(model.predict_proba, X_background)
    return explainer

def build_shap_regressor_explainer(model, X_train: pd.DataFrame):
    """
    Crea un explainer agnóstico para regresión, usando predict.
    """
    X_background = shap.utils.sample(X_train, 100, random_state=RANDOM_STATE)
    explainer = shap.Explainer(model.predict, X_background)
    return explainer
