import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from src.config import RANDOM_STATE

def build_lime_classifier_explainer(X_train: pd.DataFrame,
                                    feature_names: list[str],
                                    class_names: list[str]) -> LimeTabularExplainer:
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=RANDOM_STATE,
    )
    return explainer

def build_lime_regressor_explainer(X_train: pd.DataFrame,
                                   feature_names: list[str]) -> LimeTabularExplainer:
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        mode="regression",
        random_state=RANDOM_STATE,
    )
    return explainer

def explain_instance_lime(explainer: LimeTabularExplainer,
                          instance: np.ndarray,
                          predict_fn,
                          num_features: int):
    """
    Wrapper fino sobre explain_instance de LIME.
    """
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=num_features,
    )
    return exp
