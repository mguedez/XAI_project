from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.config import RANDOM_STATE, N_JOBS

def build_rf_classifier(params: dict | None = None) -> RandomForestClassifier:
    base_params = dict(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    if params:
        base_params.update(params)
    return RandomForestClassifier(**base_params)

def build_rf_regressor(params: dict | None = None) -> RandomForestRegressor:
    base_params = dict(
        n_estimators=400,   # igual que en tu notebook de BCF
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    if params:
        base_params.update(params)
    return RandomForestRegressor(**base_params)
