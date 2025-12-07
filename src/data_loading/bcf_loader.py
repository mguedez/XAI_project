import pandas as pd
from pathlib import Path

def load_bcf_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset de Grisoni para BCF.

    Parameters
    ----------
    csv_path : str | Path
        Ruta al archivo CSV Grisoni_et_al_2016_EnvInt88.csv

    Returns
    -------
    df : pd.DataFrame
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    return df
