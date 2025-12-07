import numpy as np
import pandas as pd
from typing import Union

# función creada por copilot (hay que revisarla)

def clean_descriptors(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    label_col: str = "LABEL",
    corr_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Limpia un DataFrame de descriptores moleculares.

    Pasos realizados:
    - Elimina columnas que contienen valores nulos (cualquier NaN).
    - Elimina columnas constantes (una sola categoría/valor).
    - Elimina columnas altamente correladas (Pearson abs(corr) > corr_threshold).

    Mantiene las columnas `smiles_col` y `label_col` intactas si existen.

    Devuelve una copia del DataFrame con las columnas no deseadas eliminadas.
    """
    df = df.copy()

    # Identificar columnas de descriptor (excluir smiles y label si existen)
    desc_cols = [c for c in df.columns if c not in (smiles_col, label_col)]
    if len(desc_cols) == 0:
        return df

    # 1) eliminar columnas con nulos
    cols_with_nulls = [c for c in desc_cols if df[c].isnull().any()]
    if cols_with_nulls:
        df.drop(columns=cols_with_nulls, inplace=True)

    # refrescar lista de descriptores
    desc_cols = [c for c in df.columns if c not in (smiles_col, label_col)]
    if len(desc_cols) == 0:
        return df

    # 2) eliminar columnas constantes (una sola valor único)
    const_cols = [c for c in desc_cols if df[c].nunique(dropna=False) <= 1]
    if const_cols:
        df.drop(columns=const_cols, inplace=True)

    # refrescar lista de descriptores
    desc_cols = [c for c in df.columns if c not in (smiles_col, label_col)]
    if len(desc_cols) == 0:
        return df

    # 3) eliminar columnas altamente correladas (mantener la primera, borrar la segunda)
    # calcular matriz de correlación (valores absolutos)
    corr_matrix = df[desc_cols].corr().abs()
    # tomar triangulo superior sin diagonal
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)

    return df
