# src/preprocessing/tiresia_descriptors.py

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


def compute_rdkit_descriptors(df: pd.DataFrame,
                              smiles_col: str = "smiles",
                              label_col: str = None) -> pd.DataFrame:
    """
    Calcula TODOS los descriptores RDKit disponibles en Descriptors.descList
    siguiendo exactamente la implementación del notebook TIRESIA.

    Parameters
    ----------
    df : pd.DataFrame
        Debe contener al menos la columna de SMILES.
    smiles_col : str
        Nombre de la columna de SMILES.
    label_col : str | None
        Nombre de la columna de etiqueta (opcional).

    Returns
    -------
    df_desc : pd.DataFrame
        DataFrame completo con los descriptores RDKit.
    """

    descriptores_list = []  # lista final de diccionarios

    for idx, row in df.iterrows():
        smile = row[smiles_col]
        mol = Chem.MolFromSmiles(smile)
        label = row[label_col] if label_col is not None else None

        if mol is None:
            continue  # descarta SMILES inválido
        descriptor_dict = {"SMILES": smile}
        if label_col is not None:
            descriptor_dict[label_col] = label
        
        # calcula TODOS los descriptores de RDKit
        for descriptor_name, descriptor_function in Descriptors.descList:
            descriptor_dict[descriptor_name] = descriptor_function(mol)

        descriptores_list.append(descriptor_dict)

    # Crear el DataFrame final
    df_desc = pd.DataFrame(descriptores_list)

    return df_desc
