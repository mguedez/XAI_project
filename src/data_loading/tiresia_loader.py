import pandas as pd
from pathlib import Path
from rdkit import Chem

def load_raw_tiresia(path: str | Path) -> pd.DataFrame:
    """
    Carga el archivo TIRESIA original (ci2c01126_si_002.txt).

    Asume dos columnas: smiles y label.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=",", header=None) 
    df.columns = ["SMILES", "LABEL"]
    return df

def clean_smiles_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce la lógica del notebook (preprocesamiento):
    - Convierte SMILES a Mol
    - Descarta los que no se puedan parsear
    - Canonicaliza SMILES
    - Mapea 'Toxicant' -> 1, el resto -> 0
    """
    clean_mols = []
    clean_sm = []
    labels = []

    for s, y in zip(df["SMILES"], df["LABEL"]):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            smiles_clean = Chem.MolToSmiles(mol, canonical=True)
            clean_mols.append(mol)
            clean_sm.append(smiles_clean)
            labels.append(1 if y == "Toxicant" else 0)

    df_clean = pd.DataFrame({"SMILES": clean_sm, "LABEL": labels})
    print(df_clean["LABEL"].value_counts())
    return df_clean
