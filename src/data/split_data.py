"""
Script para dividir los datos en conjunto de entrenamiento y conjunto de prueba.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

def split_and_save_data(raw_data_path: str, interim_data_path: str):
    """
    INSTRUCCIONES:
    1. Lee el archivo CSV descargado previamente en `raw_data_path` usando pandas.
    2. Separa los datos con `train_test_split()`. Te recomendamos un test_size=0.2 y random_state=42.
    3. (Opcional pero recomendado) Puedes usar `StratifiedShuffleSplit` basado en la variable
       del ingreso medio (median_income) para que la muestra sea representativa.
    4. Guarda los archivos resultantes (ej. train_set.csv y test_set.csv) en la carpeta `interim_data_path`.
    """
    raw_path = Path(raw_data_path)
    interim_path = Path(interim_data_path)
    interim_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"No existe el archivo de datos: {raw_path}")
    
    housing = pd.read_csv(raw_path)

    # Estratificación por rangos de median_income
    # http://14.139.161.31/OddSem-0822-1122/Hands-On_Machine_Learning_with_Scikit-Learn-Keras-and-TensorFlow-2nd-Edition-Aurelien-Geron.pdf
    # bins obtenidos de la página 54 (pdf 84)
    housing["income_range"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, float("inf")],
        labels=[1, 2, 3, 4, 5]
    )

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in splitter.split(housing, housing["income_range"]):
        train_set = housing.iloc[train_idx].copy()
        test_set = housing.iloc[test_idx].copy()

    for dataset in (train_set, test_set):
        dataset.drop(columns=["income_range"], inplace=True)

    train_set.to_csv(interim_path / "train_set.csv", index=False)
    test_set.to_csv(interim_path / "test_set.csv", index=False)

    return train_set, test_set

if __name__ == "__main__":
    RAW_PATH = "data/raw/housing/housing.csv"
    INTERIM_PATH = "data/interim/"
    split_and_save_data(RAW_PATH, INTERIM_PATH)
    print(f"Train y test guardados en: {INTERIM_PATH}")
