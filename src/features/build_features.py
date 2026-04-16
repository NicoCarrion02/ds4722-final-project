"""
Módulo para limpieza y enriquecimiento (Feature Engineering) usando funciones simples.
"""

import pandas as pd
from pathlib import Path

TARGET_COLUMN = "median_house_value"
CATEGORICAL_COLUMN = "ocean_proximity"
CATEGORIES = ["INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

def clean_data(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Limpia el DataFrame imputando valores faltantes numéricos con la mediana.
    """
    df_clean = df.copy()

    drop_nulls = kwargs.get("drop_nulls", False)
    if drop_nulls:
        df_clean = df_clean.dropna()
        return df_clean

    numeric_cols = df_clean.select_dtypes(include="number").columns
    for col in numeric_cols:
        if col == TARGET_COLUMN:
            continue
        if df_clean[col].isna().any():
            median_value = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_value)

    return df_clean

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables derivadas útiles para el modelo.

    Crea ratios que suelen capturar mejor la información que las variables
    originales por separado.
    """
    df_featured = df.copy()

    # Evita división entre cero reemplazando 0 por NA temporalmente.
    households = df_featured["households"].replace(0, pd.NA)
    total_rooms = df_featured["total_rooms"].replace(0, pd.NA)

    df_featured["rooms_per_household"] = df_featured["total_rooms"] / households
    df_featured["population_per_household"] = df_featured["population"] / households
    df_featured["bedrooms_per_room"] = df_featured["total_bedrooms"] / total_rooms

    # Si alguna división generó NA por ceros, se vuelve a imputar con mediana.
    ratio_cols = [
        "rooms_per_household",
        "population_per_household",
        "bedrooms_per_room",
    ]
    for col in ratio_cols:
        if df_featured[col].isna().any():
            df_featured[col] = df_featured[col].fillna(0)

    return df_featured


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica CATEGORICAL_COLUMN con one-hot encoding en CATEGORIES.
    """
    df_encoded = df.copy()

    # No usa pd.get_dummies para mantener control total sobre el proceso y evitar columnas inesperadas o faltantes.
    for category in CATEGORIES:
        df_encoded[f"{CATEGORICAL_COLUMN}_{category}"] = df_encoded[CATEGORICAL_COLUMN].apply(lambda x: 1 if x == category else 0)

    df_encoded.drop(columns=[CATEGORICAL_COLUMN], inplace=True)

    return df_encoded

def preprocess_pipeline(df: pd.DataFrame, clean_data_kwargs: dict = {}) -> pd.DataFrame:
    """
    Función orquestadora que toma el DataFrame crudo y aplica limpieza y enriquecimiento.
    """
    df_clean = clean_data(df, **clean_data_kwargs)
    df_featured = create_features(df_clean)
    df_processed = encode_categorical(df_featured)

    return df_processed

if __name__ == "__main__":
    # Get project root (src/features -> src -> project root)
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "interim"
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["train_set.csv", "test_set.csv"]
    for dataset in datasets:
        df = pd.read_csv(data_dir / dataset)
        clean_kwargs = {"drop_nulls": True} if dataset == "test_set.csv" else {}
        df_prepared = preprocess_pipeline(df, clean_data_kwargs=clean_kwargs)
        df_prepared.to_csv(output_dir / dataset, index=False)
