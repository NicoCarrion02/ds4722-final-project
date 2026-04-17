"""
Script de entrenamiento para el modelo final y evaluación básica.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

TARGET_COLUMN = "median_house_value"

BEST_MODEL_PARAMS = {
    "n_estimators": 205,
    "max_features": 6,
    "n_jobs": -1,
    "random_state": 42,
}

def train_best_model(processed_train_data_path: str, model_save_path: str):
    """
    INSTRUCCIONES:
    1. Carga los datos de entrenamiento procesados (que ya pasaron por `build_features.py`).
    2. Separa las características (X) de la etiqueta a predecir (y = 'median_house_value').
    3. Instancia tu mejor modelo encontrado después de la fase de experimentación y "fine tuning"
       (Por ejemplo: RandomForestRegressor con los mejores hiperparámetros).
    4. Entrena el modelo haciendo fit(X, y).
    5. Guarda el modelo entrenado en `model_save_path` (ej. 'models/best_model.pkl') usando joblib.dump().
    """
    train_path = Path(processed_train_data_path)
    model_path = Path(model_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{TARGET_COLUMN}' en {train_path}.")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    model = RandomForestRegressor(**BEST_MODEL_PARAMS)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    print(f"Modelo entrenado y guardado en: {model_path}")

    return model

def evaluate_model(model_path: str, processed_test_data_path: str):
    """
    INSTRUCCIONES:
    1. Carga el modelo guardado con joblib.load().
    2. Carga los datos de prueba preprocesados.
    3. Genera predicciones (y_pred) sobre los datos de prueba usando predict().
    4. Compara y_pred con las etiquetas reales calculando el RMSE y repórtalo en la terminal.
    """
    model_path = Path(model_path)
    test_path = Path(processed_test_data_path)

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{TARGET_COLUMN}' en {test_path}.")

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"RMSE en el set de prueba: {rmse:.4f}")
    return rmse

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent

    PROCESSED_TRAIN_PATH = project_root / "data" / "processed" / "train_set.csv"
    PROCESSED_TEST_PATH = project_root / "data" / "processed" / "test_set.csv"
    MODEL_OUTPUT_PATH = project_root / "models" / "best_model.pkl"

    train_best_model(str(PROCESSED_TRAIN_PATH), str(MODEL_OUTPUT_PATH))
    evaluate_model(str(MODEL_OUTPUT_PATH), str(PROCESSED_TEST_PATH))
