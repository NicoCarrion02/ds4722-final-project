"""
API Básica usando FastAPI para servir el modelo entrenado.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from enum import Enum

# Permite ejecutar este archivo directamente: python src/api/main.py
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.features.build_features import preprocess_pipeline

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler para cargar el modelo al iniciar.
    """
    global model
    try:
        model = joblib.load("models/best_model.pkl")
        print("Modelo cargado correctamente")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
    yield
from enum import Enum

# Permite ejecutar este archivo directamente: python src/api/main.py
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.features.build_features import preprocess_pipeline

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler para cargar el modelo al iniciar.
    """
    global model
    try:
        model = joblib.load("models/best_model.pkl")
        print("Modelo cargado correctamente")
    except Exception as e:
        print(f"Error cargando modelo: {e}")
    yield

# Inicializamos la app
app = FastAPI(title="API de Predicción de Precios de Vivienda (California)", version="1.0", lifespan=lifespan)

class OceanProximity(str, Enum):
    LESS_THAN_1H_OCEAN = "<1H OCEAN"
    INLAND = "INLAND"
    NEAR_OCEAN = "NEAR OCEAN"
    NEAR_BAY = "NEAR BAY"
    ISLAND = "ISLAND"

app = FastAPI(title="API de Predicción de Precios de Vivienda (California)", version="1.0", lifespan=lifespan)

class OceanProximity(str, Enum):
    LESS_THAN_1H_OCEAN = "<1H OCEAN"
    INLAND = "INLAND"
    NEAR_OCEAN = "NEAR OCEAN"
    NEAR_BAY = "NEAR BAY"
    ISLAND = "ISLAND"

class HousingFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: OceanProximity
    ocean_proximity: OceanProximity

@app.get("/")
def home():
    return {"mensaje": "Bienvenido a la API del Proyecto Final de Ciencia de Datos"}

@app.post("/predict")
def predict_price(features: HousingFeatures):
    """
    INSTRUCCIONES:
    1. Convierte el objeto 'features' (Pydantic) a un formato que Scikit-Learn entienda (ej un DataFrame o Array 2D).
       Toma en cuenta que el modelo en producción espera exactamente las mismas columnas que usaste para entrenar.
    2. Usa model.predict()
    3. Retorna la predicción en un diccionario, ej: {"predicted_price": 250000.0}
    """
    if model is None:
        return {"error": "El modelo no se ha cargado."}
    
    # 1. Convertir input a DataFrame
    input_data = pd.DataFrame([features.model_dump()])

    # 2. Aplicar pipeline de features (MISMA lógica que entrenamiento)
    processed_data = preprocess_pipeline(input_data, ignore_cleaning=True) # asumiendo que el input ya viene limpio, solo queremos crear features y codificar

    # 3. Asegurar mismo orden de columnas
    processed_data = processed_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # 4. Predecir
    prediction = model.predict(processed_data)[0]

    return {"predicted_price": float(prediction)}

# Instrucciones para correr la API localmente:
# En la terminal, ejecuta:
# uvicorn src.api.main:app --reload

"""
Ejemplo:

{
  "longitude": -121.95,
  "latitude": 37.11,
  "housing_median_age": 21.0,
  "total_rooms": 2387.0,
  "total_bedrooms": 357.0,
  "population": 913.0,
  "households": 341.0,
  "median_income": 7.736,
  "ocean_proximity": "<1H OCEAN"
}

Respuesta esperada:
{
  "predicted_price": 410030.9512195122
}
"""