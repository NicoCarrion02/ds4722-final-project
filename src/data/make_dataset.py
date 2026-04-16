"""
Script para descargar y extraer los datos originales del proyecto.
"""

import urllib.request
import tarfile
from pathlib import Path

def fetch_housing_data(housing_url: str, housing_path: str):
    """
    Descarga el archivo .tgz y extrae su contenido en `housing_path`.

    Parameters
    ----------
    housing_url : str
        URL del archivo comprimido (.tgz).
    housing_path : str
        Carpeta destino donde se guardará y extraerá el dataset.
    """
    destination = Path(housing_path)
    destination.mkdir(parents=True, exist_ok=True)

    tgz_path = destination / "housing.tgz"

    urllib.request.urlretrieve(housing_url, tgz_path)

    with tarfile.open(tgz_path) as tar:
        tar.extractall(path=destination)

if __name__ == "__main__":
    URL = "https://github.com/ageron/data/raw/main/housing.tgz"
    PATH = "data/raw/"
    fetch_housing_data(URL, PATH)
    print(f"Datos descargados y extraídos en: {PATH}")
