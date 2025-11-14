# Análisis de demanda de bicicletas en Seúl

Este proyecto contiene:

- Una aplicación en **Streamlit** (`app.py`) para explorar los datos.
- Una carpeta `notebooks/` con notebooks de análisis exploratorio.
- El conjunto de datos `SeoulBikeData.csv` en la carpeta raíz.

## Requisitos

Antes de empezar, necesitas tener instalado:

- [Python](https://www.python.org/) (3.9 o superior recomendado)
- Un gestor de paquetes:
  - `pip` (viene con muchas instalaciones de Python), **o**
  - `conda` (si usas Anaconda/Miniconda)

## Crear entorno e instalar dependencias

### Opción 1: Usando `pip` directamente

Desde la carpeta raíz del proyecto ejecuta en la terminal:

```bash
# (Opcional) Crear y activar un entorno virtual
python -m venv .venv
# En Windows
.venv\Scripts\activate
# En macOS / Linux
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
