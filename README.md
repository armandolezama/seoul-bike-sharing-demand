# Análisis de demanda de bicicletas en Seúl

Este proyecto contiene:

- Una aplicación en **Streamlit** (`app.py`) para explorar los datos.
- Una carpeta `notebooks/` con notebooks de análisis exploratorio.
- El conjunto de datos `SeoulBikeData.csv` en la carpeta raíz.

Estructura sugerida del proyecto:

```text
seoul-bike-project/
├── app.py               # Aplicación de Streamlit
├── SeoulBikeData.csv    # Dataset original
├── requirements.txt
└── notebooks/
    └── analisis.ipynb   # Notebook de análisis
```

La notebook está configurada para leer el dataset mediante una ruta relativa desde la carpeta raíz.

---

## Requisitos previos

Antes de empezar, necesitas tener instalado:

- [Python](https://www.python.org/) (3.9 o superior recomendado)
- Un gestor de paquetes:
  - `pip` (suele venir con muchas instalaciones de Python), **o**
  - `conda` (Anaconda/Miniconda)

---

## Crear entorno e instalar dependencias

### Opción 1: Usando `pip`

Desde la carpeta raíz del proyecto (`seoul-bike-project/`):

```bash
# (Opcional) Crear un entorno virtual
python -m venv .venv

# Activar el entorno virtual
# En Windows:
.venv\Scripts\activate
# En macOS / Linux:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 2: Usando `conda`

```bash
# Crear un entorno nuevo con Python
conda create -n seoul-bike-env python=3.11 pip

# Activar el entorno
conda activate seoul-bike-env

# Instalar dependencias usando pip dentro del entorno conda
pip install -r requirements.txt
```

---

## Ejecutar la aplicación de Streamlit

Desde la carpeta raíz del proyecto:

```bash
streamlit run app.py
```

Esto abrirá la aplicación en tu navegador o mostrará una URL local para acceder.

---

## Ejecutar los notebooks

1. Asegúrate de tener activado el mismo entorno donde instalaste las dependencias.
2. Desde la carpeta raíz del proyecto, lanza Jupyter:

```bash
jupyter notebook
# o
jupyter lab
```

3. Navega a `notebooks/analisis.ipynb` y ábrelo.

La notebook asume que el archivo `SeoulBikeData.csv` está en la carpeta raíz del proyecto. Si cambias la ubicación del dataset, actualiza también la ruta en el código de la notebook y de la app de Streamlit.

## Demo en Streamlit

Puedes probar la aplicación en línea aquí:

👉 [Seoul Bike Sharing – Streamlit App](https://seoul-bike-sharing-demand.onrender.com/)
