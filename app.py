# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Seoul Bike Sharing - EDA",
    layout="wide",
)

# ---------------------------------------------------------
# 1. Carga de datos y preprocesado
# ---------------------------------------------------------

DATA_PATH = Path(__file__).parent / "SeoulBikeData.csv"


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """Carga el CSV original.

    Ajusta 'encoding' y 'sep' si tu archivo lo necesita.
    """
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    return df


@st.cache_data
def clean_and_engineer(df_raw: pd.DataFrame) -> dict:
    """Aplica las transformaciones básicas que ya usaste en el notebook.

    Devuelve un diccionario con los distintos dataframes derivados
    que usaremos en las páginas.
    """

    df = df_raw.copy()

    # 🔧 Ajusta este bloque a los nombres reales de tu CSV
    # (estos son los que comentaste que estabas usando ya normalizados).
    rename_cols = {
        "Date": "date",
        "Rented Bike Count": "rented_bike_count",
        "Hour": "hour",
        "Temperature(°C)": "temperature°c",
        "Humidity(%)": "humidity",
        "Wind speed (m/s)": "wind_speed_ms",
        "Visibility (10m)": "visibility_10m",
        "Dew point temperature(°C)": "dew_point_temperature°c",
        "Solar Radiation (MJ/m2)": "solar_radiation_mjm2",
        "Rainfall(mm)": "rainfallmm",
        "Snowfall (cm)": "snowfall_cm",
        "Seasons": "seasons",
        "Holiday": "holiday",
        "Functioning Day": "functioning_day",
    }

    df = df.rename(columns={k: v for k, v in rename_cols.items() if k in df.columns})

    # Tipos
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["hour"] = df["hour"].astype(int)
    df["seasons"] = df["seasons"].astype("category")
    df["holiday"] = df["holiday"].astype("category")
    df["functioning_day"] = df["functioning_day"].astype("category")

    # -------------------------------------------------
    # daily_df: sum de rentas por día + clima medio
    # -------------------------------------------------
    weather_cols = [
        "temperature°c",
        "humidity",
        "wind_speed_ms",
        "visibility_10m",
        "solar_radiation_mjm2",
        "rainfallmm",
        "snowfall_cm",
    ]

    daily_df = (
        df.groupby("date", as_index=False)
        .agg(
            daily_rentals=("rented_bike_count", "sum"),
            seasons=("seasons", "first"),   # cada día tiene una sola estación
            **{col: (col, "mean") for col in weather_cols}
        )
    )

    daily_df["day_of_year"] = daily_df["date"].dt.dayofyear
    daily_df["weekday"] = daily_df["date"].dt.dayofweek
    daily_df["is_weekend"] = daily_df["weekday"] >= 5

    # Holiday a nivel diario (al menos un registro "Holiday" en el día)
    holiday_daily = (
        df.groupby("date")["holiday"]
        .apply(lambda s: (s == "Holiday").any())
        .rename("is_holiday")
        .reset_index()
    )
    daily_df = daily_df.merge(holiday_daily, on="date", how="left")

    # Media móvil (para las curvas suavizadas)
    window = 7
    daily_df = daily_df.sort_values("day_of_year")
    daily_df["daily_rentals_smooth"] = (
        daily_df["daily_rentals"]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )
    for col in ["temperature°c", "humidity", "rainfallmm"]:
        daily_df[f"{col}_smooth"] = (
            daily_df[col]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )

    # -------------------------------------------------
    # season_hour_pivot: para el mapa de calor
    # -------------------------------------------------
    season_hour_summary = (
        df.groupby(["seasons", "hour"])["rented_bike_count"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    season_hour_pivot_mean = season_hour_summary.pivot(
        index="hour",
        columns="seasons",
        values="mean",
    )

    return {
        "bikes_df": df,
        "daily_df": daily_df,
        "season_hour_pivot_mean": season_hour_pivot_mean,
    }


raw_df = load_raw_data()
data = clean_and_engineer(raw_df)
bikes_df = data["bikes_df"]
daily_df = data["daily_df"]
season_hour_pivot_mean = data["season_hour_pivot_mean"]

# ---------------------------------------------------------
# 2. Páginas de la app
# ---------------------------------------------------------


def page_overview():
    st.title("Visión general de la demanda")

    st.markdown(
        "Esta sección resume la demanda anual del sistema de bicicletas compartidas, "
        "destacando estacionalidad y niveles generales de uso."
    )

    # KPI sencillos
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Renta diaria promedio",
            f"{daily_df['daily_rentals'].mean():.0f}",
        )
    with col2:
        max_day = daily_df.loc[daily_df["daily_rentals"].idxmax()]
        st.metric(
            "Día de máxima demanda",
            max_day["date"].strftime("%Y-%m-%d"),
            f"{int(max_day['daily_rentals'])} rentas",
        )
    with col3:
        season_mean = (
            daily_df.groupby("seasons")["daily_rentals"].mean().sort_values(ascending=False)
        )
        top_season = season_mean.index[0]
        st.metric(
            "Estación con mayor demanda",
            top_season,
            f"{season_mean.iloc[0]:.0f} rentas/día",
        )

    st.subheader("Curva anual de demanda por estaciones")

    fig, ax = plt.subplots(figsize=(10, 4))

    df_plot = daily_df.sort_values("day_of_year")

    season_colors = {
      "Spring": "#ccebc5",
      "Summer": "#fed9a6",
      "Autumn": "#fdd0a2",
      "Winter": "#c6dbef",
    }

    ax.plot(
        df_plot["day_of_year"],
        df_plot["daily_rentals_smooth"],
        label="Demanda diaria (media móvil 7 días)",
        color="tab:blue",
        linewidth=2,
    )

    used_labels = set()

    for season, color in season_colors.items():
        season_days = (
        df_plot.loc[df_plot["seasons"] == season, "day_of_year"]
          .dropna()
          .sort_values()
          .to_numpy()
        )

        if len(season_days) == 0:
          continue

        start = season_days[0]
        prev = season_days[0]

        for d in season_days[1:]:
            if d == prev + 1:
                prev = d
            else: 
              ax.axvspan(
                  start - 0.5,
                  prev + 0.5,
                  color=color,
                  alpha=0.5,
                  zorder=0,
                  label=season if season not in used_labels else None,
              )
              used_labels.add(season)
              start = d
              prev = d

        ax.axvspan(
            start,
            prev,
            color=color,
            alpha=0.5,
            zorder=0,
            label=season if season not in used_labels else None,
        )

    ax.set_xlabel("Día del año")
    ax.set_ylabel("Demanda diaria (rentas)")
    ax.set_title("Evolución anual de la demanda diaria de bicicletas")
    ax.legend()

    st.pyplot(fig)


def page_hour_season():
    st.title("Patrones por hora y estación")

    st.markdown(
        "Esta sección muestra cómo varía la demanda por hora del día en cada estación."
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    # Ejemplo mínimo de placeholder:
    cax = ax.imshow(season_hour_pivot_mean.values, aspect="auto", origin="lower")
    ax.set_xticks(range(len(season_hour_pivot_mean.columns)))
    ax.set_xticklabels(season_hour_pivot_mean.columns)
    ax.set_yticks(range(24))
    ax.set_yticklabels(season_hour_pivot_mean.index)
    ax.set_xlabel("Estación")
    ax.set_ylabel("Hora del día")
    fig.colorbar(cax, ax=ax, label="Demanda promedio (rentas)")
    st.pyplot(fig)


def page_climate():
    st.title("Demanda vs variables climáticas")

    st.markdown(
        "Curvas suavizadas de demanda diaria junto con temperatura, humedad y lluvia, "
        "para observar cómo el clima modula los picos y valles de uso."
    )

    fig, ax1 = plt.subplots(figsize=(10, 4))

    df_plot = daily_df.sort_values("day_of_year")

    ax1.plot(
        df_plot["day_of_year"],
        df_plot["daily_rentals_smooth"],
        label="Demanda diaria (media móvil)",
        color="tab:blue",
        linewidth=2,
    )

    ax1.set_xlabel("Día del año")
    ax1.set_ylabel("Rentas diarias de bicicletas")

    ax2 = ax1.twinx()

    ax2.plot(
        df_plot["day_of_year"],
        df_plot["temperature°c_smooth"],
        linestyle="--",
        label="Temperatura media (°C, suavizada)",
        color="tab:orange",
    )

    ax2.plot(
        df_plot["day_of_year"],
        df_plot["humidity_smooth"],
        linestyle=":",
        label="Humedad (% suavizada)",
        color="tab:green",
        linewidth=2,
    )

    ax2.plot(
        df_plot["day_of_year"],
        df_plot["rainfallmm_smooth"],
        linestyle="-.",
        label="Lluvia (mm, suavizada)",
        color="tab:red",
    )

    st.pyplot(fig)


def page_calendar():
    st.title("Demanda vs fines de semana y días festivos")

    st.markdown(
        "Curva suavizada de demanda diaria con la indicación de fines de semana "
        "y días festivos para observar el impacto del calendario."
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    df_plot = daily_df.sort_values("day_of_year")

    ax.plot(
        df_plot["day_of_year"],
        df_plot["daily_rentals_smooth"],
        label="Rentas diarias (media móvil)",
        color="tab:blue",
        linewidth=2,
    )

    weekend_days = sorted(df_plot.loc[df_plot["is_weekend"], "day_of_year"].unique())

    weekend_ranges = []
    start = None
    prev = None

    for d in weekend_days:
      if start is None:
          start = d
          prev = d
      elif d == prev + 1:
          prev = d
      else:
          weekend_ranges.append((start, prev))
          start = d
          prev = d

    if start is not None:
      weekend_ranges.append((start, prev))

    first_weekend = True

    for start, end in weekend_ranges:
      ax.axvspan(
          start - 0.5,
          end + 0.5,
          color="grey",
          alpha=0.2,
          zorder=0,
          label="Fin de semana" if first_weekend else None,
      )
      first_weekend = False

    holiday_days = sorted(df_plot.loc[df_plot["is_holiday"], "day_of_year"].unique())

    first_holiday = True

    for d in holiday_days:
      ax.axvline(
          d,
          color="red",
          linestyle="--",
          alpha=0.7,
          linewidth=1,
          label="Día festivo" if first_holiday else None,
      )
      first_holiday = False

    ax.set_xlabel("Día del año")
    ax.set_ylabel("Demanda diaria (rentas)")
    ax.set_title("Evolución anual de la demanda diaria de bicicletas")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left")

    st.pyplot(fig)

def page_distribution():
    st.title("Distribución y relaciones básicas")

    st.markdown(
        "Explora la distribución de las rentas diarias y su relación con la temperatura, "
        "filtrando por estación del año."
    )

    # Selector de estación
    estaciones = ["Todas"] + sorted(daily_df["seasons"].unique().tolist())
    season_sel = st.selectbox("Estación", estaciones, index=0)

    if season_sel == "Todas":
        df_plot = daily_df.copy()
    else:
        df_plot = daily_df[daily_df["seasons"] == season_sel].copy()

    col1, col2 = st.columns(2)

    # --- Histograma de rentas diarias ---
    with col1:
        st.subheader("Histograma de rentas diarias")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df_plot["daily_rentals"], bins=20, edgecolor="black")
        ax.set_xlabel("Rentas diarias")
        ax.set_ylabel("Frecuencia")
        ax.set_title(f"Distribución de rentas ({season_sel})")
        st.pyplot(fig)

    # --- Dispersión demanda vs temperatura ---
    with col2:
        st.subheader("Rentas vs temperatura media")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(
            df_plot["temperature°c"],
            df_plot["daily_rentals"],
            alpha=0.4,
            s=15,
        )
        ax.set_xlabel("Temperatura media diaria (°C)")
        ax.set_ylabel("Rentas diarias")
        ax.set_title(f"Demanda vs temperatura ({season_sel})")
        st.pyplot(fig)

def page_prediction_placeholder():
    st.title("Predicción de demanda (futuro)")

    st.info(
        "Esta sección está pensada para una versión futura de la app.\n\n"
        "Aquí podrás seleccionar una fecha y condiciones climáticas esperadas "
        "y el modelo entrenado estimará la demanda de bicicletas para ese día."
    )

    st.markdown(
        "- Inputs previstos:\n"
        "  - Fecha objetivo.\n"
        "  - Temperatura esperada.\n"
        "  - Humedad esperada.\n"
        "  - Lluvia prevista.\n"
        "- El backend de esta página cargará un modelo entrenado en otro notebook "
        "y devolverá una estimación de `daily_rentals`, junto con un rango de confianza."
    )


# ---------------------------------------------------------
# 3. Navegación
# ---------------------------------------------------------

PAGES = {
    "Visión general": page_overview,
    "Hora vs estación": page_hour_season,
    "Distribución y relaciones": page_distribution,
    "Clima": page_climate,
    "Calendario": page_calendar,
    "Predicción (placeholder)": page_prediction_placeholder,
}

st.sidebar.title("Navegación")
page_name = st.sidebar.radio("Ir a:", list(PAGES.keys()))

# Ejecutar página seleccionada
PAGES[page_name]()
