import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(
    page_title="Sistema Inteligente de Sismos - IGP",
    page_icon="🌎",
    layout="wide"
)

# Título principal
st.title("🌎 Sistema Inteligente para Análisis y Predicción Referencial de Sismos")
st.write("Proyecto académico desarrollado para la asignatura Sistemas Inteligentes.")

# Mensaje ético
st.warning(
    "Este sistema tiene fines académicos. No predice sismos reales con exactitud "
    "ni reemplaza información oficial del Instituto Geofísico del Perú."
)

# Cargar modelo
try:
    modelo = joblib.load("modelo/modelo_sismos.pkl")
except:
    st.error("No se encontró el modelo entrenado. Ejecute primero entrenamiento_modelo.py")
    st.stop()

# Cargar datos
try:
    df = pd.read_csv("data/sismos_igp.csv")
    df.columns = df.columns.str.lower().str.strip()
except:
    st.error("No se encontró el dataset en la carpeta data.")
    st.stop()

# Menú lateral
st.sidebar.header("Parámetros del evento sísmico")

latitud = st.sidebar.number_input(
    "Latitud",
    min_value=-20.0,
    max_value=0.0,
    value=-12.0,
    step=0.1
)

longitud = st.sidebar.number_input(
    "Longitud",
    min_value=-85.0,
    max_value=-65.0,
    value=-77.0,
    step=0.1
)

profundidad = st.sidebar.number_input(
    "Profundidad en km",
    min_value=0.0,
    max_value=700.0,
    value=50.0,
    step=1.0
)

# Botón de predicción
if st.sidebar.button("Estimar magnitud"):
    entrada = pd.DataFrame({
        "latitud": [latitud],
        "longitud": [longitud],
        "profundidad": [profundidad]
    })

    prediccion = modelo.predict(entrada)[0]

    st.subheader("Resultado de la estimación")
    st.metric("Magnitud estimada", f"{prediccion:.2f}")

    if prediccion < 4:
        st.success("Nivel referencial: sismo leve")
    elif prediccion < 6:
        st.warning("Nivel referencial: sismo moderado")
    else:
        st.error("Nivel referencial: sismo fuerte")

# Mostrar dataset
st.subheader("Vista previa del dataset")
st.dataframe(df.head(20))

# Estadísticas básicas
st.subheader("Estadísticas del dataset")
st.write(df.describe())

# Gráfico de magnitudes
if "magnitud" in df.columns:
    st.subheader("Distribución de magnitudes sísmicas")

    fig, ax = plt.subplots()
    ax.hist(df["magnitud"].dropna(), bins=20)
    ax.set_xlabel("Magnitud")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Histograma de magnitudes")

    st.pyplot(fig)

# Gráfico profundidad vs magnitud
if "profundidad" in df.columns and "magnitud" in df.columns:
    st.subheader("Relación entre profundidad y magnitud")

    fig2, ax2 = plt.subplots()
    ax2.scatter(df["profundidad"], df["magnitud"])
    ax2.set_xlabel("Profundidad")
    ax2.set_ylabel("Magnitud")
    ax2.set_title("Profundidad vs Magnitud")

    st.pyplot(fig2)
