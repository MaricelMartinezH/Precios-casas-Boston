import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from pipelines.inference_pipeline import (
    generate_predictions,
    load_model,
    transform_data,
)

MODEL_PATH = Path("data/06_models/train_pipeline_model.joblib")
PREPROCESSOR_PATH = Path("data/03_primary/preprocessor.joblib")

FEATURES = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "black",
    "lstat",
]


@st.cache_resource
def load_pipeline() -> tuple[object, object]:
    model = load_model(MODEL_PATH)
    preprocessor = load_model(PREPROCESSOR_PATH)
    return model, preprocessor


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    model, preprocessor = load_pipeline()
    transformed_data = transform_data(df, preprocessor)
    predictions = generate_predictions(model, transformed_data)
    return pd.DataFrame({"medv_predicho": predictions})


st.set_page_config(
    page_title="Predicción de precios de casas de Boston",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Predicción de precios de casas de Boston")
st.write(
    "Aplicación para predecir el valor mediano de las viviendas a partir de sus características."
)

tab_individual, tab_batch = st.tabs(["Predicción individual", "Predicción por lote"])

with tab_individual:
    st.header("Predicción individual")
    st.write("Ingresa las características de una vivienda.")

    values = {}

    col1, col2, col3 = st.columns(3)

    with col1:
        values["crim"] = st.number_input("crim", value=0.00632, format="%.5f")
        values["zn"] = st.number_input("zn", value=18.0)
        values["indus"] = st.number_input("indus", value=2.31)
        values["chas"] = st.number_input("chas", value=0.0, min_value=0.0, max_value=1.0)
        values["nox"] = st.number_input("nox", value=0.538, format="%.3f")

    with col2:
        values["rm"] = st.number_input("rm", value=6.575)
        values["age"] = st.number_input("age", value=65.2)
        values["dis"] = st.number_input("dis", value=4.0900)
        values["rad"] = st.number_input("rad", value=1.0)
        values["tax"] = st.number_input("tax", value=296.0)

    with col3:
        values["ptratio"] = st.number_input("ptratio", value=15.3)
        values["black"] = st.number_input("black", value=396.9)
        values["lstat"] = st.number_input("lstat", value=4.98)

    if st.button("Predecir precio", type="primary"):
        input_df = pd.DataFrame([values])

        try:
            result = predict_dataframe(input_df)
            prediction = result["medv_predicho"].iloc[0]

            st.success(f"Precio mediano estimado: ${prediction:.2f} mil")

        except Exception as error:
            st.error(f"No fue posible realizar la predicción: {error}")


with tab_batch:
    st.header("Predicción por lote")
    st.write("Carga un archivo CSV o Parquet con las 13 variables de entrada.")

    uploaded_file = st.file_uploader(
        "Selecciona el archivo de datos",
        type=["csv", "parquet"],
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_parquet(io.BytesIO(uploaded_file.getvalue()))

            st.subheader("Datos cargados")
            st.dataframe(input_df)

            missing = [feature for feature in FEATURES if feature not in input_df.columns]

            if missing:
                st.error("Faltan las siguientes columnas requeridas: " + ", ".join(missing))
            elif st.button("Generar predicciones", type="primary"):
                result = predict_dataframe(input_df)

                st.subheader("Resultados")
                st.dataframe(result)

                csv = result.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Descargar predicciones",
                    data=csv,
                    file_name="predicciones_boston.csv",
                    mime="text/csv",
                )

        except Exception as error:
            st.error(f"No fue posible procesar el archivo: {error}")
