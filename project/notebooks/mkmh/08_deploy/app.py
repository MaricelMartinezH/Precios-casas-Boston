# Streamlit demo del modelo de precios de vivienda (Boston Housing)
# Issue 8 - Notebook 10 - Proyecto Precios-casas-Boston
#
# Como ejecutar:
#   streamlit run project/notebooks/mkmh/08_deploy/app.py
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load


@st.cache_resource
def load_model_and_reference(model_path, split_path):
    model = load(model_path)
    split_data = load(split_path)
    x_train = split_data["x_train"]
    return model, x_train


def build_input_widgets(x_train):
    user_data = {}
    columns = list(x_train.columns)
    n_cols = 3
    cols = st.columns(n_cols)
    for i, column in enumerate(columns):
        col = cols[i % n_cols]
        series = x_train[column]
        with col:
            if pd.api.types.is_numeric_dtype(series):
                default_value = float(series.median())
                min_value = float(series.min())
                max_value = float(series.max())
                step = (max_value - min_value) / 100 if max_value > min_value else 1.0
                user_data[column] = st.number_input(
                    label=column,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    step=step,
                )
            else:
                options = sorted(series.dropna().unique().tolist())
                user_data[column] = st.selectbox(label=column, options=options, index=0)
    return pd.DataFrame([user_data])


def preprocess_batch_data(df, x_train):
    """Preprocesa el CSV subido por el usuario para que coincida con el formato
    y tipos de dato esperados por el pipeline, usando x_train como referencia
    de columnas numericas, columnas categoricas y categorias validas.

    Args:
        df (pd.DataFrame): datos originales subidos por el usuario
        x_train (pd.DataFrame): datos de entrenamiento, usados como referencia

    Returns:
        pd.DataFrame: dataframe limpio, listo para pasar al pipeline
    """
    processed_df = df.copy()

    for column in x_train.columns:
        if column not in processed_df.columns:
            continue

        if pd.api.types.is_numeric_dtype(x_train[column]):
            # fuerza a numerico, valores invalidos quedan como NaN
            processed_df[column] = pd.to_numeric(processed_df[column], errors="coerce")
        else:
            # normaliza texto (espacios, mayusculas/minusculas) contra las
            # categorias vistas en entrenamiento, ej: " Male ", "MALE" -> "Male"
            valid_categories = x_train[column].dropna().unique().tolist()
            category_map = {str(cat).strip().lower(): cat for cat in valid_categories}
            processed_df[column] = processed_df[column].map(
                lambda x, cm=category_map: cm.get(str(x).strip().lower(), x)
            )

    return processed_df


def individual_prediction_tab(model, x_train):
    st.subheader("Ingresa las caracteristicas de la vivienda")
    df_user_data = build_input_widgets(x_train)

    if st.button("Predecir precio"):
        prediction = model.predict(df_user_data)[0]
        st.title(f"Precio estimado: ${prediction * 1000:,.0f} USD")
        st.caption(
            "El modelo predice medv (precio mediano de la vivienda) en miles de "
            "USD, escala del dataset original de 1978."
        )


def batch_prediction_tab(model, x_train):
    st.subheader("Sube un archivo CSV con varias viviendas")
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

    required_cols = list(x_train.columns)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = preprocess_batch_data(df, x_train)
            st.write("Vista previa de los datos cargados:")
            st.dataframe(df.head())

            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.warning("Advertencia: faltan estas columnas: " + ", ".join(missing_cols))
                st.info("Columnas requeridas: " + ", ".join(required_cols))
            elif st.button("Predecir precios"):
                with st.spinner("Calculando predicciones..."):
                    predictions = model.predict(df[required_cols])

                result_df = df.copy()
                result_df["precio_estimado_miles_usd"] = predictions

                st.success("Predicciones completadas")
                st.subheader("Resultados")
                st.dataframe(result_df)

                st.metric(
                    "Precio promedio estimado",
                    f"${predictions.mean() * 1000:,.0f} USD",
                )

                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Descargar resultados en CSV",
                    data=csv,
                    file_name="predicciones_precios_vivienda.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
            st.info("Verifica que el archivo CSV tenga el formato correcto.")
    else:
        st.info("Sube un archivo CSV con las columnas requeridas.")
        st.subheader("Ejemplo de formato (primeras filas de entrenamiento):")
        st.dataframe(x_train.head(3))


def main():
    st.set_page_config(page_title="Precios de Vivienda - Boston Housing", page_icon="🏠")

    project_root = Path(__file__).resolve().parents[3]
    model_path = project_root / "data" / "06_models" / "best_model.joblib"
    split_path = project_root / "data" / "07_model_output" / "train_test_split.joblib"

    model, x_train = load_model_and_reference(model_path, split_path)

    st.header("Cuanto vale esta vivienda?")
    st.write(
        "Demo del modelo gradient_boosting_tuned, entrenado sobre el dataset "
        "Boston Housing (Harrison y Rubinfeld, 1978)."
    )

    tab1, tab2 = st.tabs(["Prediccion individual", "Prediccion por lote (CSV)"])

    with tab1:
        individual_prediction_tab(model, x_train)

    with tab2:
        batch_prediction_tab(model, x_train)


if __name__ == "__main__":
    main()
