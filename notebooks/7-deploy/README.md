# Despliegue del modelo con Streamlit

## Descripción

Esta aplicación permite utilizar el modelo de predicción de precios de casas de Boston mediante una interfaz web desarrollada con Streamlit.

La aplicación ofrece dos modalidades:

1. **Predicción individual:** permite ingresar manualmente las características de una vivienda y obtener una predicción.
2. **Predicción por lote:** permite cargar un archivo CSV con múltiples registros, visualizar las predicciones y descargar los resultados.

## Ejecución local

Desde la raíz del proyecto:

`uv run streamlit run demo_app.py`

La aplicación estará disponible en `http://localhost:8501`.

## Predicción individual

En la pestaña **Predicción individual**, se ingresan las siguientes variables:

- `crim`
- `zn`
- `indus`
- `chas`
- `nox`
- `rm`
- `age`
- `dis`
- `rad`
- `tax`
- `ptratio`
- `black`
- `lstat`

Al seleccionar **Predecir precio**, la aplicación muestra el valor mediano estimado de la vivienda.

## Predicción por lote

En la pestaña **Predicción por lote**, se puede cargar un archivo CSV o Parquet que contenga las 13 variables de entrada requeridas.

La aplicación carga el archivo, verifica las columnas requeridas, muestra los datos, genera las predicciones y permite descargar los resultados en formato CSV.

### Archivo de salida

El resultado contiene las predicciones en la columna `medv_predicho`.

## Modelo utilizado

La aplicación utiliza el modelo entrenado:

`data/06_models/train_pipeline_model.joblib`

y el preprocesador:

`data/03_primary/preprocessor.joblib`

La aplicación reutiliza el pipeline de inferencia desarrollado para el proyecto.

## Despliegue

La aplicación será publicada mediante Streamlit Community Cloud a partir del repositorio de GitHub.

**URL pública:** pendiente de despliegue.
