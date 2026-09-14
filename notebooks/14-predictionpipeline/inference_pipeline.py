"""Inference / Prediction pipeline para el proyecto Precios-casas-Boston.

Issue 6 — Prediction / Inference Pipeline.

Esta Issue NO entrena de nuevo el modelo ni rehace las Issues anteriores.
Usa:

* el modelo YA entrenado y persistido por `train_pipeline.py` (Issue 3):
  `data/06_models/train_pipeline_model.joblib`.
* el `ColumnTransformer` YA ajustado (`fit`) y persistido por
  `feature_pipeline_validado.py` (Issues 1 y 2):
  `data/03_primary/preprocessor.joblib`.

para generar predicciones sobre datos NUEVOS (no vistos durante
entrenamiento), sin reajustar ni el preprocesador ni el modelo.

Por qué se reutiliza el preprocesador ya ajustado (y no se vuelve a
llamar `build_preprocessor()` + `fit`): si esta Issue ajustara un
preprocesador nuevo sobre los datos de inferencia, las medias/
desviaciones del `StandardScaler`, las categorías del `OneHotEncoder` y
los bordes del `KBinsDiscretizer` serían distintos a los usados en
entrenamiento, produciendo predicciones inconsistentes con el modelo
(el punto 5 de la Issue lo marca como el riesgo principal a evitar).
Por eso `feature_pipeline_validado.py` se modificó (única modificación
a una Issue anterior, ver `save_preprocessor()` ahí) para persistir ese
preprocesador ya ajustado.

Transformaciones reutilizadas literalmente de
`pipelines.feature_pipeline_validado` (sin duplicarlas):
    - `drop_redundant_features` (elimina `tax`, igual que en Issue 2).
    - `add_rad_group_feature` (crea `rad_group`, igual que en Issue 2).
    - `flag_censored_target` (crea `medv_censored`, solo si el archivo de
      entrada trae la columna `medv`; ver docstring de `transform_data`).
    - El propio `ColumnTransformer` ya ajustado, vía `.transform()`
      (nunca `.fit()` ni `.fit_transform()`).

Y de `pipelines.train_pipeline` (sin duplicarlas):
    - `drop_leakage_columns` (excluye `binary__medv_censored` antes de
      predecir, igual que se excluyó antes de entrenar).
    - `predict` (llama a `model.predict(...)`).

Flujo:
    cargar modelo entrenado -> cargar preprocesador ajustado
    -> leer datos nuevos -> aplicar mismas transformaciones (sin fit)
    -> generar predicciones -> guardar predicciones (+ gráfica)
    -> registrar evidencia de la ejecución

Uso:
    python pipelines/inference_pipeline.py --input data/09_new_data/nuevas_casas.csv
    python pipelines/inference_pipeline.py --input nuevas_casas.csv --output mis_predicciones.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

# Raíz del proyecto: este archivo vive en <root>/pipelines/inference_pipeline.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipelines import feature_pipeline_validado as fp
from pipelines import train_pipeline as tp

# ---------------------------------------------------------------------------
# Rutas: modelo y preprocesador ya generados por Issues anteriores.
# ---------------------------------------------------------------------------
MODEL_PATH = tp.MODEL_PATH
PREPROCESSOR_PATH = fp.OUTPUT_DIR / fp.PREPROCESSOR_FILENAME

# Salida de esta Issue: predicciones, log y gráfica.
PREDICTIONS_DIR = tp.DATA_DIR / "predictions"
PREDICTIONS_PATH = PREDICTIONS_DIR / "predictions.csv"
LOG_DIR = PREDICTIONS_DIR / "logs"
LOG_PATH = LOG_DIR / "inference_log.txt"
PLOTS_DIR = PREDICTIONS_DIR / "plots"
PLOT_PATH = PLOTS_DIR / "predictions_plot.png"

# Identificador de cada registro, tal como aparece en el dataset original
# del proyecto (`Precios_Casas_Boston.csv`). Es opcional: si no viene en
# el archivo de entrada, se usa el índice de fila como identificador.
ID_COLUMN = "ID"

# Variable objetivo, igual que en el resto del proyecto.
TARGET_COL = fp.TARGET_COL  # "medv"

# Nombre de la columna de predicción: coherente con la convención del
# proyecto (`app.py` llama "precio_estimado..." al resultado; aquí se usa
# el nombre de la variable objetivo con el sufijo `_predicho`, para que
# quede claro que predice `medv`).
PREDICTION_COL = f"{TARGET_COL}_predicho"

# Columnas de entrada crudas que debe traer el archivo de datos nuevos:
# las mismas `SELECTED_FEATURES` del feature pipeline, sin el target
# (`medv` es opcional: si está presente se usa solo para el gráfico de
# comparación real-vs-predicho, nunca como entrada al modelo).
REQUIRED_INPUT_COLUMNS = [col for col in fp.SELECTED_FEATURES if col != TARGET_COL]


class InferenceDataError(ValueError):
    """Error de datos de entrada para el inference pipeline.

    Se lanza cuando el archivo de datos nuevos no tiene el formato
    esperado (columnas faltantes, archivo vacío, extensión no soportada).
    """


def load_model(model_path: Path = MODEL_PATH):
    """Carga el modelo YA entrenado por `train_pipeline.py` (Issue 3).

    No entrena nada aquí: solo deserializa el `.joblib` guardado por
    `train_pipeline.save_model`.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        msg = (
            f"No se encontró el modelo entrenado en '{model_path}'. Ejecuta "
            "primero 'python pipelines/train_pipeline.py' (Issue 3) antes de "
            "correr la inferencia."
        )
        raise FileNotFoundError(msg)
    return joblib.load(model_path)


def load_preprocessor(preprocessor_path: Path = PREPROCESSOR_PATH) -> ColumnTransformer:
    """Carga el `ColumnTransformer` YA ajustado por `feature_pipeline_validado.py`.

    Se usa solo con `.transform()`; nunca se vuelve a llamar `.fit()`
    aquí (ver docstring del módulo).
    """
    preprocessor_path = Path(preprocessor_path)
    if not preprocessor_path.exists():
        msg = (
            f"No se encontró el preprocesador ajustado en '{preprocessor_path}'. "
            "Ejecuta primero 'python pipelines/feature_pipeline_validado.py' "
            "(Issues 1 y 2) antes de correr la inferencia."
        )
        raise FileNotFoundError(msg)
    return joblib.load(preprocessor_path)


def load_new_data(
    input_path: Path, required_columns: list[str] = REQUIRED_INPUT_COLUMNS
) -> pd.DataFrame:
    """Lee los datos nuevos desde un archivo `.csv` o `.parquet`.

    Valida que estén presentes las columnas de entrada crudas que
    necesita el preprocesador (las mismas que consume
    `feature_pipeline_validado.py`, sin el target). No modifica ni
    limpia los datos todavía; eso lo hace `transform_data`.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        msg = f"No se encontró el archivo de datos nuevos: '{input_path}'."
        raise FileNotFoundError(msg)

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        new_data = pd.read_csv(input_path)
    elif suffix == ".parquet":
        new_data = pd.read_parquet(input_path, engine="pyarrow")
    else:
        msg = (
            f"Formato de archivo no soportado: '{suffix}'. Usa un archivo "
            "'.csv' o '.parquet'."
        )
        raise InferenceDataError(msg)

    if new_data.empty:
        msg = f"El archivo de datos nuevos '{input_path}' está vacío."
        raise InferenceDataError(msg)

    missing_columns = [col for col in required_columns if col not in new_data.columns]
    if missing_columns:
        msg = (
            "Faltan columnas requeridas en los datos nuevos: "
            f"{missing_columns}. Columnas requeridas: {required_columns}."
        )
        raise InferenceDataError(msg)

    return new_data


def transform_data(
    new_data: pd.DataFrame,
    preprocessor: ColumnTransformer,
    required_columns: list[str] = REQUIRED_INPUT_COLUMNS,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Aplica a los datos nuevos las MISMAS transformaciones del entrenamiento.

    Reutiliza directamente las funciones de
    `pipelines.feature_pipeline_validado` (Issues 1 y 2):
    `drop_redundant_features`, `add_rad_group_feature` y, si el archivo
    de entrada trae la columna `medv`, `flag_censored_target`.

    Nota sobre `medv_censored`: es un atributo binario derivado del
    target (`medv == 50.0`) que el `ColumnTransformer` espera como
    columna de entrada, pero que `train_pipeline.py` excluye de la
    matriz de atributos final por ser fuga de datos
    (`LEAKAGE_COLUMNS = ["binary__medv_censored"]`). Por eso, cuando los
    datos nuevos NO traen `medv` (el caso real de inferencia sobre
    viviendas sin precio conocido), se usa un valor constante (0) que
    solo sirve para que el preprocesador no falle por columna faltante:
    como esa columna se descarta después con `drop_leakage_columns`, su
    valor no influye en la predicción.

    El resultado final ya tiene excluida `binary__medv_censored`, igual
    que las features que consume el modelo en `train_pipeline.py`.
    """
    features = new_data[required_columns].copy()

    if target_col in new_data.columns:
        features[target_col] = new_data[target_col].to_numpy()
        features = fp.flag_censored_target(features, target_col=target_col)
        features = features.drop(columns=[target_col])
    else:
        features["medv_censored"] = 0

    features = fp.drop_redundant_features(features)
    features = fp.add_rad_group_feature(features)

    feature_names = preprocessor.get_feature_names_out()
    transformed = preprocessor.transform(features)
    x_transformed = pd.DataFrame(
        transformed, columns=feature_names, index=features.index
    )

    # Misma columna de fuga de datos que excluye `train_pipeline.py`
    # antes de entrenar/predecir.
    x_model = tp.drop_leakage_columns(x_transformed)
    return x_model


def generate_predictions(model, x_model: pd.DataFrame) -> np.ndarray:
    """Genera las predicciones con el modelo ya entrenado.

    Reutiliza `train_pipeline.predict`, la misma función que usa el
    training pipeline para predecir sobre train/test.
    """
    return tp.predict(model, x_model)


def build_predictions_output(
    new_data: pd.DataFrame,
    predictions: np.ndarray,
    id_column: str = ID_COLUMN,
    target_col: str = TARGET_COL,
    prediction_col: str = PREDICTION_COL,
    input_columns: list[str] = REQUIRED_INPUT_COLUMNS,
) -> pd.DataFrame:
    """Arma el DataFrame de salida: identificador + atributos + predicción.

    Conserva un identificador para relacionar cada predicción con su
    registro original: la columna `ID` si el archivo de entrada la
    trae, o si no, el índice de fila. También conserva el valor real de
    `medv` cuando está disponible, para poder comparar contra la
    predicción (hindcast/evaluación).
    """
    output = pd.DataFrame(index=new_data.index)
    if id_column in new_data.columns:
        output[id_column] = new_data[id_column].to_numpy()
    else:
        output[id_column] = new_data.index

    for col in input_columns:
        output[col] = new_data[col].to_numpy()

    if target_col in new_data.columns:
        output[target_col] = new_data[target_col].to_numpy()

    output[prediction_col] = predictions
    return output


def save_predictions(
    predictions_df: pd.DataFrame, output_path: Path = PREDICTIONS_PATH
) -> Path:
    """Guarda las predicciones en un archivo `.csv` dentro de `data/predictions/`."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    return output_path


def plot_predictions(
    predictions_df: pd.DataFrame,
    plot_path: Path = PLOT_PATH,
    target_col: str = TARGET_COL,
    prediction_col: str = PREDICTION_COL,
) -> Path:
    """Genera una gráfica de evidencia sobre las predicciones generadas.

    Si los datos nuevos traían el valor real de `medv`, grafica
    real-vs-predicho (permite ver a simple vista qué tan cerca está el
    modelo). Si no, grafica la distribución de las predicciones, útil
    como evidencia mínima de que la inferencia generó resultados
    razonables.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    if target_col in predictions_df.columns:
        y_true = predictions_df[target_col]
        y_pred = predictions_df[prediction_col]
        ax.scatter(y_true, y_pred, alpha=0.6)
        limits = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        ax.plot(limits, limits, color="red", linestyle="--", label="Predicción ideal")
        ax.set_xlabel(f"{target_col} real")
        ax.set_ylabel(f"{target_col} predicho")
        ax.set_title("Inference Pipeline: real vs. predicho")
        ax.legend()
    else:
        ax.hist(predictions_df[prediction_col], bins=20)
        ax.set_xlabel(f"{target_col} predicho")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Inference Pipeline: distribución de predicciones")

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def write_inference_log(
    *,
    model_path: Path,
    preprocessor_path: Path,
    input_path: Path,
    n_records: int,
    features_used: list[str],
    output_path: Path,
    plot_path: Path | None,
    log_path: Path = LOG_PATH,
) -> Path:
    """Registra evidencia legible de qué ocurrió durante la inferencia."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).isoformat()
    lines = [
        f"INFERENCE PIPELINE — {timestamp}",
        f"Modelo cargado desde: {model_path}",
        f"Preprocesador cargado desde: {preprocessor_path}",
        f"Archivo de entrada: {input_path}",
        f"Registros procesados: {n_records}",
        f"Features utilizadas por el modelo ({len(features_used)}): {features_used}",
        f"Predicciones generadas: {n_records}",
        f"Predicciones guardadas en: {output_path}",
    ]
    if plot_path is not None:
        lines.append(f"Gráfica de evidencia guardada en: {plot_path}")

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines) + "\n")

    return log_path


def run_inference_pipeline(
    input_path: Path,
    model_path: Path = MODEL_PATH,
    preprocessor_path: Path = PREPROCESSOR_PATH,
    output_path: Path = PREDICTIONS_PATH,
    plot_path: Path = PLOT_PATH,
    log_path: Path = LOG_PATH,
) -> dict[str, object]:
    """Ejecuta el inference pipeline completo: datos nuevos -> predicciones guardadas.

    No entrena nada: carga el modelo y el preprocesador ya existentes,
    transforma los datos nuevos con las mismas funciones/objeto usados
    en entrenamiento, predice, guarda resultados y evidencia.
    """
    input_path = Path(input_path)

    model = load_model(model_path)
    preprocessor = load_preprocessor(preprocessor_path)
    new_data = load_new_data(input_path)

    x_model = transform_data(new_data, preprocessor)
    predictions = generate_predictions(model, x_model)

    predictions_df = build_predictions_output(new_data, predictions)
    saved_output_path = save_predictions(predictions_df, output_path)
    saved_plot_path = plot_predictions(predictions_df, plot_path)

    saved_log_path = write_inference_log(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        input_path=input_path,
        n_records=len(predictions_df),
        features_used=list(x_model.columns),
        output_path=saved_output_path,
        plot_path=saved_plot_path,
        log_path=log_path,
    )

    return {
        "predictions": predictions_df,
        "n_records": len(predictions_df),
        "features_used": list(x_model.columns),
        "output_path": saved_output_path,
        "plot_path": saved_plot_path,
        "log_path": saved_log_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos del pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Inference pipeline: genera predicciones de 'medv' para datos "
            "nuevos, usando el modelo y el preprocesador ya entrenados."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Ruta al archivo de datos nuevos ('.csv' o '.parquet').",
    )
    parser.add_argument(
        "--output",
        default=PREDICTIONS_PATH,
        type=Path,
        help=f"Ruta del archivo de predicciones a generar (default: {PREDICTIONS_PATH}).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    results = run_inference_pipeline(input_path=args.input, output_path=args.output)

    print(f"Modelo cargado desde: {MODEL_PATH}")
    print(f"Preprocesador cargado desde: {PREPROCESSOR_PATH}")
    print(f"Archivo de entrada: {args.input}")
    print(f"Registros procesados: {results['n_records']}")
    print(f"Features utilizadas: {results['features_used']}")
    print(f"Predicciones guardadas en: {results['output_path']}")
    print(f"Gráfica guardada en: {results['plot_path']}")
    print(f"Log guardado en: {results['log_path']}")


if __name__ == "__main__":
    main()
