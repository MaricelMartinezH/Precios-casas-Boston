"""Training pipeline para el proyecto Precios-casas-Boston.

Consume las features YA procesadas y persistidas por
`pipelines/feature_pipeline_validado.py` (Issues 1 y 2) y entrena el mismo
modelo seleccionado en `08.Seleccion_modelo-mkmh-2026-08-28.ipynb`
(Issue 6): un `GradientBoostingRegressor` con los hiperparámetros ya
optimizados en ese notebook (`gradient_boosting_tuned`). Esta Issue no
vuelve a leer los datos originales ni a decidir el modelo o sus
hiperparámetros: solo construye el pipeline de entrenamiento a partir de
lo ya definido en el POC.

Nota importante heredada del notebook 08: la columna `binary__medv_censored`
que persiste `feature_pipeline_validado.py` se calcula a partir de la
variable objetivo (`medv == 50.0`), por lo que usarla como predictora sería
fuga de la variable objetivo (target leakage). El propio notebook 08 la
excluye de la matriz de atributos antes de entrenar cualquier modelo de
machine learning real, y este pipeline hace exactamente lo mismo.

Flujo:
    leer features procesadas -> excluir columna con fuga de datos
    -> entrenar -> predecir -> calcular métricas -> guardar modelo
    -> guardar métricas

Uso:
    python pipelines/train_pipeline.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# Raíz del proyecto: este archivo vive en <root>/pipelines/train_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Entrada: salida del feature pipeline (Issues 1 y 2), NO los datos originales.
FEATURES_DIR = DATA_DIR / "03_primary"
X_TRAIN_PATH = FEATURES_DIR / "x_train_features.parquet"
X_TEST_PATH = FEATURES_DIR / "x_test_features.parquet"
Y_TRAIN_PATH = FEATURES_DIR / "y_train.parquet"
Y_TEST_PATH = FEATURES_DIR / "y_test.parquet"

# Salida: modelo entrenado y resultados de evaluación.
MODELS_DIR = DATA_DIR / "06_models"
MODEL_OUTPUT_DIR = DATA_DIR / "07_model_output"
MODEL_PATH = MODELS_DIR / "train_pipeline_model.joblib"
METRICS_PATH = MODEL_OUTPUT_DIR / "train_pipeline_metrics.json"


@dataclass(frozen=True)
class PipelinePaths:
    """Rutas de entrada y salida del training pipeline."""

    x_train: Path = X_TRAIN_PATH
    x_test: Path = X_TEST_PATH
    y_train: Path = Y_TRAIN_PATH
    y_test: Path = Y_TEST_PATH
    model: Path = MODEL_PATH
    metrics: Path = METRICS_PATH


DEFAULT_PIPELINE_PATHS = PipelinePaths()

TARGET_COL = "medv"

# `binary__medv_censored` se descarta antes de entrenar: se calcula a partir
# de `medv` (target leakage), tal como se documenta y corrige en el
# notebook `08.Seleccion_modelo` (Issue 6).
LEAKAGE_COLUMNS = ["binary__medv_censored"]

# Modelo y métrica principal seleccionados en el notebook 08
# (`gradient_boosting_tuned`), incluyendo los hiperparámetros ya
# optimizados allí con `RandomizedSearchCV` + validación cruzada. Esta
# Issue reutiliza ese resultado; no vuelve a buscar hiperparámetros.
MODEL_NAME = "gradient_boosting_tuned"
RANDOM_STATE = 42
MODEL_HYPERPARAMETERS: dict[str, object] = {
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_samples_leaf": 1,
    "n_estimators": 400,
    "subsample": 0.7,
    "random_state": RANDOM_STATE,
}

# Métricas de regresión ya usadas en el POC (notebooks 07 y 08), con RMSE
# como métrica principal de decisión.
MAIN_METRIC = "RMSE"


def load_processed_features(
    x_train_path: Path = X_TRAIN_PATH,
    x_test_path: Path = X_TEST_PATH,
    y_train_path: Path = Y_TRAIN_PATH,
    y_test_path: Path = Y_TEST_PATH,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Lee las features y targets ya procesados por el feature pipeline.

    No vuelve a leer los datos originales del proyecto: consume
    directamente los archivos parquet generados por
    `feature_pipeline_validado.py` (Issues 1 y 2).
    """
    for path in (x_train_path, x_test_path, y_train_path, y_test_path):
        if not path.exists():
            msg = (
                f"No se encontró '{path}'. Ejecuta primero "
                "'python pipelines/feature_pipeline_validado.py' para generar "
                "las features procesadas antes de correr el training pipeline."
            )
            raise FileNotFoundError(msg)

    x_train = pd.read_parquet(x_train_path, engine="pyarrow")
    x_test = pd.read_parquet(x_test_path, engine="pyarrow")
    y_train = pd.read_parquet(y_train_path, engine="pyarrow")[target_col]
    y_test = pd.read_parquet(y_test_path, engine="pyarrow")[target_col]

    return x_train, x_test, y_train, y_test


def drop_leakage_columns(
    df: pd.DataFrame, columns_to_drop: list[str] = LEAKAGE_COLUMNS
) -> pd.DataFrame:
    """Elimina columnas que constituyen fuga de la variable objetivo.

    Ver nota del módulo sobre `binary__medv_censored` (notebook 08).
    """
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=existing_columns)


def build_model(**hyperparameters: object) -> GradientBoostingRegressor:
    """Construye el modelo seleccionado en el POC (notebook 08).

    Por defecto usa los hiperparámetros ya optimizados allí
    (`gradient_boosting_tuned`). Se pueden sobreescribir puntualmente
    (por ejemplo, en pruebas unitarias) vía kwargs.
    """
    params = {**MODEL_HYPERPARAMETERS, **hyperparameters}
    return GradientBoostingRegressor(**params)


def train_model(
    model: GradientBoostingRegressor, x_train: pd.DataFrame, y_train: pd.Series
) -> GradientBoostingRegressor:
    """Entrena el modelo sobre las features de entrenamiento ya procesadas."""
    model.fit(x_train, y_train)
    return model


def predict(model: GradientBoostingRegressor, x: pd.DataFrame) -> np.ndarray:
    """Genera predicciones con el modelo ya entrenado."""
    return model.predict(x)


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula las mismas cuatro métricas de regresión usadas en el POC.

    MAE, RMSE, R2 y MAPE (notebooks 07 y 08), con RMSE como métrica
    principal de decisión (penaliza más los errores grandes, en las
    mismas unidades que `medv`).
    """
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def save_model(model: GradientBoostingRegressor, model_path: Path = MODEL_PATH) -> None:
    """Guarda el modelo entrenado para poder cargarlo después con `joblib.load`."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def save_metrics(metrics: dict, metrics_path: Path = METRICS_PATH) -> None:
    """Guarda los resultados de evaluación en un archivo JSON."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)


def run_train_pipeline(
    paths: PipelinePaths = DEFAULT_PIPELINE_PATHS,
) -> dict[str, object]:
    """Ejecuta el training pipeline completo: features procesadas -> modelo y métricas.

    Retorna un diccionario con el modelo entrenado y las métricas, útil
    tanto para pruebas unitarias como para inspección interactiva.
    """
    x_train, x_test, y_train, y_test = load_processed_features(
        paths.x_train,
        paths.x_test,
        paths.y_train,
        paths.y_test,
    )

    x_train_model = drop_leakage_columns(x_train)
    x_test_model = drop_leakage_columns(x_test)

    model = build_model()
    model = train_model(model, x_train_model, y_train)

    y_train_pred = predict(model, x_train_model)
    y_test_pred = predict(model, x_test_model)

    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)

    save_model(model, paths.model)

    metrics_payload = {
        "model": MODEL_NAME,
        "estimator": type(model).__name__,
        "hyperparameters": MODEL_HYPERPARAMETERS,
        "main_metric": MAIN_METRIC,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "n_train": len(x_train_model),
        "n_test": len(x_test_model),
        "n_features_in": int(x_train_model.shape[1]),
        "train": train_metrics,
        "test": test_metrics,
    }
    save_metrics(metrics_payload, paths.metrics)

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "metrics_payload": metrics_payload,
    }


def main() -> None:
    results = run_train_pipeline()
    print(f"Features leídas desde: {FEATURES_DIR}")
    print(f"Modelo entrenado: {MODEL_NAME} ({type(results['model']).__name__})")
    print(f"Métricas en train: {results['train_metrics']}")
    print(f"Métricas en test : {results['test_metrics']}")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()
