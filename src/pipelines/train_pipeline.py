"""Training pipeline del modelo seleccionado para Boston Housing.

Consume las features ya procesadas por
`pipelines.feature_pipeline_validado`, entrena el
`GradientBoostingRegressor` seleccionado en el notebook 08 con sus
hiperparámetros optimizados y excluye `binary__medv_censored` para evitar
fuga de la variable objetivo.

Flujo:
1. Leer features procesadas.
2. Excluir columnas con fuga.
3. Construir y entrenar el modelo.
4. Generar predicciones.
5. Calcular métricas.
6. Guardar modelo y métricas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "03_primary"

X_TRAIN_PATH = FEATURES_DIR / "x_train_features.parquet"
X_TEST_PATH = FEATURES_DIR / "x_test_features.parquet"
Y_TRAIN_PATH = FEATURES_DIR / "y_train.parquet"
Y_TEST_PATH = FEATURES_DIR / "y_test.parquet"

MODELS_DIR = DATA_DIR / "06_models"
MODEL_OUTPUT_DIR = DATA_DIR / "07_model_output"

MODEL_PATH = MODELS_DIR / "train_pipeline_model.joblib"
METRICS_PATH = MODEL_OUTPUT_DIR / "train_pipeline_metrics.json"


@dataclass(frozen=True)
class PipelinePaths:
    """Rutas utilizadas por el training pipeline."""

    X_TRAIN_PATH: Path = X_TRAIN_PATH
    x_train: Path = X_TRAIN_PATH
    X_TEST_PATH: Path = X_TEST_PATH
    x_test: Path = X_TEST_PATH
    Y_TRAIN_PATH: Path = Y_TRAIN_PATH
    y_train: Path = Y_TRAIN_PATH
    Y_TEST_PATH: Path = Y_TEST_PATH
    y_test: Path = Y_TEST_PATH
    MODEL_PATH: Path = MODEL_PATH
    model: Path = MODEL_PATH
    METRICS_PATH: Path = METRICS_PATH
    metrics: Path = METRICS_PATH


DEFAULT_PIPELINE_PATHS = PipelinePaths()

TARGET_COL = "medv"
LEAKAGE_COLUMNS = ["binary__medv_censored"]

MODEL_NAME = "gradient_boosting_tuned"
RANDOM_STATE = 42

MODEL_HYPERPARAMETERS = {
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_samples_leaf": 1,
    "n_estimators": 400,
    "subsample": 0.7,
    "random_state": RANDOM_STATE,
}

MAIN_METRIC = "RMSE"


def load_processed_features(
    x_train_path: Path,
    x_test_path: Path,
    y_train_path: Path,
    y_test_path: Path,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga las features procesadas y la variable objetivo."""

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
    df: pd.DataFrame,
    columns_to_drop: list[str],
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
    model: GradientBoostingRegressor,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> GradientBoostingRegressor:
    """Entrena el modelo."""

    model.fit(x_train, y_train)
    return model


def predict(
    model: GradientBoostingRegressor,
    x: pd.DataFrame,
) -> np.ndarray:
    """Genera predicciones."""

    return model.predict(x)


def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calcula las métricas de regresión."""

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def save_model(
    model: GradientBoostingRegressor,
    model_path: Path,
) -> None:
    """Guarda el modelo entrenado."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


def save_metrics(
    metrics: dict[str, object],
    metrics_path: Path,
) -> None:
    """Guarda las métricas en formato JSON."""

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)


class TrainPipelineResult(TypedDict):
    """Resultado tipado del training pipeline."""

    model: GradientBoostingRegressor
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    metrics_payload: dict[str, object]


def run_train_pipeline(
    paths: PipelinePaths,
) -> TrainPipelineResult:
    """Ejecuta el training pipeline completo: features procesadas -> modelo y métricas.

    Retorna un diccionario con el modelo entrenado y las métricas, útil
    tanto para pruebas unitarias como para inspección interactiva.
    """

    x_train, x_test, y_train, y_test = load_processed_features(
        paths.x_train,
        paths.x_test,
        paths.y_train,
        paths.y_test,
        TARGET_COL,
    )

    x_train_model = drop_leakage_columns(x_train, LEAKAGE_COLUMNS)
    x_test_model = drop_leakage_columns(x_test, LEAKAGE_COLUMNS)

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
    """Ejecuta el training pipeline desde línea de comandos."""

    result = run_train_pipeline(DEFAULT_PIPELINE_PATHS)

    print(f"Modelo: {result['metrics_payload']['model']}")
    print(f"RMSE train: {result['train_metrics']['RMSE']:.4f}")
    print(f"RMSE test: {result['test_metrics']['RMSE']:.4f}")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()
