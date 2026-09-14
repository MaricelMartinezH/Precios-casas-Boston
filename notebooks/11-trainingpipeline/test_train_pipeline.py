"""Pruebas unitarias para pipelines/train_pipeline.py.

Usa datos ficticios pequeños (no los datos reales del proyecto) para
verificar, de forma aislada, que el training pipeline puede: leer las
features ya procesadas, entrenar el modelo, generar predicciones y
métricas válidas, y guardar tanto el modelo como los resultados de
evaluación.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

# Permite importar `pipelines.train_pipeline` al ejecutar pytest desde la
# raíz del proyecto o desde cualquier otro directorio.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_pipeline as tp


@pytest.fixture
def small_features_df() -> pd.DataFrame:
    """Features ficticias con la misma forma que produce el feature pipeline.

    Incluye la columna `binary__medv_censored` (fuga de datos) para poder
    comprobar que el training pipeline la excluye antes de entrenar.
    """
    rng = np.random.default_rng(42)
    n_rows = 30
    df = pd.DataFrame(
        {
            "log__crim": rng.normal(size=n_rows),
            "numeric__zn": rng.normal(size=n_rows),
            "numeric__lstat": rng.normal(size=n_rows),
            "binary__chas": rng.integers(0, 2, size=n_rows).astype(float),
            "binary__medv_censored": rng.integers(0, 2, size=n_rows).astype(float),
            "nominal__rad_group_alto": rng.integers(0, 2, size=n_rows).astype(float),
        }
    )
    return df


@pytest.fixture
def small_target_series(small_features_df: pd.DataFrame) -> pd.Series:
    """Target ficticio, con relación lineal simple a una de las features."""
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=0.1, size=len(small_features_df))
    values = 20.0 + 3.0 * small_features_df["numeric__lstat"] + noise
    return pd.Series(values, name="medv", index=small_features_df.index)


# ---------------------------------------------------------------------------
# 1. Lectura de datos
# ---------------------------------------------------------------------------


def test_load_processed_features_reads_expected_shapes(
    tmp_path: Path, small_features_df: pd.DataFrame, small_target_series: pd.Series
) -> None:
    x_train_path = tmp_path / "x_train_features.parquet"
    x_test_path = tmp_path / "x_test_features.parquet"
    y_train_path = tmp_path / "y_train.parquet"
    y_test_path = tmp_path / "y_test.parquet"

    small_features_df.iloc[:20].to_parquet(x_train_path, engine="pyarrow")
    small_features_df.iloc[20:].to_parquet(x_test_path, engine="pyarrow")
    small_target_series.iloc[:20].to_frame().to_parquet(y_train_path, engine="pyarrow")
    small_target_series.iloc[20:].to_frame().to_parquet(y_test_path, engine="pyarrow")

    x_train, x_test, y_train, y_test = tp.load_processed_features(
        x_train_path, x_test_path, y_train_path, y_test_path
    )

    assert x_train.shape == (20, small_features_df.shape[1])
    assert x_test.shape == (10, small_features_df.shape[1])
    assert len(y_train) == 20
    assert len(y_test) == 10
    assert y_train.name == "medv"


def test_load_processed_features_raises_clear_error_when_file_missing(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "no_existe.parquet"

    with pytest.raises(FileNotFoundError, match="feature_pipeline_validado"):
        tp.load_processed_features(
            missing_path, missing_path, missing_path, missing_path
        )


def test_drop_leakage_columns_removes_medv_censored(
    small_features_df: pd.DataFrame,
) -> None:
    assert "binary__medv_censored" in small_features_df.columns

    result = tp.drop_leakage_columns(small_features_df)

    assert "binary__medv_censored" not in result.columns
    # El resto de columnas se conserva intacto
    assert set(result.columns) == set(small_features_df.columns) - {
        "binary__medv_censored"
    }
    assert len(result) == len(small_features_df)


# ---------------------------------------------------------------------------
# 2. Entrenamiento
# ---------------------------------------------------------------------------


def test_build_model_returns_gradient_boosting_with_expected_hyperparameters() -> None:
    model = tp.build_model()

    assert isinstance(model, GradientBoostingRegressor)
    assert model.learning_rate == tp.MODEL_HYPERPARAMETERS["learning_rate"]
    assert model.max_depth == tp.MODEL_HYPERPARAMETERS["max_depth"]
    assert model.n_estimators == tp.MODEL_HYPERPARAMETERS["n_estimators"]
    assert model.subsample == tp.MODEL_HYPERPARAMETERS["subsample"]
    assert model.random_state == tp.MODEL_HYPERPARAMETERS["random_state"]


def test_build_model_allows_overriding_hyperparameters_for_tests() -> None:
    """Las pruebas pueden pedir un modelo más pequeño/rápido sin tocar el default."""
    model = tp.build_model(n_estimators=5)

    assert model.n_estimators == 5
    # El resto de hiperparámetros del POC no cambia
    assert model.learning_rate == tp.MODEL_HYPERPARAMETERS["learning_rate"]


def test_train_model_fits_successfully(
    small_features_df: pd.DataFrame, small_target_series: pd.Series
) -> None:
    x_train = tp.drop_leakage_columns(small_features_df)
    model = tp.build_model(n_estimators=5)

    trained_model = tp.train_model(model, x_train, small_target_series)

    # Un modelo ya ajustado expone `estimators_` y puede predecir
    assert hasattr(trained_model, "estimators_")
    predictions = trained_model.predict(x_train)
    assert len(predictions) == len(x_train)


# ---------------------------------------------------------------------------
# 3. Generación de métricas
# ---------------------------------------------------------------------------


def test_predict_generates_expected_number_of_predictions(
    small_features_df: pd.DataFrame, small_target_series: pd.Series
) -> None:
    x_train = tp.drop_leakage_columns(small_features_df)
    model = tp.train_model(tp.build_model(n_estimators=5), x_train, small_target_series)

    predictions = tp.predict(model, x_train)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(x_train)


def test_compute_regression_metrics_returns_valid_values() -> None:
    y_true = pd.Series([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 19.0, 33.0, 37.0])

    metrics = tp.compute_regression_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"MAE", "RMSE", "R2", "MAPE"}
    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0
    assert metrics["MAPE"] >= 0
    assert metrics["R2"] <= 1
    # Para predicciones cercanas al valor real, el error debe ser pequeño
    assert metrics["MAE"] < 5


def test_compute_regression_metrics_perfect_predictions_are_ideal() -> None:
    y_true = pd.Series([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])

    metrics = tp.compute_regression_metrics(y_true, y_pred)

    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["RMSE"] == pytest.approx(0.0)
    assert metrics["MAPE"] == pytest.approx(0.0)
    assert metrics["R2"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Almacenamiento del modelo
# ---------------------------------------------------------------------------


def test_save_model_creates_file_and_can_be_reloaded(
    tmp_path: Path, small_features_df: pd.DataFrame, small_target_series: pd.Series
) -> None:
    x_train = tp.drop_leakage_columns(small_features_df)
    model = tp.train_model(tp.build_model(n_estimators=5), x_train, small_target_series)
    model_path = tmp_path / "modelo" / "train_pipeline_model.joblib"

    tp.save_model(model, model_path)

    assert model_path.exists()

    loaded_model = joblib.load(model_path)
    original_predictions = model.predict(x_train)
    loaded_predictions = loaded_model.predict(x_train)
    np.testing.assert_allclose(original_predictions, loaded_predictions)


# ---------------------------------------------------------------------------
# 5. Almacenamiento de resultados (métricas)
# ---------------------------------------------------------------------------


def test_save_metrics_writes_expected_json(tmp_path: Path) -> None:
    metrics_path = tmp_path / "resultados" / "train_pipeline_metrics.json"
    payload = {"model": "gradient_boosting_tuned", "test": {"RMSE": 2.65, "R2": 0.84}}

    tp.save_metrics(payload, metrics_path)

    assert metrics_path.exists()
    with metrics_path.open(encoding="utf-8") as metrics_file:
        loaded_payload = json.load(metrics_file)
    assert loaded_payload == payload


# ---------------------------------------------------------------------------
# Prueba de integración: pipeline completo
# ---------------------------------------------------------------------------


def test_run_train_pipeline_end_to_end(
    tmp_path: Path, small_features_df: pd.DataFrame, small_target_series: pd.Series
) -> None:
    """features procesadas ficticias -> modelo entrenado + métricas guardadas."""
    x_train_path = tmp_path / "x_train_features.parquet"
    x_test_path = tmp_path / "x_test_features.parquet"
    y_train_path = tmp_path / "y_train.parquet"
    y_test_path = tmp_path / "y_test.parquet"
    model_path = tmp_path / "models" / "train_pipeline_model.joblib"
    metrics_path = tmp_path / "model_output" / "train_pipeline_metrics.json"

    small_features_df.iloc[:20].to_parquet(x_train_path, engine="pyarrow")
    small_features_df.iloc[20:].to_parquet(x_test_path, engine="pyarrow")
    small_target_series.iloc[:20].to_frame().to_parquet(y_train_path, engine="pyarrow")
    small_target_series.iloc[20:].to_frame().to_parquet(y_test_path, engine="pyarrow")

    results = tp.run_train_pipeline(
        x_train_path=x_train_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    assert set(results.keys()) == {
        "model",
        "train_metrics",
        "test_metrics",
        "metrics_payload",
    }
    assert model_path.exists()
    assert metrics_path.exists()

    # El modelo no debe haber recibido la columna con fuga de datos
    assert "binary__medv_censored" not in results["model"].feature_names_in_

    with metrics_path.open(encoding="utf-8") as metrics_file:
        saved_metrics = json.load(metrics_file)
    assert saved_metrics["model"] == tp.MODEL_NAME
    assert "RMSE" in saved_metrics["test"]
