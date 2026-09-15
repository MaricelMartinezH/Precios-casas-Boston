"""Pruebas de regresión para pipelines/inference_pipeline.py.

Usa datos sintéticos pequeños (no los datos reales del proyecto) para
ejecutar el flujo completo de inferencia — exactamente el que usa la app
de Streamlit (`demo_app.py`): `transform_data` -> `generate_predictions`.

Motivación: la firma de `train_pipeline.drop_leakage_columns` pasó de
`columns_to_drop` con valor por defecto a parámetro obligatorio y la
llamada de `transform_data` quedó desactualizada, lanzando `TypeError` en
runtime (rompía toda predicción de la app) sin que la suite lo detectara.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from pipelines import feature_pipeline_validado as fp
from pipelines import inference_pipeline as ip
from pipelines import train_pipeline as tp


def synthetic_raw_frame() -> pd.DataFrame:
    """DataFrame ficticio con las 14 columnas del esquema de entrada del proyecto."""
    return pd.DataFrame(
        {
            "crim": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            "zn": [0.0, 0.0, 12.5, 12.5, 0.0, 0.0, 25.0, 25.0],
            "indus": [2.3, 2.3, 7.1, 7.1, 4.0, 4.0, 8.0, 8.0],
            "chas": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "nox": [0.5, 0.5, 0.4, 0.4, 0.6, 0.6, 0.45, 0.45],
            "rm": [6.0, 6.0, 6.5, 6.5, 5.8, 5.8, 7.0, 7.0],
            "age": [65.0, 65.0, 40.0, 40.0, 80.0, 80.0, 20.0, 20.0],
            "dis": [4.0, 4.0, 5.0, 5.0, 3.5, 3.5, 6.0, 6.0],
            "rad": [1.0, 1.0, 24.0, 24.0, 3.0, 3.0, 24.0, 24.0],
            "tax": [300.0, 300.0, 666.0, 666.0, 280.0, 280.0, 666.0, 666.0],
            "ptratio": [15.0, 15.0, 18.0, 18.0, 16.0, 16.0, 19.0, 19.0],
            "black": [396.0, 396.0, 390.0, 390.0, 380.0, 380.0, 395.0, 395.0],
            "lstat": [5.0, 5.0, 9.0, 9.0, 12.0, 12.0, 3.0, 3.0],
            "medv": [24.0, 22.0, 21.0, 19.0, 50.0, 19.0, 33.0, 31.0],
        }
    )


def raw_inference_row(overrides: dict[str, float] | None = None) -> dict[str, float]:
    """Una vivienda nueva (sin target), igual que la fila que arma `demo_app.py`."""
    row = {
        "crim": 0.00632,
        "zn": 18.0,
        "indus": 2.31,
        "chas": 0.0,
        "nox": 0.538,
        "rm": 6.575,
        "age": 65.2,
        "dis": 4.09,
        "rad": 1.0,
        "tax": 296.0,
        "ptratio": 15.3,
        "black": 396.9,
        "lstat": 4.98,
    }
    if overrides:
        row.update(overrides)
    return row


@pytest.fixture
def fitted_preprocessor() -> ColumnTransformer:
    """Preprocesador ajustado sobre datos sintéticos con el esquema de producción."""
    raw = synthetic_raw_frame()
    prepared = fp.add_rad_group_feature(fp.drop_redundant_features(fp.flag_censored_target(raw)))
    preprocessor = fp.build_preprocessor()
    features = prepared.drop(columns=[fp.TARGET_COL])
    preprocessor.fit(features)
    return preprocessor


def test_transform_data_without_target_excludes_leakage_column(
    fitted_preprocessor: ColumnTransformer,
) -> None:
    """Inferencia sin `medv` (el caso de la demo) no debe lanzar error ni dejar fuga."""
    new_data = pd.DataFrame([raw_inference_row()])

    x_model = ip.transform_data(new_data, fitted_preprocessor)

    assert isinstance(x_model, pd.DataFrame)
    assert "binary__medv_censored" not in x_model.columns
    assert not x_model.isna().any().any()


def test_transform_data_returns_same_columns_as_training_features(
    fitted_preprocessor: ColumnTransformer,
) -> None:
    """Las features de inferencia deben coincidir con las que ve el modelo al entrenar."""
    transformed_names = list(fitted_preprocessor.get_feature_names_out())
    expected_columns = [col for col in transformed_names if col not in tp.LEAKAGE_COLUMNS]

    x_model = ip.transform_data(pd.DataFrame([raw_inference_row()]), fitted_preprocessor)

    assert list(x_model.columns) == expected_columns
    assert all(np.issubdtype(dtype, np.number) for dtype in x_model.dtypes)


def test_transform_data_with_target_produces_same_features_as_without(
    fitted_preprocessor: ColumnTransformer,
) -> None:
    """Si el archivo trae `medv`, debe usarse solo para el flag, no como feature."""
    with_target = pd.DataFrame([raw_inference_row({"medv": 25.0})])
    without_target = pd.DataFrame([raw_inference_row()])

    x_with = ip.transform_data(with_target, fitted_preprocessor)
    x_without = ip.transform_data(without_target, fitted_preprocessor)

    assert "medv" not in x_with.columns
    assert list(x_with.columns) == list(x_without.columns)
    pd.testing.assert_frame_equal(x_with, x_without)


def test_generate_predictions_end_to_end_matches_demo_flow() -> None:
    """Flujo completo de la demo: preprocesar -> transformar -> predecir.

    Entrena un modelo diminuto con los mismos pasos que `train_pipeline`
    (sobre datos sintéticos, sin depender de `data/`) y ejecuta el flujo
    de inferencia. Si `transform_data` lanza `TypeError` por una llamada
    desactualizada, este test falla — era el bug que rompía la app.
    """
    raw = synthetic_raw_frame()
    prepared = fp.add_rad_group_feature(fp.drop_redundant_features(fp.flag_censored_target(raw)))
    x_train, _, y_train, _ = fp.split_train_test(prepared, test_size=0.25, random_state=42)

    preprocessor = fp.build_preprocessor()
    x_train_transformed, _ = fp.fit_transform_features(preprocessor, x_train, x_train)
    # Mismo orden que train_pipeline.py: excluir la fuga ANTES de entrenar,
    # para que el modelo vea las mismas columnas que produce transform_data.
    x_train_model = tp.drop_leakage_columns(x_train_transformed, tp.LEAKAGE_COLUMNS)

    model = tp.build_model(n_estimators=10, max_depth=3)
    tp.train_model(model, x_train_model, y_train)

    new_data = pd.DataFrame([raw_inference_row()])
    x_model = ip.transform_data(new_data, preprocessor)
    predictions = ip.generate_predictions(model, x_model)

    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()
