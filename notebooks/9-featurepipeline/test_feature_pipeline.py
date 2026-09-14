"""Pruebas unitarias para pipelines/feature_pipeline.py.

Usa datos ficticios pequeños (no los datos reales del proyecto) para
verificar que las transformaciones del feature pipeline funcionan, que
generan las features esperadas y que el resultado tiene la estructura
esperada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Permite importar `pipelines.feature_pipeline` al ejecutar pytest desde
# la raíz del proyecto o desde cualquier otro directorio.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.feature_pipeline import feature_pipeline as fp


@pytest.fixture
def raw_boston_df() -> pd.DataFrame:
    """DataFrame ficticio con la misma forma que `boston_type_fixed.parquet`.

    Incluye: filas duplicadas, un nulo en `medv`, un valor censurado
    (medv = 50.0) y ambos grupos de `rad` (24 y distinto de 24), para
    poder probar cada paso del pipeline.
    """
    data = pd.DataFrame(
        {
            "crim": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
            "zn": [0.0, 0.0, 12.5, 12.5, 0.0, 0.0, 25.0, 25.0],
            "indus": [2.3, 2.3, 7.1, 7.1, 4.0, 4.0, 8.0, 8.0],
            "chas": [0, 0, 1, 1, 0, 0, 1, 1],
            "nox": [0.5, 0.5, 0.4, 0.4, 0.6, 0.6, 0.45, 0.45],
            "rm": [6.0, 6.0, 6.5, 6.5, 5.8, 5.8, 7.0, 7.0],
            "age": [65.0, 65.0, 40.0, 40.0, 80.0, 80.0, 20.0, 20.0],
            "dis": [4.0, 4.0, 5.0, 5.0, 3.5, 3.5, 6.0, 6.0],
            "rad": [1, 1, 24, 24, 3, 3, 24, 24],
            "tax": [300, 300, 666, 666, 280, 280, 666, 666],
            "ptratio": [15.0, 15.0, 18.0, 18.0, 16.0, 16.0, 19.0, 19.0],
            "black": [396.0, 396.0, 390.0, 390.0, 380.0, 380.0, 395.0, 395.0],
            "lstat": [5.0, 5.0, 9.0, 9.0, 12.0, 12.0, 3.0, 3.0],
            "medv": [24.0, 24.0, 21.0, np.nan, 50.0, 19.0, 33.0, 33.0],
        }
    )
    return data


def test_load_intermediate_data(tmp_path: Path, raw_boston_df: pd.DataFrame) -> None:
    """Verifica que se pueda leer correctamente un parquet de entrada."""
    input_path = tmp_path / "boston_type_fixed.parquet"
    raw_boston_df.to_parquet(input_path, engine="pyarrow")

    loaded_df = fp.load_intermediate_data(input_path)

    assert loaded_df.shape == raw_boston_df.shape
    assert list(loaded_df.columns) == list(raw_boston_df.columns)


def test_select_initial_columns_returns_expected_columns(
    raw_boston_df: pd.DataFrame,
) -> None:
    result = fp.select_initial_columns(raw_boston_df)

    assert list(result.columns) == fp.SELECTED_FEATURES
    assert result.shape[0] == raw_boston_df.shape[0]


def test_clean_data_drops_duplicates_and_missing_target(
    raw_boston_df: pd.DataFrame,
) -> None:
    result = fp.clean_data(raw_boston_df)

    # Se eliminan las 4 filas duplicadas (quedan 4 filas únicas) y la fila
    # con medv nulo ya no está entre las duplicadas, por lo que el
    # resultado no debe tener duplicados ni nulos en medv.
    assert result.duplicated().sum() == 0
    assert result["medv"].isna().sum() == 0
    assert result.shape[0] <= raw_boston_df.shape[0]


def test_flag_censored_target_marks_expected_rows(raw_boston_df: pd.DataFrame) -> None:
    clean_df = fp.clean_data(raw_boston_df)
    result = fp.flag_censored_target(clean_df)

    assert "medv_censored" in result.columns
    # Toda fila con medv == 50.0 debe quedar marcada con 1, el resto con 0
    assert (result.loc[result["medv"] == 50.0, "medv_censored"] == 1).all()
    assert (result.loc[result["medv"] != 50.0, "medv_censored"] == 0).all()
    assert set(result["medv_censored"].unique()).issubset({0, 1})


def test_drop_redundant_features_removes_tax_column(
    raw_boston_df: pd.DataFrame,
) -> None:
    assert "tax" in raw_boston_df.columns

    result = fp.drop_redundant_features(raw_boston_df)

    assert "tax" not in result.columns
    # No debe perder ninguna otra columna
    assert set(result.columns) == set(raw_boston_df.columns) - {"tax"}


def test_add_rad_group_feature_creates_expected_categories(
    raw_boston_df: pd.DataFrame,
) -> None:
    result = fp.add_rad_group_feature(raw_boston_df)

    assert "rad_group" in result.columns
    assert set(result["rad_group"].unique()).issubset({"alto", "bajo"})
    assert (result.loc[result["rad"] == 24, "rad_group"] == "alto").all()
    assert (result.loc[result["rad"] != 24, "rad_group"] == "bajo").all()


def test_split_train_test_shapes_and_no_leakage(raw_boston_df: pd.DataFrame) -> None:
    prepared_df = fp.add_rad_group_feature(
        fp.drop_redundant_features(
            fp.flag_censored_target(fp.clean_data(raw_boston_df))
        )
    )

    x_train, x_test, y_train, y_test = fp.split_train_test(
        prepared_df, test_size=0.5, random_state=42
    )

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) + len(x_test) == len(prepared_df)
    # El target no debe quedar como columna de las features
    assert fp.TARGET_COL not in x_train.columns
    assert fp.TARGET_COL not in x_test.columns
    # No debe haber filas compartidas entre train y test (sin fuga de datos)
    assert set(x_train.index).isdisjoint(set(x_test.index))


def test_build_preprocessor_returns_column_transformer() -> None:
    from sklearn.compose import ColumnTransformer

    preprocessor = fp.build_preprocessor()

    assert isinstance(preprocessor, ColumnTransformer)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert transformer_names == ["log", "numeric", "discretize", "binary", "nominal"]


def test_fit_transform_features_produces_numeric_matrix_without_nulls(
    raw_boston_df: pd.DataFrame,
) -> None:
    prepared_df = fp.add_rad_group_feature(
        fp.drop_redundant_features(
            fp.flag_censored_target(fp.clean_data(raw_boston_df))
        )
    )
    x_train, x_test, _, _ = fp.split_train_test(
        prepared_df, test_size=0.5, random_state=42
    )

    preprocessor = fp.build_preprocessor()
    x_train_transformed, x_test_transformed = fp.fit_transform_features(
        preprocessor, x_train, x_test
    )

    # Todas las columnas resultantes deben ser numéricas y sin nulos
    assert (
        x_train_transformed.select_dtypes(include="number").shape[1]
        == (x_train_transformed.shape[1])
    )
    assert x_train_transformed.isna().sum().sum() == 0
    assert x_test_transformed.isna().sum().sum() == 0
    # Mismas columnas en train y test (mismo preprocesador ya ajustado)
    assert list(x_train_transformed.columns) == list(x_test_transformed.columns)
    # Debe conservar el número de filas
    assert x_train_transformed.shape[0] == x_train.shape[0]
    assert x_test_transformed.shape[0] == x_test.shape[0]


def test_save_processed_features_writes_expected_files(
    tmp_path: Path, raw_boston_df: pd.DataFrame
) -> None:
    prepared_df = fp.add_rad_group_feature(
        fp.drop_redundant_features(
            fp.flag_censored_target(fp.clean_data(raw_boston_df))
        )
    )
    x_train, x_test, y_train, y_test = fp.split_train_test(
        prepared_df, test_size=0.5, random_state=42
    )
    preprocessor = fp.build_preprocessor()
    x_train_transformed, x_test_transformed = fp.fit_transform_features(
        preprocessor, x_train, x_test
    )

    output_dir = tmp_path / "03_primary"
    fp.save_processed_features(
        x_train_transformed, x_test_transformed, y_train, y_test, output_dir
    )

    expected_files = {
        "x_train_features.parquet",
        "x_test_features.parquet",
        "y_train.parquet",
        "y_test.parquet",
    }
    created_files = {path.name for path in output_dir.iterdir()}
    assert expected_files == created_files


def test_run_feature_pipeline_end_to_end(
    tmp_path: Path, raw_boston_df: pd.DataFrame
) -> None:
    """Prueba de integración: datos originales -> features procesadas guardadas."""
    input_path = tmp_path / "boston_type_fixed.parquet"
    raw_boston_df.to_parquet(input_path, engine="pyarrow")
    output_dir = tmp_path / "03_primary"

    results = fp.run_feature_pipeline(input_path=input_path, output_dir=output_dir)

    assert set(results.keys()) == {"x_train", "x_test", "y_train", "y_test"}
    assert results["x_train"].shape[0] > 0
    assert results["x_train"].isna().sum().sum() == 0
    assert (output_dir / "x_train_features.parquet").exists()
    assert (output_dir / "x_test_features.parquet").exists()
