"""Pruebas unitarias para pipelines/train_split_check_pipeline.py (Issue 4).

Usa DataFrames ficticios pequeños (no los datos reales del proyecto) para
probar la lógica de `validate_train_test_split()` de forma aislada: casos
válidos, fuga de datos, distribución problemática, tamaño/estructura
insuficiente y (como capacidad genérica extra de la función) orden
temporal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines import train_split_check_pipeline as tsc

MIN_EXPECTED_CHECKS = 5

# ---------------------------------------------------------------------------
# Fixtures: datos ficticios pequeños con la misma forma que produce el
# feature pipeline (columnas numéricas continuas + columnas binarias 0/1
# estilo One-Hot), sin solapamiento entre train y test.
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_split() -> dict[str, pd.DataFrame | pd.Series]:
    # Tamaños cercanos a los del proyecto real (~274 train / ~69 test, ver
    # docstring del módulo), lo bastante grandes para que dos muestras
    # i.i.d. de la misma distribución no disparen advertencias de drift
    # por simple variación muestral.
    rng = np.random.default_rng(42)
    n_train, n_test = 260, 70

    x_train = pd.DataFrame(
        {
            "numeric__lstat": rng.normal(loc=0.0, scale=1.0, size=n_train),
            "numeric__rm": rng.normal(loc=6.0, scale=0.5, size=n_train),
            "binary__chas": rng.integers(0, 2, size=n_train).astype(float),
        },
        index=range(n_train),
    )
    x_test = pd.DataFrame(
        {
            "numeric__lstat": rng.normal(loc=0.0, scale=1.0, size=n_test),
            "numeric__rm": rng.normal(loc=6.0, scale=0.5, size=n_test),
            "binary__chas": rng.integers(0, 2, size=n_test).astype(float),
        },
        index=range(n_train, n_train + n_test),
    )

    y_train = pd.Series(20.0 + 3.0 * x_train["numeric__lstat"], name="medv", index=x_train.index)
    y_test = pd.Series(20.0 + 3.0 * x_test["numeric__lstat"], name="medv", index=x_test.index)

    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}



DEFAULT_CONFIG = tsc.TrainTestSplitConfig(target_col="medv")
SMALL_DATA_CONFIG = tsc.TrainTestSplitConfig(
    target_col="medv",
    min_train_size=50,
    min_test_size=15,
)
TEMPORAL_CONFIG = tsc.TrainTestSplitConfig(
    target_col="medv",
    time_column="fecha",
)

# ---------------------------------------------------------------------------
# CASO 1 — Split válido
# ---------------------------------------------------------------------------


def test_valid_split_passes_without_raising(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    result = tsc.validate_train_test_split(
        valid_split["x_train"],
        valid_split["x_test"],
        valid_split["y_train"],
        valid_split["y_test"],
        DEFAULT_CONFIG,
    )

    assert result.passed
    assert result.failed_checks() == []
    # Se registran múltiples checks, no solo un booleano "passed".
    assert len(result.checks) > MIN_EXPECTED_CHECKS
    assert any(c.name == "overlap_indices" and c.status == "passed" for c in result.checks)


# ---------------------------------------------------------------------------
# CASO 2 — Data leakage (registro compartido entre train y test)
# ---------------------------------------------------------------------------


def test_shared_index_raises_leakage_error(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train = valid_split["x_train"]
    x_test = valid_split["x_test"].copy()
    # Fuerza un índice de test a coincidir con uno de train.
    x_test.index = [x_train.index[0], *list(x_test.index[1:])]

    with pytest.raises(tsc.TrainTestSplitValidationError, match="overlap_indices"):
        tsc.validate_train_test_split(
            x_train,
            x_test,
            valid_split["y_train"],
            valid_split["y_test"],
            DEFAULT_CONFIG,
        )


def test_duplicate_row_content_raises_leakage_error(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train = valid_split["x_train"].copy()
    x_test = valid_split["x_test"].copy()
    y_train = valid_split["y_train"].copy()
    y_test = valid_split["y_test"].copy()

    # Copia una fila de train hacia test con un índice distinto: no hay
    # solapamiento de índice, pero el contenido es idéntico.
    fila_duplicada = x_train.iloc[[0]].copy()
    fila_duplicada.index = [max(x_test.index) + 1]
    x_test = pd.concat([x_test, fila_duplicada])
    y_test = pd.concat(
        [y_test, pd.Series([y_train.iloc[0]], index=fila_duplicada.index, name="medv")]
    )

    with pytest.raises(tsc.TrainTestSplitValidationError, match="duplicate_rows"):
        tsc.validate_train_test_split(x_train, x_test, y_train, y_test, DEFAULT_CONFIG)


def test_target_present_in_features_raises_leakage_error(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train = valid_split["x_train"].copy()
    x_test = valid_split["x_test"].copy()
    x_train["medv"] = valid_split["y_train"]
    x_test["medv"] = valid_split["y_test"]

    with pytest.raises(tsc.TrainTestSplitValidationError, match="target_not_in_features"):
        tsc.validate_train_test_split(
            x_train,
            x_test,
            valid_split["y_train"],
            valid_split["y_test"],
            DEFAULT_CONFIG,
        )


# ---------------------------------------------------------------------------
# CASO 3 — Distribución problemática (advertencia, no error)
# ---------------------------------------------------------------------------


def test_distribution_mismatch_produces_warning_but_still_passes(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train = valid_split["x_train"].copy()
    x_test = valid_split["x_test"].copy()
    # Desplaza fuertemente la distribución de una columna en test.
    x_test["numeric__lstat"] = x_test["numeric__lstat"] + 15.0

    with pytest.warns(UserWarning, match="distribution_numeric__lstat"):
        result = tsc.validate_train_test_split(
            x_train,
            x_test,
            valid_split["y_train"],
            valid_split["y_test"],
            DEFAULT_CONFIG,
        )

    assert result.passed  # una diferencia de distribución no es un error crítico
    assert result.has_warnings
    warning_names = {c.name for c in result.warning_checks()}
    assert "distribution_numeric__lstat" in warning_names


def test_new_category_in_test_produces_warning(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train = valid_split["x_train"].copy()
    x_test = valid_split["x_test"].copy()
    # `binary__chas` solo tiene 0/1 en train; se agrega una categoría
    # nueva (2) únicamente en test.
    x_test.loc[x_test.index[0], "binary__chas"] = 2.0

    with pytest.warns(UserWarning, match="new_category_binary__chas"):
        result = tsc.validate_train_test_split(
            x_train,
            x_test,
            valid_split["y_train"],
            valid_split["y_test"],
            DEFAULT_CONFIG,
        )

    assert result.passed
    assert any(
        c.name == "new_category_binary__chas" and c.status == "warning" for c in result.checks
    )


# ---------------------------------------------------------------------------
# CASO 4 — Problema de tamaño/estructura
# ---------------------------------------------------------------------------


def test_train_set_too_small_raises_error(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train_pequeno = valid_split["x_train"].iloc[:5]
    y_train_pequeno = valid_split["y_train"].iloc[:5]

    with pytest.raises(tsc.TrainTestSplitValidationError, match="min_train_size"):
        tsc.validate_train_test_split(
            x_train_pequeno,
            valid_split["x_test"],
            y_train_pequeno,
            valid_split["y_test"],
            SMALL_DATA_CONFIG,
        )


def test_mismatched_target_length_raises_error(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    y_train_incompleto = valid_split["y_train"].iloc[:-1]

    with pytest.raises(tsc.TrainTestSplitValidationError, match="target_length_train"):
        tsc.validate_train_test_split(
            valid_split["x_train"],
            valid_split["x_test"],
            y_train_incompleto,
            valid_split["y_test"],
            DEFAULT_CONFIG,
        )


# ---------------------------------------------------------------------------
# CASO 5 (extra) — Orden temporal
#
# El proyecto Precios-casas-Boston NO es un problema temporal (no existe
# ninguna columna de fecha en el dataset), por lo que `run_split_check_pipeline`
# no activa este check. Se prueba aquí de forma aislada porque
# `validate_train_test_split()` lo ofrece como capacidad genérica
# (parámetro opcional `time_column`), para dejar constancia de que
# funciona correctamente si en el futuro se usa con un problema temporal.
# ---------------------------------------------------------------------------


def test_future_information_in_train_raises_temporal_error(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train = valid_split["x_train"].copy()
    x_test = valid_split["x_test"].copy()
    x_train["fecha"] = pd.date_range("2024-01-01", periods=len(x_train), freq="D")
    # Fuerza que la última fecha de train sea posterior a la primera de test.
    x_test["fecha"] = pd.date_range("2024-01-01", periods=len(x_test), freq="D")

    with pytest.raises(tsc.TrainTestSplitValidationError, match="temporal_order"):
        tsc.validate_train_test_split(
            x_train,
            x_test,
            valid_split["y_train"],
            valid_split["y_test"],
            TEMPORAL_CONFIG,
        )


def test_temporal_order_respected_passes(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train = valid_split["x_train"].copy()
    x_test = valid_split["x_test"].copy()
    x_train["fecha"] = pd.date_range("2020-01-01", periods=len(x_train), freq="D")
    x_test["fecha"] = pd.date_range("2030-01-01", periods=len(x_test), freq="D")

    result = tsc.validate_train_test_split(
        x_train,
        x_test,
        valid_split["y_train"],
        valid_split["y_test"],
        TEMPORAL_CONFIG,
    )

    assert result.passed
    assert any(c.name == "temporal_order" and c.status == "passed" for c in result.checks)


# ---------------------------------------------------------------------------
# TrainTestSplitValidationResult: estructura del resultado
# ---------------------------------------------------------------------------


def test_result_to_dict_contains_all_checks(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    result = tsc.validate_train_test_split(
        valid_split["x_train"],
        valid_split["x_test"],
        valid_split["y_train"],
        valid_split["y_test"],
        DEFAULT_CONFIG,
    )

    payload = result.to_dict()

    assert payload["passed"] is True
    assert len(payload["checks"]) == len(result.checks)
    assert {"name", "status", "message"} <= payload["checks"][0].keys()


# ---------------------------------------------------------------------------
# run_split_check_pipeline: prueba de integración con archivos parquet
# (usa tmp_path, no toca los datos reales del proyecto).
# ---------------------------------------------------------------------------


def test_run_split_check_pipeline_end_to_end_valid_case(
    tmp_path: Path, valid_split: dict[str, pd.DataFrame | pd.Series]
) -> None:
    x_train_path = tmp_path / "x_train_features.parquet"
    x_test_path = tmp_path / "x_test_features.parquet"
    y_train_path = tmp_path / "y_train.parquet"
    y_test_path = tmp_path / "y_test.parquet"
    results_path = tmp_path / "train_split_check_results.json"

    valid_split["x_train"].to_parquet(x_train_path, engine="pyarrow")
    valid_split["x_test"].to_parquet(x_test_path, engine="pyarrow")
    valid_split["y_train"].to_frame().to_parquet(y_train_path, engine="pyarrow")
    valid_split["y_test"].to_frame().to_parquet(y_test_path, engine="pyarrow")

    result = tsc.run_split_check_pipeline(
        x_train_path=x_train_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
        results_path=results_path,
        config=DEFAULT_CONFIG,
    )

    assert result.passed
    assert results_path.exists()
    with results_path.open(encoding="utf-8") as results_file:
        saved_payload = json.load(results_file)
    assert saved_payload["passed"] is True


def test_run_split_check_pipeline_writes_results_even_when_it_raises(
    tmp_path: Path, valid_split: dict[str, pd.DataFrame | pd.Series]
) -> None:
    x_train_path = tmp_path / "x_train_features.parquet"
    x_test_path = tmp_path / "x_test_features.parquet"
    y_train_path = tmp_path / "y_train.parquet"
    y_test_path = tmp_path / "y_test.parquet"
    results_path = tmp_path / "train_split_check_results.json"

    x_train = valid_split["x_train"]
    x_test = valid_split["x_test"].copy()
    x_test.index = [x_train.index[0], *list(x_test.index[1:])]  # fuerza fuga de datos

    x_train.to_parquet(x_train_path, engine="pyarrow")
    x_test.to_parquet(x_test_path, engine="pyarrow")
    valid_split["y_train"].to_frame().to_parquet(y_train_path, engine="pyarrow")
    valid_split["y_test"].to_frame().to_parquet(y_test_path, engine="pyarrow")

    with pytest.raises(tsc.TrainTestSplitValidationError):
        tsc.run_split_check_pipeline(
            x_train_path=x_train_path,
            x_test_path=x_test_path,
            y_train_path=y_train_path,
            y_test_path=y_test_path,
            results_path=results_path,
            config=DEFAULT_CONFIG,
        )

    assert results_path.exists()
    with results_path.open(encoding="utf-8") as results_file:
        saved_payload = json.load(results_file)
    assert saved_payload["passed"] is False
    assert any(c["name"] == "overlap_indices" for c in saved_payload["checks"])
