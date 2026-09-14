"""Pruebas unitarias para pipelines/train_split_check_pipeline.py (Issue 4).

Usa DataFrames ficticios pequeños (no los datos reales del proyecto) para
probar la lógica de `validate_train_test_split()` de forma aislada: casos
válidos, fuga de datos, distribución problemática, tamaño/estructura
insuficiente y (como capacidad genérica extra de la función) orden
temporal.
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

# Permite importar `pipelines.train_split_check_pipeline` al ejecutar
# pytest desde la raíz del proyecto o desde cualquier otro directorio,
# igual que hace `test_train_pipeline.py` (Issue 3).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines import model_validation_pipeline as mvp

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

    y_train = pd.Series(
        20.0 + 3.0 * x_train["numeric__lstat"], name="medv", index=x_train.index
    )
    y_test = pd.Series(
        20.0 + 3.0 * x_test["numeric__lstat"], name="medv", index=x_test.index
    )

    return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}


# ---------------------------------------------------------------------------
# CASO 1 — Split válido
# ---------------------------------------------------------------------------


def test_valid_split_passes_without_raising(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    result = mvp.validate_train_test_split(
        valid_split["x_train"],
        valid_split["x_test"],
        valid_split["y_train"],
        valid_split["y_test"],
        target_col="medv",
    )

    assert result.passed
    assert result.failed_checks() == []
    # Se registran múltiples checks, no solo un booleano "passed".
    assert len(result.checks) > 5
    assert any(
        c.name == "overlap_indices" and c.status == "passed" for c in result.checks
    )


# ---------------------------------------------------------------------------
# CASO 2 — Data leakage (registro compartido entre train y test)
# ---------------------------------------------------------------------------


def test_shared_index_raises_leakage_error(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train = valid_split["x_train"]
    x_test = valid_split["x_test"].copy()
    # Fuerza un índice de test a coincidir con uno de train.
    x_test.index = [x_train.index[0]] + list(x_test.index[1:])

    with pytest.raises(mvp.TrainTestSplitValidationError, match="overlap_indices"):
        mvp.validate_train_test_split(
            x_train,
            x_test,
            valid_split["y_train"],
            valid_split["y_test"],
            target_col="medv",
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

    with pytest.raises(mvp.TrainTestSplitValidationError, match="duplicate_rows"):
        mvp.validate_train_test_split(
            x_train, x_test, y_train, y_test, target_col="medv"
        )


def test_target_present_in_features_raises_leakage_error(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train = valid_split["x_train"].copy()
    x_test = valid_split["x_test"].copy()
    x_train["medv"] = valid_split["y_train"]
    x_test["medv"] = valid_split["y_test"]

    with pytest.raises(
        mvp.TrainTestSplitValidationError, match="target_not_in_features"
    ):
        mvp.validate_train_test_split(
            x_train,
            x_test,
            valid_split["y_train"],
            valid_split["y_test"],
            target_col="medv",
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
        result = mvp.validate_train_test_split(
            x_train,
            x_test,
            valid_split["y_train"],
            valid_split["y_test"],
            target_col="medv",
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
        result = mvp.validate_train_test_split(
            x_train,
            x_test,
            valid_split["y_train"],
            valid_split["y_test"],
            target_col="medv",
        )

    assert result.passed
    assert any(
        c.name == "new_category_binary__chas" and c.status == "warning"
        for c in result.checks
    )


# ---------------------------------------------------------------------------
# CASO 4 — Problema de tamaño/estructura
# ---------------------------------------------------------------------------


def test_train_set_too_small_raises_error(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train_pequeno = valid_split["x_train"].iloc[:5]
    y_train_pequeno = valid_split["y_train"].iloc[:5]

    with pytest.raises(mvp.TrainTestSplitValidationError, match="min_train_size"):
        mvp.validate_train_test_split(
            x_train_pequeno,
            valid_split["x_test"],
            y_train_pequeno,
            valid_split["y_test"],
            target_col="medv",
            min_train_size=50,
            min_test_size=15,
        )


def test_mismatched_target_length_raises_error(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    y_train_incompleto = valid_split["y_train"].iloc[:-1]

    with pytest.raises(mvp.TrainTestSplitValidationError, match="target_length_train"):
        mvp.validate_train_test_split(
            valid_split["x_train"],
            valid_split["x_test"],
            y_train_incompleto,
            valid_split["y_test"],
            target_col="medv",
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

    with pytest.raises(mvp.TrainTestSplitValidationError, match="temporal_order"):
        mvp.validate_train_test_split(
            x_train,
            x_test,
            valid_split["y_train"],
            valid_split["y_test"],
            target_col="medv",
            time_column="fecha",
        )


def test_temporal_order_respected_passes(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train = valid_split["x_train"].copy()
    x_test = valid_split["x_test"].copy()
    x_train["fecha"] = pd.date_range("2020-01-01", periods=len(x_train), freq="D")
    x_test["fecha"] = pd.date_range("2030-01-01", periods=len(x_test), freq="D")

    result = mvp.validate_train_test_split(
        x_train,
        x_test,
        valid_split["y_train"],
        valid_split["y_test"],
        target_col="medv",
        time_column="fecha",
    )

    assert result.passed
    assert any(
        c.name == "temporal_order" and c.status == "passed" for c in result.checks
    )


# ---------------------------------------------------------------------------
# TrainTestSplitValidationResult: estructura del resultado
# ---------------------------------------------------------------------------


def test_result_to_dict_contains_all_checks(
    valid_split: dict[str, pd.DataFrame | pd.Series],
) -> None:
    result = mvp.validate_train_test_split(
        valid_split["x_train"],
        valid_split["x_test"],
        valid_split["y_train"],
        valid_split["y_test"],
        target_col="medv",
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

    result = mvp.run_split_check_pipeline(
        x_train_path=x_train_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
        results_path=results_path,
        target_col="medv",
    )

    assert result.passed
    assert results_path.exists()
    with results_path.open(encoding="utf-8") as results_file:
        saved_payload = json.load(results_file)
    assert saved_payload["passed"] is True


def _make_valid_model_path(tmp_path: Path, x_train: pd.DataFrame, y_train: pd.Series) -> Path:
    """Entrena un modelo pequeño/rápido y lo guarda como si fuera la salida
    de `train_pipeline.py` (Issue 3), para las pruebas de Model Validation
    que necesitan un modelo ya persistido.
    """
    model = GradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(x_train, y_train)
    model_path = tmp_path / "modelo" / "train_pipeline_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


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
    x_test.index = [x_train.index[0]] + list(x_test.index[1:])  # fuerza fuga de datos

    x_train.to_parquet(x_train_path, engine="pyarrow")
    x_test.to_parquet(x_test_path, engine="pyarrow")
    valid_split["y_train"].to_frame().to_parquet(y_train_path, engine="pyarrow")
    valid_split["y_test"].to_frame().to_parquet(y_test_path, engine="pyarrow")

    with pytest.raises(mvp.TrainTestSplitValidationError):
        mvp.run_split_check_pipeline(
            x_train_path=x_train_path,
            x_test_path=x_test_path,
            y_train_path=y_train_path,
            y_test_path=y_test_path,
            results_path=results_path,
            target_col="medv",
        )

    assert results_path.exists()
    with results_path.open(encoding="utf-8") as results_file:
        saved_payload = json.load(results_file)
    assert saved_payload["passed"] is False
    assert any(c["name"] == "overlap_indices" for c in saved_payload["checks"])


# ===========================================================================
# MODEL VALIDATION (Issue 5)
# ===========================================================================


@pytest.fixture
def model_validation_data() -> dict[str, pd.DataFrame | pd.Series]:
    """Datos ficticios pequeños con relación aproximadamente lineal, en el
    mismo estilo de columnas (`numeric__`, `binary__`) que produce el
    feature pipeline. Se usan para probar la lógica de Model Validation de
    forma aislada, no los datos reales del proyecto.
    """
    rng = np.random.default_rng(123)
    n_train = 60
    x_train = pd.DataFrame(
        {
            "numeric__lstat": rng.normal(size=n_train),
            "numeric__rm": rng.normal(loc=6.0, scale=0.5, size=n_train),
            "binary__chas": rng.integers(0, 2, size=n_train).astype(float),
        }
    )
    noise = rng.normal(scale=0.5, size=n_train)
    y_train = pd.Series(
        20.0 - 3.0 * x_train["numeric__lstat"] + 2.0 * x_train["numeric__rm"] + noise,
        name="medv",
    )
    return {"x_train": x_train, "y_train": y_train}


def _fast_model_builder() -> GradientBoostingRegressor:
    """Modelo pequeño/rápido para pruebas unitarias (no cambia el modelo
    real del proyecto, solo acelera las pruebas de cross-validation).
    """
    return GradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=42)


# ---------------------------------------------------------------------------
# 1-3. Cross-validation: ejecución con datos válidos, métricas y resultados
# ---------------------------------------------------------------------------


def test_cross_validation_with_valid_data_returns_scores_per_fold(
    model_validation_data: dict[str, pd.DataFrame | pd.Series],
) -> None:
    result = mvp.validate_model_with_cross_validation(
        model_validation_data["x_train"],
        model_validation_data["y_train"],
        model_builder=_fast_model_builder,
        n_splits=3,
        random_state=42,
    )

    assert isinstance(result, mvp.CrossValidationResult)
    assert result.n_splits == 3
    assert set(result.scores.keys()) == {"MAE", "RMSE", "R2", "MAPE"}
    # Cada métrica debe tener exactamente un valor por fold.
    for metric_scores in result.scores.values():
        assert len(metric_scores) == 3


def test_cross_validation_mean_and_std_are_computed_correctly(
    model_validation_data: dict[str, pd.DataFrame | pd.Series],
) -> None:
    result = mvp.validate_model_with_cross_validation(
        model_validation_data["x_train"],
        model_validation_data["y_train"],
        model_builder=_fast_model_builder,
        n_splits=3,
        random_state=42,
    )

    for metric in ("MAE", "RMSE", "R2", "MAPE"):
        assert result.mean[metric] == pytest.approx(np.mean(result.scores[metric]))
        assert result.std[metric] == pytest.approx(np.std(result.scores[metric]))


def test_cross_validation_result_to_dict_contains_fold_and_summary_values(
    model_validation_data: dict[str, pd.DataFrame | pd.Series],
) -> None:
    result = mvp.validate_model_with_cross_validation(
        model_validation_data["x_train"],
        model_validation_data["y_train"],
        model_builder=_fast_model_builder,
        n_splits=3,
        random_state=42,
    )

    payload = result.to_dict()

    assert payload["strategy"] == "KFold(shuffle=True)"
    assert payload["n_splits"] == 3
    assert set(payload["scores_by_fold"].keys()) == {"MAE", "RMSE", "R2", "MAPE"}
    assert set(payload["mean"].keys()) == {"MAE", "RMSE", "R2", "MAPE"}
    assert set(payload["std"].keys()) == {"MAE", "RMSE", "R2", "MAPE"}


# ---------------------------------------------------------------------------
# 4. Comparación train / CV / test
# ---------------------------------------------------------------------------


def test_compare_model_performance_aligns_train_cv_and_test() -> None:
    cv_result = mvp.CrossValidationResult(
        scores={"MAE": [1.0, 1.2], "RMSE": [1.5, 1.7], "R2": [0.9, 0.88], "MAPE": [0.1, 0.11]},
        strategy="KFold(shuffle=True)",
        n_splits=2,
        random_state=42,
    )
    train_metrics = {"MAE": 0.5, "RMSE": 0.7, "R2": 0.97, "MAPE": 0.05}
    test_metrics = {"MAE": 1.3, "RMSE": 1.8, "R2": 0.85, "MAPE": 0.12}

    comparison = mvp.compare_model_performance(train_metrics, cv_result, test_metrics)

    assert comparison.train == train_metrics
    assert comparison.test == test_metrics
    assert comparison.cv_mean == cv_result.mean
    assert comparison.cv_std == cv_result.std
    payload = comparison.to_dict()
    assert set(payload.keys()) == {"train", "cross_validation_mean", "cross_validation_std", "test"}


# ---------------------------------------------------------------------------
# 5-6. Detección de posible overfitting / underfitting
# ---------------------------------------------------------------------------


def test_diagnose_generalization_detects_overfitting() -> None:
    # train R2 casi perfecto, CV y test notablemente más bajos.
    comparison = mvp.PerformanceComparison(
        train={"MAE": 0.1, "RMSE": 0.1, "R2": 0.99, "MAPE": 0.01},
        cv_mean={"MAE": 2.0, "RMSE": 2.5, "R2": 0.70, "MAPE": 0.15},
        cv_std={"MAE": 0.2, "RMSE": 0.3, "R2": 0.02, "MAPE": 0.02},
        test={"MAE": 2.1, "RMSE": 2.6, "R2": 0.68, "MAPE": 0.16},
    )

    with pytest.warns(UserWarning, match="overfitting_train_vs_cv"):
        checks = mvp.diagnose_generalization(comparison)

    checks_by_name = {c.name: c for c in checks}
    assert checks_by_name["overfitting_train_vs_cv"].status == "warning"
    assert checks_by_name["overfitting_train_vs_test"].status == "warning"
    assert checks_by_name["underfitting"].status == "passed"
    assert checks_by_name["generalization_summary"].status == "warning"
    assert "regularización" in checks_by_name["generalization_summary"].message


def test_diagnose_generalization_detects_underfitting() -> None:
    # train R2 ya bajo, CV similar: el modelo no ajusta ni siquiera train.
    comparison = mvp.PerformanceComparison(
        train={"MAE": 4.0, "RMSE": 5.0, "R2": 0.35, "MAPE": 0.25},
        cv_mean={"MAE": 4.1, "RMSE": 5.1, "R2": 0.33, "MAPE": 0.26},
        cv_std={"MAE": 0.3, "RMSE": 0.4, "R2": 0.02, "MAPE": 0.02},
        test={"MAE": 4.2, "RMSE": 5.2, "R2": 0.32, "MAPE": 0.27},
    )

    with pytest.warns(UserWarning, match="underfitting"):
        checks = mvp.diagnose_generalization(comparison)

    checks_by_name = {c.name: c for c in checks}
    assert checks_by_name["underfitting"].status == "warning"
    assert checks_by_name["overfitting_train_vs_cv"].status == "passed"
    assert checks_by_name["overfitting_train_vs_test"].status == "passed"
    assert "complejidad del modelo" in checks_by_name["generalization_summary"].message


def test_diagnose_generalization_passes_when_model_generalizes_well() -> None:
    comparison = mvp.PerformanceComparison(
        train={"MAE": 1.0, "RMSE": 1.2, "R2": 0.90, "MAPE": 0.08},
        cv_mean={"MAE": 1.1, "RMSE": 1.3, "R2": 0.88, "MAPE": 0.09},
        cv_std={"MAE": 0.1, "RMSE": 0.1, "R2": 0.01, "MAPE": 0.01},
        test={"MAE": 1.15, "RMSE": 1.35, "R2": 0.87, "MAPE": 0.09},
    )

    checks = mvp.diagnose_generalization(comparison)

    assert all(check.status == "passed" for check in checks)


# ---------------------------------------------------------------------------
# 7. Generación de evidencia
# ---------------------------------------------------------------------------


def test_save_model_validation_evidence_writes_metrics_log_and_plots(tmp_path: Path) -> None:
    cv_result = mvp.CrossValidationResult(
        scores={"MAE": [1.0, 1.1], "RMSE": [1.4, 1.5], "R2": [0.9, 0.89], "MAPE": [0.1, 0.1]},
        strategy="KFold(shuffle=True)",
        n_splits=2,
        random_state=42,
    )
    comparison = mvp.compare_model_performance(
        train_metrics={"MAE": 0.5, "RMSE": 0.7, "R2": 0.97, "MAPE": 0.05},
        cv_result=cv_result,
        test_metrics={"MAE": 1.2, "RMSE": 1.6, "R2": 0.88, "MAPE": 0.11},
    )
    diagnosis = mvp.diagnose_generalization(comparison)
    result = mvp.ModelValidationResult(
        train_metrics=comparison.train,
        cv_result=cv_result,
        test_metrics=comparison.test,
        comparison=comparison,
        diagnosis=diagnosis,
    )

    metrics_path = tmp_path / "metrics" / "model_validation_metrics.json"
    log_path = tmp_path / "logs" / "model_validation_log.txt"
    comparison_plot_path = tmp_path / "plots" / "train_cv_test_comparison.png"
    cv_plot_path = tmp_path / "plots" / "cv_scores_by_fold.png"

    mvp.save_model_validation_evidence(
        result,
        metrics_path=metrics_path,
        log_path=log_path,
        comparison_plot_path=comparison_plot_path,
        cv_plot_path=cv_plot_path,
    )

    assert metrics_path.exists()
    assert log_path.exists()
    assert comparison_plot_path.exists()
    assert cv_plot_path.exists()

    with metrics_path.open(encoding="utf-8") as metrics_file:
        saved_payload = json.load(metrics_file)
    assert saved_payload["train"] == comparison.train
    assert saved_payload["test"] == comparison.test
    assert len(saved_payload["diagnosis"]) == len(diagnosis)

    log_content = log_path.read_text(encoding="utf-8")
    assert "MODEL VALIDATION" in log_content
    assert "TRAIN" in log_content


# ---------------------------------------------------------------------------
# 8. Casos inválidos / errores controlados
# ---------------------------------------------------------------------------


def test_cross_validation_raises_error_for_n_splits_below_two(
    model_validation_data: dict[str, pd.DataFrame | pd.Series],
) -> None:
    with pytest.raises(ValueError, match="n_splits"):
        mvp.validate_model_with_cross_validation(
            model_validation_data["x_train"],
            model_validation_data["y_train"],
            model_builder=_fast_model_builder,
            n_splits=1,
        )


def test_cross_validation_raises_error_when_not_enough_rows(
    model_validation_data: dict[str, pd.DataFrame | pd.Series],
) -> None:
    x_train_pequeno = model_validation_data["x_train"].iloc[:3]
    y_train_pequeno = model_validation_data["y_train"].iloc[:3]

    with pytest.raises(ValueError, match="No hay suficientes registros"):
        mvp.validate_model_with_cross_validation(
            x_train_pequeno, y_train_pequeno, model_builder=_fast_model_builder, n_splits=5
        )


def test_run_model_validation_pipeline_raises_when_model_missing(
    tmp_path: Path, model_validation_data: dict[str, pd.DataFrame | pd.Series]
) -> None:
    x_train_path = tmp_path / "x_train_features.parquet"
    x_test_path = tmp_path / "x_test_features.parquet"
    y_train_path = tmp_path / "y_train.parquet"
    y_test_path = tmp_path / "y_test.parquet"

    x_train = model_validation_data["x_train"]
    y_train = model_validation_data["y_train"]
    x_train.iloc[:40].to_parquet(x_train_path, engine="pyarrow")
    x_train.iloc[40:].to_parquet(x_test_path, engine="pyarrow")
    y_train.iloc[:40].to_frame().to_parquet(y_train_path, engine="pyarrow")
    y_train.iloc[40:].to_frame().to_parquet(y_test_path, engine="pyarrow")

    modelo_inexistente = tmp_path / "no_existe" / "modelo.joblib"

    with pytest.raises(FileNotFoundError, match="train_pipeline"):
        mvp.run_model_validation_pipeline(
            x_train_path=x_train_path,
            x_test_path=x_test_path,
            y_train_path=y_train_path,
            y_test_path=y_test_path,
            model_path=modelo_inexistente,
        )


# ---------------------------------------------------------------------------
# 9. Reproducibilidad
# ---------------------------------------------------------------------------


def test_cross_validation_is_reproducible_with_same_random_state(
    model_validation_data: dict[str, pd.DataFrame | pd.Series],
) -> None:
    result_1 = mvp.validate_model_with_cross_validation(
        model_validation_data["x_train"],
        model_validation_data["y_train"],
        model_builder=_fast_model_builder,
        n_splits=3,
        random_state=42,
    )
    result_2 = mvp.validate_model_with_cross_validation(
        model_validation_data["x_train"],
        model_validation_data["y_train"],
        model_builder=_fast_model_builder,
        n_splits=3,
        random_state=42,
    )

    assert result_1.scores == result_2.scores


# ---------------------------------------------------------------------------
# Integración: run_model_validation_pipeline end-to-end (usa tmp_path, no
# toca los datos ni el modelo reales del proyecto).
# ---------------------------------------------------------------------------


def test_run_model_validation_pipeline_end_to_end(
    tmp_path: Path, model_validation_data: dict[str, pd.DataFrame | pd.Series]
) -> None:
    x_train_path = tmp_path / "x_train_features.parquet"
    x_test_path = tmp_path / "x_test_features.parquet"
    y_train_path = tmp_path / "y_train.parquet"
    y_test_path = tmp_path / "y_test.parquet"

    x_train = model_validation_data["x_train"]
    y_train = model_validation_data["y_train"]
    x_train_split = x_train.iloc[:45]
    x_test_split = x_train.iloc[45:]
    y_train_split = y_train.iloc[:45]
    y_test_split = y_train.iloc[45:]

    x_train_split.to_parquet(x_train_path, engine="pyarrow")
    x_test_split.to_parquet(x_test_path, engine="pyarrow")
    y_train_split.to_frame().to_parquet(y_train_path, engine="pyarrow")
    y_test_split.to_frame().to_parquet(y_test_path, engine="pyarrow")

    model_path = _make_valid_model_path(tmp_path, x_train_split, y_train_split)

    metrics_path = tmp_path / "model_validation" / "metrics" / "model_validation_metrics.json"
    log_path = tmp_path / "model_validation" / "logs" / "model_validation_log.txt"
    comparison_plot_path = tmp_path / "model_validation" / "plots" / "comparison.png"
    cv_plot_path = tmp_path / "model_validation" / "plots" / "cv.png"

    result = mvp.run_model_validation_pipeline(
        x_train_path=x_train_path,
        x_test_path=x_test_path,
        y_train_path=y_train_path,
        y_test_path=y_test_path,
        model_path=model_path,
        n_splits=3,
        random_state=42,
        metrics_path=metrics_path,
        log_path=log_path,
        comparison_plot_path=comparison_plot_path,
        cv_plot_path=cv_plot_path,
    )

    assert isinstance(result, mvp.ModelValidationResult)
    assert set(result.train_metrics.keys()) == {"MAE", "RMSE", "R2", "MAPE"}
    assert result.cv_result.n_splits == 3
    assert metrics_path.exists()
    assert log_path.exists()
    assert comparison_plot_path.exists()
    assert cv_plot_path.exists()
