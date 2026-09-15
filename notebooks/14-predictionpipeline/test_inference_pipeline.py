"""Pruebas unitarias para pipelines/inference_pipeline.py (Issue 6).

Usa un modelo dummy (`sklearn.dummy.DummyRegressor`), un preprocesador
ajustado sobre datos ficticios pequeños y archivos temporales (`tmp_path`)
para verificar, de forma aislada y sin depender del modelo real ni de los
datos reales del proyecto, que el inference pipeline puede: cargar el
modelo y el preprocesador, leer datos nuevos, aplicar las mismas
transformaciones del entrenamiento (sin reajustarlas), generar
predicciones, y guardar resultados y evidencia. También cubre el manejo
controlado de datos inválidos o con columnas faltantes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

# Permite importar `pipelines.inference_pipeline` al ejecutar pytest desde
# la raíz del proyecto o desde cualquier otro directorio, igual que hacen
# los tests de las Issues anteriores.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines import feature_pipeline_validado as fp
from pipelines import inference_pipeline as ip
from pipelines import train_pipeline as tp

# ---------------------------------------------------------------------------
# Fixtures: datos ficticios pequeños con la misma forma (columnas crudas)
# que consume `feature_pipeline_validado.py`, y un preprocesador + modelo
# dummy ajustados sobre esos datos ficticios (nunca sobre datos reales).
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_training_like_df() -> pd.DataFrame:
    """DataFrame ficticio con las mismas columnas crudas que `Informacion.txt`.

    Se usa SOLO para ajustar (`fit`) un preprocesador de prueba; no son
    los datos reales del proyecto.
    """
    rng = np.random.default_rng(42)
    n_rows = 60
    return pd.DataFrame(
        {
            "crim": rng.uniform(0.01, 20.0, size=n_rows),
            "zn": rng.uniform(0.0, 100.0, size=n_rows),
            "indus": rng.uniform(0.5, 25.0, size=n_rows),
            "chas": rng.integers(0, 2, size=n_rows).astype(float),
            "nox": rng.uniform(0.3, 0.9, size=n_rows),
            "rm": rng.uniform(4.0, 8.5, size=n_rows),
            "age": rng.uniform(2.0, 99.0, size=n_rows),
            "dis": rng.uniform(1.5, 10.0, size=n_rows),
            "rad": rng.choice([1, 2, 3, 4, 5, 24], size=n_rows).astype(float),
            "tax": rng.uniform(180.0, 700.0, size=n_rows),
            "ptratio": rng.uniform(13.0, 22.0, size=n_rows),
            "black": rng.uniform(50.0, 396.9, size=n_rows),
            "lstat": rng.uniform(2.0, 30.0, size=n_rows),
            "medv": rng.uniform(10.0, 45.0, size=n_rows),
        }
    )


@pytest.fixture
def fitted_preprocessor_and_model(raw_training_like_df: pd.DataFrame) -> dict:
    """Ajusta un preprocesador y un modelo dummy sobre datos ficticios.

    Reutiliza las MISMAS funciones que usa `feature_pipeline_validado.py`
    (`flag_censored_target`, `drop_redundant_features`,
    `add_rad_group_feature`, `build_preprocessor`) para construir el
    escenario de prueba, en vez de reimplementar la transformación a
    mano; así el test refleja el flujo real de entrenamiento.
    """
    df = fp.flag_censored_target(raw_training_like_df)
    df = fp.drop_redundant_features(df)
    df = fp.add_rad_group_feature(df)

    x = df.drop(columns=[fp.TARGET_COL])
    y = df[fp.TARGET_COL]

    preprocessor = fp.build_preprocessor()
    x_transformed = preprocessor.fit_transform(x)
    feature_names = preprocessor.get_feature_names_out()
    x_transformed = pd.DataFrame(x_transformed, columns=feature_names)

    x_model = tp.drop_leakage_columns(x_transformed)

    model = DummyRegressor(strategy="mean")
    model.fit(x_model, y)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "expected_feature_names": list(x_model.columns),
    }


@pytest.fixture
def model_path(tmp_path: Path, fitted_preprocessor_and_model: dict) -> Path:
    path = tmp_path / "models" / "dummy_model.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_preprocessor_and_model["model"], path)
    return path


@pytest.fixture
def preprocessor_path(tmp_path: Path, fitted_preprocessor_and_model: dict) -> Path:
    path = tmp_path / "features" / "preprocessor.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_preprocessor_and_model["preprocessor"], path)
    return path


@pytest.fixture
def new_data_with_target_df() -> pd.DataFrame:
    """Datos "nuevos" ficticios que sí incluyen `medv` (permite hindcast)."""
    rng = np.random.default_rng(7)
    n_rows = 5
    return pd.DataFrame(
        {
            "ID": range(1001, 1001 + n_rows),
            "crim": rng.uniform(0.01, 20.0, size=n_rows),
            "zn": rng.uniform(0.0, 100.0, size=n_rows),
            "indus": rng.uniform(0.5, 25.0, size=n_rows),
            "chas": rng.integers(0, 2, size=n_rows).astype(float),
            "nox": rng.uniform(0.3, 0.9, size=n_rows),
            "rm": rng.uniform(4.0, 8.5, size=n_rows),
            "age": rng.uniform(2.0, 99.0, size=n_rows),
            "dis": rng.uniform(1.5, 10.0, size=n_rows),
            "rad": rng.choice([1, 2, 3, 4, 5, 24], size=n_rows).astype(float),
            "tax": rng.uniform(180.0, 700.0, size=n_rows),
            "ptratio": rng.uniform(13.0, 22.0, size=n_rows),
            "black": rng.uniform(50.0, 396.9, size=n_rows),
            "lstat": rng.uniform(2.0, 30.0, size=n_rows),
            "medv": rng.uniform(10.0, 45.0, size=n_rows),
        }
    )


@pytest.fixture
def new_data_without_target_df(new_data_with_target_df: pd.DataFrame) -> pd.DataFrame:
    """Los mismos datos "nuevos", pero sin `medv` (caso real de inferencia)."""
    return new_data_with_target_df.drop(columns=["medv"])


# ---------------------------------------------------------------------------
# 1. Carga del modelo
# ---------------------------------------------------------------------------


def test_load_model_returns_model_with_matching_predictions(
    model_path: Path, fitted_preprocessor_and_model: dict
) -> None:
    loaded_model = ip.load_model(model_path)

    x_dummy = pd.DataFrame(
        np.zeros((3, len(fitted_preprocessor_and_model["expected_feature_names"]))),
        columns=fitted_preprocessor_and_model["expected_feature_names"],
    )
    original_predictions = fitted_preprocessor_and_model["model"].predict(x_dummy)
    loaded_predictions = loaded_model.predict(x_dummy)
    np.testing.assert_allclose(original_predictions, loaded_predictions)


def test_load_model_raises_file_not_found_when_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "no_existe.joblib"
    with pytest.raises(FileNotFoundError):
        ip.load_model(missing_path)


# ---------------------------------------------------------------------------
# 2. Carga del preprocesador
# ---------------------------------------------------------------------------


def test_load_preprocessor_returns_fitted_transformer(
    preprocessor_path: Path, raw_training_like_df: pd.DataFrame
) -> None:
    loaded_preprocessor = ip.load_preprocessor(preprocessor_path)

    df = fp.flag_censored_target(raw_training_like_df)
    df = fp.drop_redundant_features(df)
    df = fp.add_rad_group_feature(df)
    x = df.drop(columns=[fp.TARGET_COL])

    # Un preprocesador ya ajustado debe poder transformar sin volver a
    # llamar `fit`, y producir columnas conocidas.
    transformed = loaded_preprocessor.transform(x)
    assert transformed.shape[0] == len(x)
    assert "binary__medv_censored" in loaded_preprocessor.get_feature_names_out()


def test_load_preprocessor_raises_file_not_found_when_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "no_existe.joblib"
    with pytest.raises(FileNotFoundError):
        ip.load_preprocessor(missing_path)


# ---------------------------------------------------------------------------
# 3. Lectura de datos nuevos
# ---------------------------------------------------------------------------


def test_load_new_data_reads_csv_with_required_columns(
    tmp_path: Path, new_data_without_target_df: pd.DataFrame
) -> None:
    input_path = tmp_path / "nuevas_casas.csv"
    new_data_without_target_df.to_csv(input_path, index=False)

    loaded = ip.load_new_data(input_path)

    assert len(loaded) == len(new_data_without_target_df)
    for col in ip.REQUIRED_INPUT_COLUMNS:
        assert col in loaded.columns


def test_load_new_data_reads_parquet(
    tmp_path: Path, new_data_without_target_df: pd.DataFrame
) -> None:
    input_path = tmp_path / "nuevas_casas.parquet"
    new_data_without_target_df.to_parquet(input_path, engine="pyarrow")

    loaded = ip.load_new_data(input_path)

    assert len(loaded) == len(new_data_without_target_df)


def test_load_new_data_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ip.load_new_data(tmp_path / "no_existe.csv")


def test_load_new_data_raises_for_unsupported_extension(
    tmp_path: Path, new_data_without_target_df: pd.DataFrame
) -> None:
    input_path = tmp_path / "nuevas_casas.txt"
    new_data_without_target_df.to_csv(input_path, index=False)

    with pytest.raises(ip.InferenceDataError):
        ip.load_new_data(input_path)


def test_load_new_data_raises_for_empty_file(tmp_path: Path) -> None:
    input_path = tmp_path / "vacio.csv"
    pd.DataFrame(columns=ip.REQUIRED_INPUT_COLUMNS).to_csv(input_path, index=False)

    with pytest.raises(ip.InferenceDataError):
        ip.load_new_data(input_path)


def test_load_new_data_raises_for_missing_columns(
    tmp_path: Path, new_data_without_target_df: pd.DataFrame
) -> None:
    incomplete_df = new_data_without_target_df.drop(columns=["rm", "lstat"])
    input_path = tmp_path / "incompleto.csv"
    incomplete_df.to_csv(input_path, index=False)

    with pytest.raises(ip.InferenceDataError, match="rm"):
        ip.load_new_data(input_path)


# ---------------------------------------------------------------------------
# 4. Aplicación de transformaciones (sin reajustar el preprocesador)
# ---------------------------------------------------------------------------


def test_transform_data_matches_preprocessor_transform_without_refitting(
    fitted_preprocessor_and_model: dict, new_data_without_target_df: pd.DataFrame
) -> None:
    preprocessor = fitted_preprocessor_and_model["preprocessor"]

    x_model = ip.transform_data(new_data_without_target_df, preprocessor)

    # Mismo número de filas que la entrada, mismas columnas que usa el
    # modelo (leakage ya excluido) y sin nulos.
    assert len(x_model) == len(new_data_without_target_df)
    assert list(x_model.columns) == fitted_preprocessor_and_model["expected_feature_names"]
    assert x_model.isna().sum().sum() == 0
    assert "binary__medv_censored" not in x_model.columns


def test_transform_data_works_with_and_without_target_column(
    fitted_preprocessor_and_model: dict,
    new_data_with_target_df: pd.DataFrame,
    new_data_without_target_df: pd.DataFrame,
) -> None:
    preprocessor = fitted_preprocessor_and_model["preprocessor"]

    x_with_target = ip.transform_data(new_data_with_target_df, preprocessor)
    x_without_target = ip.transform_data(new_data_without_target_df, preprocessor)

    # Las columnas de features del modelo no dependen de si vino `medv`.
    assert list(x_with_target.columns) == list(x_without_target.columns)
    assert len(x_with_target) == len(x_without_target)


def test_transform_data_does_not_refit_preprocessor(
    fitted_preprocessor_and_model: dict, new_data_without_target_df: pd.DataFrame
) -> None:
    """El preprocesador debe quedar intacto (mismos parámetros ajustados)."""
    preprocessor = fitted_preprocessor_and_model["preprocessor"]
    numeric_scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    mean_before = numeric_scaler.mean_.copy()

    ip.transform_data(new_data_without_target_df, preprocessor)

    np.testing.assert_array_equal(mean_before, numeric_scaler.mean_)


# ---------------------------------------------------------------------------
# 5. Generación de predicciones
# ---------------------------------------------------------------------------


def test_generate_predictions_returns_expected_number(
    fitted_preprocessor_and_model: dict, new_data_without_target_df: pd.DataFrame
) -> None:
    preprocessor = fitted_preprocessor_and_model["preprocessor"]
    model = fitted_preprocessor_and_model["model"]

    x_model = ip.transform_data(new_data_without_target_df, preprocessor)
    predictions = ip.generate_predictions(model, x_model)

    assert len(predictions) == len(new_data_without_target_df)
    assert isinstance(predictions, np.ndarray)


# ---------------------------------------------------------------------------
# 6. Formato del resultado (identificador + columna de predicción)
# ---------------------------------------------------------------------------


def test_build_predictions_output_has_id_and_prediction_column(
    new_data_with_target_df: pd.DataFrame,
) -> None:
    predictions = np.arange(len(new_data_with_target_df), dtype=float)

    output = ip.build_predictions_output(new_data_with_target_df, predictions)

    assert ip.ID_COLUMN in output.columns
    assert ip.PREDICTION_COL in output.columns
    assert ip.TARGET_COL in output.columns  # se conserva para poder comparar
    assert list(output[ip.ID_COLUMN]) == list(new_data_with_target_df[ip.ID_COLUMN])
    np.testing.assert_array_equal(output[ip.PREDICTION_COL].to_numpy(), predictions)


def test_build_predictions_output_uses_index_when_id_missing(
    new_data_without_target_df: pd.DataFrame,
) -> None:
    df_without_id = new_data_without_target_df.drop(columns=["ID"])
    predictions = np.arange(len(df_without_id), dtype=float)

    output = ip.build_predictions_output(df_without_id, predictions)

    assert list(output[ip.ID_COLUMN]) == list(df_without_id.index)
    assert ip.TARGET_COL not in output.columns  # no había `medv` en la entrada


# ---------------------------------------------------------------------------
# 7. Almacenamiento de resultados
# ---------------------------------------------------------------------------


def test_save_predictions_writes_csv_with_expected_content(tmp_path: Path) -> None:
    predictions_df = pd.DataFrame({"ID": [1, 2], "medv_predicho": [20.5, 30.1]})
    output_path = tmp_path / "predictions" / "predictions.csv"

    saved_path = ip.save_predictions(predictions_df, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    reloaded = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(reloaded, predictions_df)


def test_plot_predictions_creates_file_with_target(tmp_path: Path) -> None:
    predictions_df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "medv": [20.0, 25.0, 30.0],
            "medv_predicho": [21.0, 24.0, 29.5],
        }
    )
    plot_path = tmp_path / "plots" / "grafico.png"

    saved_path = ip.plot_predictions(predictions_df, plot_path)

    assert saved_path == plot_path
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_plot_predictions_creates_file_without_target(tmp_path: Path) -> None:
    predictions_df = pd.DataFrame({"ID": [1, 2, 3], "medv_predicho": [21.0, 24.0, 29.5]})
    plot_path = tmp_path / "plots" / "grafico.png"

    saved_path = ip.plot_predictions(predictions_df, plot_path)

    assert saved_path.exists()


def test_write_inference_log_creates_file_with_expected_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "inference_log.txt"

    saved_path = ip.write_inference_log(
        model_path=Path("modelo.joblib"),
        preprocessor_path=Path("preprocesador.joblib"),
        input_path=Path("nuevas_casas.csv"),
        n_records=5,
        features_used=["numeric__lstat", "binary__chas"],
        output_path=Path("predictions.csv"),
        plot_path=Path("grafico.png"),
        log_path=log_path,
    )

    assert saved_path == log_path
    content = log_path.read_text(encoding="utf-8")
    assert "modelo.joblib" in content
    assert "preprocesador.joblib" in content
    assert "nuevas_casas.csv" in content
    assert "5" in content


# ---------------------------------------------------------------------------
# 8. Flujo completo de inferencia con datos sintéticos (extremo a extremo)
# ---------------------------------------------------------------------------


def test_run_inference_pipeline_end_to_end_without_target(
    tmp_path: Path,
    model_path: Path,
    preprocessor_path: Path,
    new_data_without_target_df: pd.DataFrame,
    fitted_preprocessor_and_model: dict,
) -> None:
    input_path = tmp_path / "nuevas_casas.csv"
    new_data_without_target_df.to_csv(input_path, index=False)

    output_path = tmp_path / "predictions" / "predictions.csv"
    plot_path = tmp_path / "predictions" / "plots" / "grafico.png"
    log_path = tmp_path / "predictions" / "logs" / "inference_log.txt"

    results = ip.run_inference_pipeline(
        input_path=input_path,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        output_path=output_path,
        plot_path=plot_path,
        log_path=log_path,
    )

    assert results["n_records"] == len(new_data_without_target_df)
    assert results["features_used"] == fitted_preprocessor_and_model["expected_feature_names"]
    assert output_path.exists()
    assert plot_path.exists()
    assert log_path.exists()

    saved_predictions = pd.read_csv(output_path)
    assert len(saved_predictions) == len(new_data_without_target_df)
    assert ip.PREDICTION_COL in saved_predictions.columns
    assert ip.ID_COLUMN in saved_predictions.columns
    assert saved_predictions[ip.PREDICTION_COL].isna().sum() == 0


def test_run_inference_pipeline_end_to_end_with_target(
    tmp_path: Path,
    model_path: Path,
    preprocessor_path: Path,
    new_data_with_target_df: pd.DataFrame,
) -> None:
    input_path = tmp_path / "nuevas_casas_con_medv.csv"
    new_data_with_target_df.to_csv(input_path, index=False)

    output_path = tmp_path / "predictions" / "predictions.csv"
    plot_path = tmp_path / "predictions" / "plots" / "grafico.png"
    log_path = tmp_path / "predictions" / "logs" / "inference_log.txt"

    ip.run_inference_pipeline(
        input_path=input_path,
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        output_path=output_path,
        plot_path=plot_path,
        log_path=log_path,
    )

    saved_predictions = pd.read_csv(output_path)
    assert ip.TARGET_COL in saved_predictions.columns  # se conservó para comparar
    assert len(saved_predictions) == len(new_data_with_target_df)


def test_run_inference_pipeline_raises_when_model_missing(
    tmp_path: Path,
    preprocessor_path: Path,
    new_data_without_target_df: pd.DataFrame,
) -> None:
    input_path = tmp_path / "nuevas_casas.csv"
    new_data_without_target_df.to_csv(input_path, index=False)

    with pytest.raises(FileNotFoundError):
        ip.run_inference_pipeline(
            input_path=input_path,
            model_path=tmp_path / "no_existe_modelo.joblib",
            preprocessor_path=preprocessor_path,
            output_path=tmp_path / "predictions.csv",
            plot_path=tmp_path / "grafico.png",
            log_path=tmp_path / "log.txt",
        )


# ---------------------------------------------------------------------------
# 9. Manejo controlado de datos inválidos o columnas faltantes
# ---------------------------------------------------------------------------


def test_run_inference_pipeline_raises_controlled_error_for_missing_columns(
    tmp_path: Path,
    model_path: Path,
    preprocessor_path: Path,
    new_data_without_target_df: pd.DataFrame,
) -> None:
    incomplete_df = new_data_without_target_df.drop(columns=["nox", "dis"])
    input_path = tmp_path / "incompleto.csv"
    incomplete_df.to_csv(input_path, index=False)

    with pytest.raises(ip.InferenceDataError):
        ip.run_inference_pipeline(
            input_path=input_path,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            output_path=tmp_path / "predictions.csv",
            plot_path=tmp_path / "grafico.png",
            log_path=tmp_path / "log.txt",
        )


def test_run_inference_pipeline_raises_controlled_error_for_unsupported_format(
    tmp_path: Path,
    model_path: Path,
    preprocessor_path: Path,
    new_data_without_target_df: pd.DataFrame,
) -> None:
    input_path = tmp_path / "nuevas_casas.json"
    new_data_without_target_df.to_json(input_path, orient="records")

    with pytest.raises(ip.InferenceDataError):
        ip.run_inference_pipeline(
            input_path=input_path,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            output_path=tmp_path / "predictions.csv",
            plot_path=tmp_path / "grafico.png",
            log_path=tmp_path / "log.txt",
        )
