"""Train/Test Split Check pipeline para el proyecto Precios-casas-Boston.

Issue 4 — Train/Test Split Check.

Esta Issue NO rehace el split ni el entrenamiento (eso ya existe en
`feature_pipeline_validado.py` e Issue 3 / `train_pipeline.py`). Lo que
hace es **verificar** que la separación train/test que consume
`train_pipeline.py` (los parquet de `data/03_primary/`) es correcta:
sin fuga de datos, con tamaños razonables y con distribuciones
comparables entre los dos conjuntos.

Referencia oficial revisada para diseñar los checks (curso "Ciencia de
Datos en Producción" — Train/Test validation):
https://joserzapata.github.io/courses/ciencia-datos-en-produccion/data-validation/train_test-checks/

La referencia usa DeepChecks (`train_test_validation()`), que agrupa
checks como "Index Train-Test Leakage", "Train Test Samples Mix",
"Datasets Size Comparison", "Feature Drift", "Label Drift" y "New
Category Train Test" (los checks de fecha no aplican: el dataset de
Boston no tiene columnas de tipo fecha). DeepChecks NO es obligatorio
según la Issue, así que aquí se reimplementan los mismos conceptos con
pandas/numpy/scipy (dependencias que el proyecto ya tiene, scipy es
dependencia de scikit-learn), sin agregar una librería pesada nueva.

Contexto del proyecto (revisado en los notebooks 01, 02 y 06, y en
`Informacion.txt`) relevante para estos checks:

* El problema es de **regresión** (target `medv`), NO es un problema
  temporal: el dataset no tiene ninguna columna de fecha/tiempo. Por
  eso el check de orden temporal existe en este módulo como capacidad
  genérica (parámetro opcional `time_column`), pero el pipeline real
  del proyecto (`run_split_check_pipeline`) NO lo activa.
* No existe una columna de ID de negocio (se elimina en el notebook 02
  de exploración inicial). El identificador natural de cada fila es el
  índice de pandas, que se preserva sin resetear a lo largo de
  `feature_pipeline_validado.py` (`split_train_test` ->
  `fit_transform_features`). Por eso el check principal de fuga usa el
  índice, igual que hace `validate_processed_features` en el feature
  pipeline (Issue 2); aquí además se agrega un check de contenido
  duplicado (todas las columnas en común) como red de seguridad
  adicional, independiente del índice.

Flujo:
    leer features/targets ya persistidos por el feature pipeline
    (mismas rutas y misma función que usa `train_pipeline.py`)
    -> ejecutar `validate_train_test_split()`
    -> registrar resultados (qué pasó, qué falló, qué generó advertencia)
    -> error controlado si hay un problema crítico (p. ej. fuga de datos)

Uso:
    python pipelines/train_split_check_pipeline.py
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

# Rutas de los artefactos generados por el feature pipeline.
# Esta Issue valida directamente el mismo split persistido en data/03_primary.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "03_primary"
MODEL_OUTPUT_DIR = DATA_DIR / "07_model_output"

X_TRAIN_PATH = FEATURES_DIR / "x_train_features.parquet"
X_TEST_PATH = FEATURES_DIR / "x_test_features.parquet"
Y_TRAIN_PATH = FEATURES_DIR / "y_train.parquet"
Y_TEST_PATH = FEATURES_DIR / "y_test.parquet"
TARGET_COL = "medv"

# Resultados propios de esta Issue.
RESULTS_PATH = MODEL_OUTPUT_DIR / "train_split_check_results.json"


def load_processed_features(
    x_train_path: Path = X_TRAIN_PATH,
    x_test_path: Path = X_TEST_PATH,
    y_train_path: Path = Y_TRAIN_PATH,
    y_test_path: Path = Y_TEST_PATH,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Lee las features y targets procesados por el feature pipeline."""
    for path in (x_train_path, x_test_path, y_train_path, y_test_path):
        if not path.exists():
            msg = (
                f"No se encontró '{path}'. Ejecuta primero "
                "'python pipelines/feature_pipeline_validado.py' para generar "
                "las features procesadas."
            )
            raise FileNotFoundError(msg)

    x_train = pd.read_parquet(x_train_path, engine="pyarrow")
    x_test = pd.read_parquet(x_test_path, engine="pyarrow")
    y_train = pd.read_parquet(y_train_path, engine="pyarrow")[target_col]
    y_test = pd.read_parquet(y_test_path, engine="pyarrow")[target_col]

    return x_train, x_test, y_train, y_test


# --- Umbrales de validación ---------------------------------------------
# El dataset de Boston tiene 343 filas tras la limpieza (ver notebook 02),
# y `split_train_test` usa test_size=0.2 -> ~274 filas de train y ~69 de
# test. Los mínimos se fijan cómodamente por debajo de esos valores reales
# (para no generar falsos positivos con el dataset real) pero lo
# suficientemente altos como para detectar un split mal configurado
# (p. ej. un train/test invertido por error, o un subconjunto minúsculo).
DEFAULT_MIN_TRAIN_SIZE = 50
DEFAULT_MIN_TEST_SIZE = 15

# Rango razonable de la proporción test/train. DeepChecks, en la
# referencia del curso, usa como condición por defecto que esa proporción
# sea mayor a 0.01 ("Datasets Size Comparison"); aquí se es un poco más
# estricto (0.05) porque en este proyecto se conoce de antemano que el
# split es 80/20 (proporción ~0.25), y se agrega un máximo (1.5) para
# detectar el caso contrario (test mucho más grande que train).
DEFAULT_MIN_TEST_TRAIN_RATIO = 0.05
DEFAULT_MAX_TEST_TRAIN_RATIO = 1.5

# Umbral de "drift" (diferencia de distribución) entre train y test, tanto
# para el estadístico de Kolmogorov-Smirnov (variables numéricas) como
# para la diferencia máxima de proporciones (variables categóricas). Se
# usa el mismo valor (0.2) que DeepChecks usa por defecto para sus checks
# "Feature Drift" y "Label Drift" en la referencia del curso.
DEFAULT_DRIFT_WARNING_THRESHOLD = 0.2

# Una columna numérica con máximo 2 valores distintos (p. ej. 0/1 de una
# variable binaria u One-Hot ya codificada) se trata como categórica para
# los checks de distribución, no como una variable continua.
DEFAULT_MAX_BINARY_UNIQUE = 2


class TrainTestSplitValidationError(Exception):
    """Error de validación del split train/test (Issue 4).

    Se lanza cuando se detecta un problema crítico en la separación entre
    train y test (fuga de datos, target inconsistente o conjuntos
    demasiado pequeños). Es análogo en espíritu a `DataValidationError`
    del feature pipeline (Issue 2), pero específico de la validez del
    split, no de la calidad general de los datos de entrada.

    Expone el atributo `checks` (lista de `CheckResult`) con el detalle
    completo de todos los checks ejecutados, no solo los que fallaron,
    para que quien capture el error pueda registrar el resultado
    completo si lo necesita.
    """

    def __init__(self, message: str, checks: list[CheckResult] | None = None) -> None:
        super().__init__(message)
        self.checks: list[CheckResult] = checks or []


@dataclass
class CheckResult:
    """Resultado individual de un check de validación del split.

    `status` es uno de: "passed", "warning", "failed".
    """

    name: str
    status: str
    message: str


@dataclass
class TrainTestSplitValidationResult:
    """Resultado consolidado de `validate_train_test_split()`.

    No es solamente un booleano: guarda el detalle de cada check
    ejecutado para poder responder qué se comprobó, qué pasó, qué falló
    y qué generó advertencias (requisito explícito de la Issue).
    """

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True si ningún check crítico falló (puede tener advertencias)."""
        return not any(check.status == "failed" for check in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(check.status == "warning" for check in self.checks)

    def failed_checks(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status == "failed"]

    def warning_checks(self) -> list[CheckResult]:
        return [check for check in self.checks if check.status == "warning"]

    def summary(self) -> str:
        """Mensaje legible con el resultado de cada check (no solo un booleano)."""
        icon = {"passed": "[OK]  ", "warning": "[WARN]", "failed": "[FAIL]"}
        lines = ["Resultado de la validación del split train/test:"]
        for check in self.checks:
            lines.append(f"  {icon[check.status]} {check.name}: {check.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "has_warnings": self.has_warnings,
            "checks": [
                {"name": c.name, "status": c.status, "message": c.message}
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# Checks individuales (funciones pequeñas y testeables por separado)
# ---------------------------------------------------------------------------


def _check_index_overlap(x_train: pd.DataFrame, x_test: pd.DataFrame) -> CheckResult:
    """Check de fuga de datos vía índice compartido (equivalente a "Index
    Train-Test Leakage" en la referencia del curso).

    El índice de pandas es el identificador natural de cada fila en este
    proyecto (no hay columna de ID de negocio) y se preserva sin
    resetear a lo largo de `feature_pipeline_validado.py`.
    """
    overlap = set(x_train.index) & set(x_test.index)
    if overlap:
        ejemplo = sorted(overlap, key=str)[:5]
        return CheckResult(
            "overlap_indices",
            "failed",
            f"Se encontraron {len(overlap)} índice(s) compartido(s) entre "
            f"train y test (posible fuga de datos). Ejemplos: {ejemplo}.",
        )
    return CheckResult(
        "overlap_indices", "passed", "No hay índices compartidos entre train y test."
    )


def _check_duplicate_rows(
    x_train: pd.DataFrame, x_test: pd.DataFrame, id_columns: list[str] | None
) -> CheckResult:
    """Check de solapamiento por contenido (equivalente a "Train Test
    Samples Mix").

    Si se pasa `id_columns`, se usa esa clave. Si no, se usa la
    combinación de TODAS las columnas en común entre train y test: para
    variables continuas como las de este proyecto, dos registros con
    exactamente los mismos valores en todas las columnas son, en la
    práctica, la misma observación duplicada entre los dos conjuntos.
    """
    columnas_clave = (
        id_columns
        if id_columns
        else [col for col in x_train.columns if col in x_test.columns]
    )
    if not columnas_clave:
        return CheckResult(
            "duplicate_rows",
            "passed",
            "No hay columnas en común entre train y test para comparar duplicados.",
        )

    claves_train = set(x_train[columnas_clave].apply(tuple, axis=1))
    claves_test = set(x_test[columnas_clave].apply(tuple, axis=1))
    duplicados = claves_train & claves_test

    if duplicados:
        return CheckResult(
            "duplicate_rows",
            "failed",
            f"Se encontraron {len(duplicados)} registro(s) con el mismo contenido "
            f"en train y test (columnas comparadas: {columnas_clave}); posible "
            "fuga de datos / solapamiento.",
        )
    return CheckResult(
        "duplicate_rows",
        "passed",
        f"No hay registros con contenido duplicado entre train y test "
        f"({len(columnas_clave)} columna(s) comparada(s)).",
    )


def _check_target_presence(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_col: str,
) -> list[CheckResult]:
    """Checks de presencia/consistencia del target en ambos conjuntos."""
    checks: list[CheckResult] = []

    if target_col in x_train.columns or target_col in x_test.columns:
        checks.append(
            CheckResult(
                "target_not_in_features",
                "failed",
                f"La columna target '{target_col}' está presente entre las "
                "features de train y/o test; debe excluirse antes de "
                "entrenar (fuga directa del target).",
            )
        )
    else:
        checks.append(
            CheckResult(
                "target_not_in_features",
                "passed",
                f"El target '{target_col}' no está presente entre las columnas de features.",
            )
        )

    if len(x_train) != len(y_train):
        checks.append(
            CheckResult(
                "target_length_train",
                "failed",
                f"x_train tiene {len(x_train)} fila(s) pero y_train tiene {len(y_train)}.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "target_length_train",
                "passed",
                "x_train y y_train tienen el mismo número de filas.",
            )
        )

    if len(x_test) != len(y_test):
        checks.append(
            CheckResult(
                "target_length_test",
                "failed",
                f"x_test tiene {len(x_test)} fila(s) pero y_test tiene {len(y_test)}.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "target_length_test",
                "passed",
                "x_test y y_test tienen el mismo número de filas.",
            )
        )

    train_nulls = int(pd.isna(y_train).sum())
    test_nulls = int(pd.isna(y_test).sum())
    if train_nulls or test_nulls:
        checks.append(
            CheckResult(
                "target_no_nulls",
                "failed",
                f"El target tiene valores nulos: {train_nulls} en train, "
                f"{test_nulls} en test.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "target_no_nulls",
                "passed",
                "El target no tiene valores nulos ni en train ni en test.",
            )
        )

    return checks


def _check_sizes(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    min_train_size: int,
    min_test_size: int,
    min_ratio: float,
    max_ratio: float,
) -> list[CheckResult]:
    """Checks de tamaño de los conjuntos (equivalente a "Datasets Size Comparison")."""
    checks: list[CheckResult] = []
    n_train, n_test = len(x_train), len(x_test)

    if n_train < min_train_size:
        checks.append(
            CheckResult(
                "min_train_size",
                "failed",
                f"train tiene {n_train} registro(s), por debajo del mínimo "
                f"requerido ({min_train_size}) para entrenar el modelo.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "min_train_size",
                "passed",
                f"train tiene {n_train} registro(s) (mínimo requerido: {min_train_size}).",
            )
        )

    if n_test < min_test_size:
        checks.append(
            CheckResult(
                "min_test_size",
                "failed",
                f"test tiene {n_test} registro(s), por debajo del mínimo "
                f"requerido ({min_test_size}) para evaluar el modelo.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "min_test_size",
                "passed",
                f"test tiene {n_test} registro(s) (mínimo requerido: {min_test_size}).",
            )
        )

    ratio = (n_test / n_train) if n_train else float("inf")
    if n_train and (ratio < min_ratio or ratio > max_ratio):
        checks.append(
            CheckResult(
                "test_train_ratio",
                "warning",
                f"La proporción test/train es {ratio:.2f}, fuera del rango "
                f"esperado [{min_ratio}, {max_ratio}].",
            )
        )
    else:
        checks.append(
            CheckResult(
                "test_train_ratio",
                "passed",
                f"La proporción test/train es {ratio:.2f}, dentro del rango esperado.",
            )
        )

    return checks


def _infer_column_groups(
    df: pd.DataFrame,
    numeric_columns: list[str] | None,
    categorical_columns: list[str] | None,
    max_binary_unique: int,
) -> tuple[list[str], list[str]]:
    """Determina qué columnas tratar como numéricas continuas y cuáles como
    categóricas para los checks de distribución.

    Si se pasan explícitamente `numeric_columns`/`categorical_columns` se
    respetan tal cual (para que el pipeline real del proyecto pueda fijar
    las columnas relevantes). Si no, se infieren por tipo de dato: una
    columna numérica con más de `max_binary_unique` valores distintos se
    trata como continua; el resto (no numéricas, o numéricas binarias
    tipo 0/1 de variables dummy/One-Hot) se tratan como categóricas.
    """
    if numeric_columns is not None or categorical_columns is not None:
        return list(numeric_columns or []), list(categorical_columns or [])

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in df.columns:
        serie = df[col]
        if (
            pd.api.types.is_numeric_dtype(serie)
            and serie.nunique(dropna=True) > max_binary_unique
        ):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def _check_numeric_drift(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    numeric_columns: list[str],
    threshold: float,
) -> list[CheckResult]:
    """Check de distribución para variables continuas (equivalente a
    "Feature Drift" numérico), usando el estadístico de
    Kolmogorov-Smirnov de dos muestras.

    Es una advertencia, no un error: una diferencia moderada entre train
    y test es esperable con un split aleatorio y no implica un problema
    por sí sola.
    """
    checks: list[CheckResult] = []
    for col in numeric_columns:
        if col not in x_train.columns or col not in x_test.columns:
            continue
        train_vals = x_train[col].dropna()
        test_vals = x_test[col].dropna()
        if len(train_vals) < 2 or len(test_vals) < 2:
            continue

        statistic, _p_value = ks_2samp(train_vals, test_vals)
        if statistic > threshold:
            checks.append(
                CheckResult(
                    f"distribution_{col}",
                    "warning",
                    f"La distribución de '{col}' difiere entre train y test "
                    f"(estadístico KS={statistic:.2f} > {threshold}).",
                )
            )
        else:
            checks.append(
                CheckResult(
                    f"distribution_{col}",
                    "passed",
                    f"La distribución de '{col}' es comparable entre train y "
                    f"test (estadístico KS={statistic:.2f}).",
                )
            )
    return checks


def _check_categorical_drift(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    categorical_columns: list[str],
    threshold: float,
) -> list[CheckResult]:
    """Check de distribución para variables categóricas/binarias:
    categorías nuevas en test (equivalente a "New Category Train Test")
    y diferencia de proporciones entre categorías (equivalente a
    "Feature Drift" categórico).
    """
    checks: list[CheckResult] = []
    for col in categorical_columns:
        if col not in x_train.columns or col not in x_test.columns:
            continue

        train_props = x_train[col].value_counts(normalize=True, dropna=True)
        test_props = x_test[col].value_counts(normalize=True, dropna=True)

        categorias_nuevas = set(test_props.index) - set(train_props.index)
        if categorias_nuevas:
            checks.append(
                CheckResult(
                    f"new_category_{col}",
                    "warning",
                    f"'{col}' tiene categoría(s) en test no vistas en train: "
                    f"{sorted(categorias_nuevas, key=str)}.",
                )
            )
        else:
            checks.append(
                CheckResult(
                    f"new_category_{col}",
                    "passed",
                    f"'{col}' no tiene categorías nuevas en test respecto a train.",
                )
            )

        todas_las_categorias = set(train_props.index) | set(test_props.index)
        if todas_las_categorias:
            diferencia_maxima = max(
                abs(train_props.get(cat, 0.0) - test_props.get(cat, 0.0))
                for cat in todas_las_categorias
            )
        else:
            diferencia_maxima = 0.0

        if diferencia_maxima > threshold:
            checks.append(
                CheckResult(
                    f"distribution_{col}",
                    "warning",
                    f"La proporción de categorías de '{col}' difiere entre "
                    f"train y test (diferencia máxima={diferencia_maxima:.2f} "
                    f"> {threshold}).",
                )
            )
        else:
            checks.append(
                CheckResult(
                    f"distribution_{col}",
                    "passed",
                    f"La proporción de categorías de '{col}' es comparable "
                    f"entre train y test (diferencia máxima={diferencia_maxima:.2f}).",
                )
            )
    return checks


def _check_target_drift(
    y_train: pd.Series, y_test: pd.Series, threshold: float
) -> CheckResult:
    """Check de distribución del target entre train y test (equivalente a
    "Label Drift"), usando Kolmogorov-Smirnov, ya que `medv` es continuo.
    """
    train_vals = pd.Series(y_train).dropna()
    test_vals = pd.Series(y_test).dropna()
    if len(train_vals) < 2 or len(test_vals) < 2:
        return CheckResult(
            "label_drift",
            "passed",
            "Muy pocos datos disponibles para evaluar el desvío del target.",
        )

    statistic, _p_value = ks_2samp(train_vals, test_vals)
    if statistic > threshold:
        return CheckResult(
            "label_drift",
            "warning",
            f"La distribución del target difiere entre train y test "
            f"(estadístico KS={statistic:.2f} > {threshold}).",
        )
    return CheckResult(
        "label_drift",
        "passed",
        f"La distribución del target es comparable entre train y test "
        f"(estadístico KS={statistic:.2f}).",
    )


def _check_temporal_order(
    x_train: pd.DataFrame, x_test: pd.DataFrame, time_column: str | None
) -> CheckResult | None:
    """Check de orden temporal (SOLO si `time_column` se especifica).

    El proyecto Precios-casas-Boston NO es un problema temporal (no hay
    ninguna columna de fecha/tiempo en el dataset, ver `Informacion.txt`
    y los notebooks 01/02), por lo que `run_split_check_pipeline()` NO
    activa este check para el proyecto real. Se deja disponible como
    capacidad genérica de la función, para proyectos donde sí aplique.
    """
    if time_column is None:
        return None

    if time_column not in x_train.columns or time_column not in x_test.columns:
        return CheckResult(
            "temporal_order",
            "warning",
            f"No se pudo validar el orden temporal: falta la columna "
            f"'{time_column}' en train y/o test.",
        )

    max_train_time = x_train[time_column].max()
    min_test_time = x_test[time_column].min()
    if pd.isna(max_train_time) or pd.isna(min_test_time):
        return CheckResult(
            "temporal_order",
            "warning",
            "No se pudo validar el orden temporal: valores nulos en la columna de tiempo.",
        )

    if max_train_time > min_test_time:
        n_futuro = int((x_train[time_column] > min_test_time).sum())
        return CheckResult(
            "temporal_order",
            "failed",
            f"Se encontró información futura en train: {n_futuro} registro(s) "
            f"de train tienen '{time_column}' posterior al mínimo de test "
            f"(máx. train={max_train_time}, mín. test={min_test_time}).",
        )

    return CheckResult(
        "temporal_order",
        "passed",
        f"El orden temporal se respeta: train termina en {max_train_time}, "
        f"test comienza en {min_test_time}.",
    )


# ---------------------------------------------------------------------------
# Función principal (independiente y testeable, no depende de globales)
# ---------------------------------------------------------------------------


def validate_train_test_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    target_col: str = TARGET_COL,
    id_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    min_test_size: int = DEFAULT_MIN_TEST_SIZE,
    min_test_train_ratio: float = DEFAULT_MIN_TEST_TRAIN_RATIO,
    max_test_train_ratio: float = DEFAULT_MAX_TEST_TRAIN_RATIO,
    drift_warning_threshold: float = DEFAULT_DRIFT_WARNING_THRESHOLD,
    max_binary_unique: int = DEFAULT_MAX_BINARY_UNIQUE,
    time_column: str | None = None,
) -> TrainTestSplitValidationResult:
    """Valida la separación entre `x_train`/`x_test` (y sus targets).

    Función independiente, sin dependencia de variables globales: recibe
    todos los datos y umbrales que necesita como argumentos, por lo que
    se puede probar fácilmente con DataFrames pequeños.

    Checks ejecutados (ver docstring del módulo para la referencia del
    curso en la que se basan):

    1. `overlap_indices` — sin índices compartidos entre train y test.
    2. `duplicate_rows` — sin registros con contenido duplicado entre
       train y test.
    3. `target_not_in_features` / `target_length_*` / `target_no_nulls`
       — el target está presente, completo y no se filtra a las features.
    4. `min_train_size` / `min_test_size` / `test_train_ratio` — tamaños
       suficientes para entrenar y evaluar.
    5. `distribution_<columna numérica>` — Kolmogorov-Smirnov por
       columna numérica continua.
    6. `new_category_<columna categórica>` / `distribution_<columna
       categórica>` — categorías nuevas y diferencia de proporciones por
       columna categórica/binaria.
    7. `label_drift` — Kolmogorov-Smirnov sobre el target.
    8. `temporal_order` — SOLO si se pasa `time_column` (no aplica al
       proyecto Precios-casas-Boston, que no es temporal).

    Comportamiento ante problemas:

    * Problemas críticos (fuga de datos, target inconsistente, conjuntos
      demasiado pequeños, orden temporal violado) -> se recolectan todos
      y se lanza `TrainTestSplitValidationError` con el detalle completo.
    * Problemas de distribución (drift) -> se emiten como advertencia
      (`warnings.warn`) y quedan registrados con estado "warning" en el
      resultado, pero NO detienen la ejecución.

    Retorna un `TrainTestSplitValidationResult` con el detalle de todos
    los checks ejecutados (no solo un booleano) cuando no hay problemas
    críticos.
    """
    checks: list[CheckResult] = []

    checks.append(_check_index_overlap(x_train, x_test))
    checks.append(_check_duplicate_rows(x_train, x_test, id_columns))
    checks.extend(_check_target_presence(x_train, x_test, y_train, y_test, target_col))
    checks.extend(
        _check_sizes(
            x_train,
            x_test,
            min_train_size,
            min_test_size,
            min_test_train_ratio,
            max_test_train_ratio,
        )
    )

    numeric_cols, categorical_cols = _infer_column_groups(
        x_train, numeric_columns, categorical_columns, max_binary_unique
    )
    checks.extend(
        _check_numeric_drift(x_train, x_test, numeric_cols, drift_warning_threshold)
    )
    checks.extend(
        _check_categorical_drift(
            x_train, x_test, categorical_cols, drift_warning_threshold
        )
    )
    checks.append(_check_target_drift(y_train, y_test, drift_warning_threshold))

    temporal_check = _check_temporal_order(x_train, x_test, time_column)
    if temporal_check is not None:
        checks.append(temporal_check)

    result = TrainTestSplitValidationResult(checks=checks)

    for check in result.warning_checks():
        warnings.warn(f"[{check.name}] {check.message}", UserWarning, stacklevel=2)

    if not result.passed:
        mensajes_fallidos = [f"[{c.name}] {c.message}" for c in result.failed_checks()]
        raise TrainTestSplitValidationError(
            "Falló la validación del split train/test. Problemas críticos "
            "detectados:\n- " + "\n- ".join(mensajes_fallidos),
            checks=checks,
        )

    return result


def run_split_check_pipeline(
    x_train_path: Path = X_TRAIN_PATH,
    x_test_path: Path = X_TEST_PATH,
    y_train_path: Path = Y_TRAIN_PATH,
    y_test_path: Path = Y_TEST_PATH,
    results_path: Path = RESULTS_PATH,
    **validate_kwargs: object,
) -> TrainTestSplitValidationResult:
    """Ejecuta el check completo: lee el split ya persistido y lo valida.

    Reutiliza `train_pipeline.load_processed_features` (Issue 3) para
    leer exactamente los mismos archivos que consume el entrenamiento;
    esta Issue no vuelve a leer los datos originales ni a repetir el
    split.

    Guarda el resultado (pase o falle la validación) en `results_path`,
    para dejar registro de qué se comprobó incluso cuando se detecta un
    problema crítico.
    """
    x_train, x_test, y_train, y_test = load_processed_features(
        x_train_path, x_test_path, y_train_path, y_test_path
    )

    try:
        result = validate_train_test_split(
            x_train, x_test, y_train, y_test, **validate_kwargs
        )
    except TrainTestSplitValidationError as exc:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        payload = TrainTestSplitValidationResult(checks=exc.checks).to_dict()
        with results_path.open("w", encoding="utf-8") as results_file:
            json.dump(payload, results_file, indent=2, ensure_ascii=False)
        raise

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as results_file:
        json.dump(result.to_dict(), results_file, indent=2, ensure_ascii=False)

    return result


def main() -> None:
    result = run_split_check_pipeline()
    print(result.summary())
    print(f"Resultados guardados en: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
