"""Feature pipeline para el proyecto Precios-casas-Boston.

Convierte en un script reproducible la lógica de Feature Engineering
desarrollada en el notebook `06.feature_engineering-mkmh-2026-08-27.ipynb`
(Issue 4): limpieza, feature selection, feature engineering, escalado y
encoding, usando los mismos transformadores de scikit-learn definidos ahí.

Además (Issue: Data validation y data integrity), el pipeline valida los
datos de entrada y las features ya procesadas ANTES de persistirlas. Las
reglas de validación se implementan con pandas puro (sin dependencias
nuevas) para mantener la solución simple. Si alguna validación falla, se
lanza `DataValidationError` con un mensaje que detalla qué regla(s)
fallaron y NO se escribe ningún archivo de features.

Flujo:
    leer datos -> transformar -> validar -> (si OK) persistir
                                          -> (si falla) error, sin persistir

Uso:
    python pipelines/feature_pipeline_validado.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

# Raíz del proyecto: este archivo vive en <root>/pipelines/feature_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_PATH = DATA_DIR / "02_intermediate" / "boston_type_fixed.parquet"
OUTPUT_DIR = DATA_DIR / "03_primary"

# Columnas seleccionadas al inicio del notebook de feature engineering
SELECTED_FEATURES = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "black",
    "lstat",
    "medv",
]

TARGET_COL = "medv"
MEDV_CENSORED_VALUE = 50.0
RAD_HIGH_VALUE = 24
REDUNDANT_COLUMNS = ["tax"]

# --- Reglas de validación de datos (Issue: Data validation y data integrity) ---
# Los rangos y categorías se toman de los propios notebooks del proyecto
# (docstrings de `Informacion.txt`, notebook 02 de exploración inicial y
# notebook 06 de feature engineering), no son límites inventados.

# Columnas numéricas que debe tener el dataset ya limpio/enriquecido, antes
# de separar en train/test.
EXPECTED_NUMERIC_COLUMNS = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "ptratio",
    "black",
    "lstat",
]

# Rangos razonables por columna: (mínimo, máximo).
# - nox, rm, dis, crim: los mismos límites usados en el notebook 02 para
#   detectar y corregir columnas con error de escala (ej. "nox debería
#   estar entre 0 y ~1", "rm entre ~3 y ~9", "dis entre ~1 y ~13", "crim
#   real llega hasta ~74 en este dataset").
# - zn, indus, age, lstat: son proporciones/porcentajes -> [0, 100].
# - black: por definición 1000*(Bk-0.63)^2 con Bk en [0, 1], cuyo máximo
#   real observado en el dataset es 396.9.
# - ptratio: una razón alumno/profesor no puede ser negativa ni
#   absurdamente alta.
# - medv (target): documentado en `Informacion.txt` como valor en miles de
#   USD; el propio EDA identifica 50.0 como valor tope (posible censura).
VALID_RANGES: dict[str, tuple[float, float]] = {
    "crim": (0.0, 100.0),
    "zn": (0.0, 100.0),
    "indus": (0.0, 100.0),
    "nox": (0.0, 1.0),
    "rm": (3.0, 9.0),
    "age": (0.0, 100.0),
    "dis": (1.0, 13.0),
    "ptratio": (0.0, 40.0),
    "black": (0.0, 400.0),
    "lstat": (0.0, 100.0),
    "medv": (0.0, 50.0),
}

# Porcentaje máximo de nulos permitido por columna. Se fija en 5%: en los
# datos reales del proyecto el máximo observado tras la limpieza es
# ~1.2% (columnas `crim`/`indus`), así que 5% da margen sin dejar pasar
# una degradación seria en la calidad de un nuevo batch de datos.
MAX_NULL_FRACTION = 0.05

# Categorías válidas por columna categórica.
VALID_CATEGORIES: dict[str, set] = {
    "chas": {0, 1},
    "medv_censored": {0, 1},
    "rad_group": {"alto", "bajo"},
    # `rad` es el índice de accesibilidad a autopistas radiales: en el
    # dataset de Boston es una variable discreta con un conjunto fijo de
    # categorías (1-8 y 24), confirmado en los datos reales del proyecto.
    "rad": {1, 2, 3, 4, 5, 6, 7, 8, 24},
}


class DataValidationError(Exception):
    """Error de validación de datos/integridad del feature pipeline.

    Se lanza cuando los datos de entrada o las features procesadas no
    cumplen las reglas de calidad, consistencia, formato o integridad
    definidas para este proyecto. El mensaje detalla cada regla que
    falló para que el error sea accionable.
    """


def load_intermediate_data(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Lee el dataset intermedio (salida del notebook 02) en formato parquet."""
    return pd.read_parquet(input_path, engine="pyarrow")


def select_initial_columns(
    df: pd.DataFrame, columns: list[str] = SELECTED_FEATURES
) -> pd.DataFrame:
    """Selecciona las columnas relevantes para el modelamiento."""
    return df[columns].copy()


def clean_data(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Elimina filas duplicadas y filas sin valor en la variable objetivo.

    La variable objetivo no puede imputarse, por lo que las filas sin
    `medv` se descartan (no aportan al entrenamiento supervisado).
    """
    df_clean = df.drop_duplicates()
    df_clean = df_clean.dropna(subset=[target_col])
    return df_clean


def flag_censored_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    censored_value: float = MEDV_CENSORED_VALUE,
) -> pd.DataFrame:
    """Agrega la columna indicadora `medv_censored`.

    Marca las observaciones cuyo valor de `medv` coincide con el valor
    posiblemente censurado (detectado en el EDA), sin eliminarlas.
    """
    df_flagged = df.copy()
    df_flagged["medv_censored"] = (df_flagged[target_col] == censored_value).astype(int)
    return df_flagged


def drop_redundant_features(
    df: pd.DataFrame, columns_to_drop: list[str] = REDUNDANT_COLUMNS
) -> pd.DataFrame:
    """Elimina columnas redundantes por alta colinealidad (Feature Selection).

    `tax` se descarta por su alta correlación con `rad` (0.91 en el
    análisis bivariable), conservando `rad`.
    """
    return df.drop(columns=columns_to_drop)


def add_rad_group_feature(df: pd.DataFrame, high_value: int = RAD_HIGH_VALUE) -> pd.DataFrame:
    """Crea el atributo derivado `rad_group` (Feature Engineering).

    Captura de forma explícita el grupo `rad = 24`, que se comporta de
    forma distinta al resto según el análisis bivariable/multivariable.
    """
    df_engineered = df.copy()
    df_engineered["rad_group"] = np.where(df_engineered["rad"] == high_value, "alto", "bajo")
    return df_engineered


def _validate_types(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> list[str]:
    """Valida que las columnas esperadas sean numéricas."""
    errors: list[str] = []
    for col in numeric_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(
                f"[tipo] La columna '{col}' debería ser numérica, "
                f"pero tiene tipo '{df[col].dtype}'."
            )
    return errors


def _validate_ranges(
    df: pd.DataFrame,
    valid_ranges: dict[str, tuple[float, float]],
) -> list[str]:
    """Valida que los valores numéricos estén dentro de los rangos permitidos."""
    errors: list[str] = []
    for col, (low, high) in valid_ranges.items():
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col].dropna()
        out_of_range = values[(values < low) | (values > high)]
        if not out_of_range.empty:
            errors.append(
                f"[rango] La columna '{col}' tiene {len(out_of_range)} "
                f"valor(es) fuera del rango permitido [{low}, {high}] "
                f"(ej. valor observado: {out_of_range.iloc[0]})."
            )
    return errors


def _validate_nulls(
    df: pd.DataFrame,
    numeric_columns: list[str],
    max_null_fraction: float,
) -> list[str]:
    """Valida el porcentaje máximo permitido de valores nulos."""
    errors: list[str] = []
    for col in numeric_columns:
        if col not in df.columns:
            continue
        null_fraction = df[col].isna().mean()
        if null_fraction > max_null_fraction:
            errors.append(
                f"[nulos] La columna '{col}' tiene {null_fraction:.1%} "
                f"de valores nulos, por encima del máximo permitido "
                f"({max_null_fraction:.0%})."
            )
    return errors


def _validate_categories(
    df: pd.DataFrame,
    valid_categories: dict[str, set],
) -> list[str]:
    """Valida las categorías permitidas para las variables categóricas."""
    errors: list[str] = []
    for col, allowed_values in valid_categories.items():
        if col not in df.columns:
            continue
        observed_values = set(df[col].dropna().unique())
        invalid_values = observed_values - allowed_values
        if invalid_values:
            errors.append(
                f"[categoría] La columna '{col}' contiene categorías "
                f"inválidas {sorted(invalid_values, key=str)}; valores "
                f"permitidos: {sorted(allowed_values, key=str)}."
            )
    return errors


def _validate_uniqueness(df: pd.DataFrame) -> list[str]:
    """Valida que no existan filas completamente duplicadas."""
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates == 0:
        return []
    return [
        (
            f"[unicidad] Se encontraron {n_duplicates} fila(s) completamente "
            "duplicadas; se esperaba que ya estuvieran eliminadas."
        )
    ]


def _validate_field_integrity(df: pd.DataFrame) -> list[str]:
    """Valida la consistencia entre variables derivadas y sus variables base."""
    errors: list[str] = []

    if {"rad", "rad_group"}.issubset(df.columns):
        expected_group = np.where(
            df["rad"] == RAD_HIGH_VALUE,
            "alto",
            "bajo",
        )
        mismatched = int((df["rad_group"].to_numpy() != expected_group).sum())
        if mismatched > 0:
            errors.append(
                f"[integridad] 'rad_group' es inconsistente con 'rad' en "
                f"{mismatched} fila(s): debería ser 'alto' únicamente cuando "
                f"rad == {RAD_HIGH_VALUE}."
            )

    if {"medv", "medv_censored"}.issubset(df.columns):
        expected_flag = (df["medv"] == MEDV_CENSORED_VALUE).astype(int)
        mismatched = int((df["medv_censored"] != expected_flag).sum())
        if mismatched > 0:
            errors.append(
                f"[integridad] 'medv_censored' es inconsistente con 'medv' "
                f"en {mismatched} fila(s): debería ser 1 únicamente cuando "
                f"medv == {MEDV_CENSORED_VALUE}."
            )

    return errors


def validate_input_data(
    df: pd.DataFrame,
    numeric_columns: list[str] = EXPECTED_NUMERIC_COLUMNS,
    valid_ranges: dict[str, tuple[float, float]] = VALID_RANGES,
    valid_categories: dict[str, set] = VALID_CATEGORIES,
    max_null_fraction: float = MAX_NULL_FRACTION,
) -> None:
    """Valida calidad, consistencia, formato e integridad de los datos.

    Las fechas no se validan porque el dataset no contiene columnas de fecha.
    """
    errors: list[str] = []

    errors.extend(_validate_types(df, numeric_columns))
    errors.extend(_validate_ranges(df, valid_ranges))
    errors.extend(_validate_nulls(df, numeric_columns, max_null_fraction))
    errors.extend(_validate_categories(df, valid_categories))
    errors.extend(_validate_uniqueness(df))
    errors.extend(_validate_field_integrity(df))

    if errors:
        raise DataValidationError(
            "Falló la validación de los datos de entrada. No se generarán "
            "features. Reglas incumplidas:\n- " + "\n- ".join(errors)
        )


def validate_processed_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Valida integridad y estructura de las features ya transformadas.

    Se ejecuta justo antes de persistir, sobre la salida del
    `ColumnTransformer`. Cubre:

    - Formato/estructura: `x_train` y `x_test` deben tener exactamente las
      mismas columnas (mismo preprocesador ya ajustado).
    - Calidad: ninguna columna debe quedar con nulos tras imputar/escalar,
      y todas deben ser numéricas (requisito para entrenar el modelo).
    - Integridad entre datasets: `X` e `y` deben tener el mismo número de
      filas, y no debe haber índices compartidos entre train y test
      (evita fuga de datos / data leakage entre los dos conjuntos).

    Lanza `DataValidationError` si alguna condición no se cumple.
    """
    errors: list[str] = []

    if list(x_train.columns) != list(x_test.columns):
        errors.append(
            "[formato] 'x_train' y 'x_test' no tienen las mismas columnas tras la transformación."
        )

    train_nulls = int(x_train.isna().sum().sum())
    if train_nulls > 0:
        errors.append(
            f"[calidad] 'x_train' tiene {train_nulls} valor(es) nulo(s) tras la "
            "transformación; se esperaba 0 (el ColumnTransformer imputa todo)."
        )

    test_nulls = int(x_test.isna().sum().sum())
    if test_nulls > 0:
        errors.append(
            f"[calidad] 'x_test' tiene {test_nulls} valor(es) nulo(s) tras la "
            "transformación; se esperaba 0."
        )

    non_numeric_cols = x_train.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_cols:
        errors.append(
            f"[tipo] 'x_train' contiene columnas no numéricas tras la "
            f"transformación: {non_numeric_cols}."
        )

    if len(x_train) != len(y_train):
        errors.append(
            f"[integridad] 'x_train' ({len(x_train)} filas) y 'y_train' "
            f"({len(y_train)} filas) no tienen el mismo número de filas."
        )

    if len(x_test) != len(y_test):
        errors.append(
            f"[integridad] 'x_test' ({len(x_test)} filas) y 'y_test' "
            f"({len(y_test)} filas) no tienen el mismo número de filas."
        )

    overlapping_indices = set(x_train.index) & set(x_test.index)
    if overlapping_indices:
        errors.append(
            f"[integridad] Se encontraron {len(overlapping_indices)} registro(s) "
            "compartidos entre train y test (posible fuga de datos / data leakage)."
        )

    if errors:
        raise DataValidationError(
            "Falló la validación de las features procesadas. No se "
            "persistirán archivos. Reglas incumplidas:\n- " + "\n- ".join(errors)
        )


def build_preprocessor() -> ColumnTransformer:
    """Construye el `ColumnTransformer` de escalado/encoding del notebook 06.

    Grupos de columnas:
    - log: transformación log1p + escalado (variable muy asimétrica).
    - numeric: variables continuas, solo imputación + escalado.
    - discretize: `age` discretizada en 4 categorías (atributo adicional).
    - binary: variables binarias, solo imputación.
    - nominal: variables categóricas nominales, One-Hot Encoding.
    """
    cols_log = ["crim"]
    cols_numeric = ["zn", "indus", "nox", "rm", "age", "dis", "ptratio", "black", "lstat"]
    cols_discretize = ["age"]
    cols_binary = ["chas", "medv_censored"]
    cols_nominal = ["rad", "rad_group"]

    log_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    discretize_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("discretizer", KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")),
        ]
    )

    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    nominal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("log", log_pipe, cols_log),
            ("numeric", numeric_pipe, cols_numeric),
            ("discretize", discretize_pipe, cols_discretize),
            ("binary", binary_pipe, cols_binary),
            ("nominal", nominal_pipe, cols_nominal),
        ]
    )


def split_train_test(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa features/target y hace el split train/test estratificado por `chas`."""
    x_features = df.drop(columns=[target_col])
    y_target = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x_features,
        y_target,
        test_size=test_size,
        stratify=x_features["chas"],
        random_state=random_state,
    )
    return x_train, x_test, y_train, y_test


def fit_transform_features(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajusta el preprocesador en train y transforma train y test.

    El preprocesador solo se ajusta (`fit`) sobre `x_train`; `x_test` se
    transforma con `.transform()` para evitar fuga de información.
    """
    preprocessor.fit(x_train)
    feature_names = preprocessor.get_feature_names_out()

    x_train_transformed = pd.DataFrame(
        preprocessor.transform(x_train), columns=feature_names, index=x_train.index
    )
    x_test_transformed = pd.DataFrame(
        preprocessor.transform(x_test), columns=feature_names, index=x_test.index
    )
    return x_train_transformed, x_test_transformed


def save_processed_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Guarda las features procesadas y los targets en formato parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train.to_parquet(output_dir / "x_train_features.parquet", engine="pyarrow")
    x_test.to_parquet(output_dir / "x_test_features.parquet", engine="pyarrow")
    y_train.to_frame().to_parquet(output_dir / "y_train.parquet", engine="pyarrow")
    y_test.to_frame().to_parquet(output_dir / "y_test.parquet", engine="pyarrow")


def run_feature_pipeline(
    input_path: Path = INPUT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Ejecuta el feature pipeline completo: datos originales -> features procesadas.

    Retorna un diccionario con los conjuntos resultantes, útil tanto para
    pruebas unitarias como para inspección interactiva.
    """
    boston_df = load_intermediate_data(input_path)

    boston_features = select_initial_columns(boston_df)
    boston_features = clean_data(boston_features)
    boston_features = flag_censored_target(boston_features)
    boston_features = drop_redundant_features(boston_features)
    boston_features = add_rad_group_feature(boston_features)

    # Validar datos de entrada ANTES de transformar/persistir. Si falla,
    # `DataValidationError` interrumpe la ejecución aquí y no se llega a
    # generar ni guardar ningún archivo de features.
    validate_input_data(boston_features)

    x_train, x_test, y_train, y_test = split_train_test(boston_features)

    preprocessor = build_preprocessor()
    x_train_transformed, x_test_transformed = fit_transform_features(preprocessor, x_train, x_test)

    # Validar las features ya procesadas ANTES de persistirlas.
    validate_processed_features(x_train_transformed, x_test_transformed, y_train, y_test)

    save_processed_features(x_train_transformed, x_test_transformed, y_train, y_test, output_dir)

    return {
        "x_train": x_train_transformed,
        "x_test": x_test_transformed,
        "y_train": y_train,
        "y_test": y_test,
    }


def main() -> None:
    results = run_feature_pipeline()
    print(f"Datos originales leídos desde: {INPUT_PATH}")
    print(f"Features procesadas guardadas en: {OUTPUT_DIR}")
    print(f"x_train shape: {results['x_train'].shape}")
    print(f"x_test shape: {results['x_test'].shape}")


if __name__ == "__main__":
    main()
