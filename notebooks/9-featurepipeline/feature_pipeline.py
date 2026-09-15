"""Feature pipeline para el proyecto Precios-casas-Boston.

Convierte en un script reproducible la lógica de Feature Engineering
desarrollada en el notebook `06.feature_engineering-mkmh-2026-08-27.ipynb`
(Issue 4): limpieza, feature selection, feature engineering, escalado y
encoding, usando los mismos transformadores de scikit-learn definidos ahí.

Uso:
    python pipelines/feature_pipeline.py
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
    cols_numeric = [
        "zn",
        "indus",
        "nox",
        "rm",
        "age",
        "dis",
        "ptratio",
        "black",
        "lstat",
    ]
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
            (
                "discretizer",
                KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile"),
            ),
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

    x_train, x_test, y_train, y_test = split_train_test(boston_features)

    preprocessor = build_preprocessor()
    x_train_transformed, x_test_transformed = fit_transform_features(preprocessor, x_train, x_test)

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
