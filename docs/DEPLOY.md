# Despliegue en Streamlit Community Cloud

La app pública vive en https://precios-casas-boston.streamlit.app/

## Qué se despliega

- **Entrypoint:** `demo_app.py` (raíz del repo) — usa el pipeline de
  inferencia real (`src/pipelines/inference_pipeline.py`) con los
  artefactos versionados:
  - modelo: `data/06_models/train_pipeline_model.joblib`
  - preprocesador: `data/03_primary/preprocessor.joblib`

## Configuración en el dashboard (share.streamlit.io)

| Campo | Valor |
|---|---|
| Repository | `MaricelMartinezH/Precios-casas-Boston` |
| Branch | `main` |
| Main file path | `demo_app.py` |
| Python version | 3.12 |

Streamlit Cloud instala dependencias desde `requirements.txt`
(los pins se copian del `uv.lock`).

## Pasos para (re)desplegar o actualizar

1. **Fusionar a `main`** los PRs con el código y el `requirements.txt`.
   Streamlit Cloud **no despliega ramas feature**, solo la branch elegida.
2. En https://share.streamlit.io/ abre la app → **⋮ menu → Settings**:
   - Si la app apuntaba a otro entrypoint (`project/notebooks/mkmh/08_deploy/app.py`),
     cambiar **Main file path** a `demo_app.py` y guardar → fuerza reboot.
   - Si algo quedó cacheado: **⋮ → Reboot app**.
3. **Hacerla pública:** ⋮ → **Settings → Sharing → "Share app publicly"**
   (si está en modo privado, los visitantes ven una redirección a login).
4. Verificar: abrir https://precios-casas-boston.streamlit.app/ en una
   ventana de incógnito (sin tu sesión) y comprobar que carga el formulario
   y que "Predecir precio" devuelve un valor (~24 con los valores por defecto).

## Actualizar dependencias

`requirements.txt` es un subconjunto de `uv.lock`. Tras cambiar
`pyproject.toml` y correr `uv lock`, actualiza los pins de
`requirements.txt` para que no diverjan.
