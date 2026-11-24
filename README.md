# credit-card-fraud-detection-dataset-2026

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Deteccion de fraude financiero mediante el kernel de Christoffel-Darboux

## Fuente de datos

**[Credit Card Fraud Detection Dataset 2026](https://www.kaggle.com/datasets/mohankrishnathalla/credit-card-fraud-detection-dataset-2026)** — Mohan Krishna Thalla (Kaggle), licencia CC0: Public Domain.

- 100,000 transacciones sintéticas de tarjeta de crédito, calendario 2024 completo
- 43 features con nombres reales: detalles de transacción, perfil de cliente, señales de velocidad, indicadores de riesgo, dispositivo/canal y comportamiento de cuenta
- Calibrado contra investigación publicada (Federal Reserve, Nilson Report, FICO, Banco Central Europeo); tasa de fraude global ~2.06%
- 3 targets: `is_fraud` (binario), `fraud_type` (5 clases), `fraud_amount_lost` (regresión)

**Descarga:** `python credit_card_fraud_detection_dataset_2026/dataset.py` (o `make data`) usa `kagglehub` y requiere credenciales en `~/.kaggle/kaggle.json`. El CSV se copia a `data/raw/credit_card_fraud_detection_2026.csv` (inmutable; `data/` está en `.gitignore`).

## Diccionario de datos

`dataset.py` genera `references/data_dictionary.csv` y `references/data_dictionary.md` en cada corrida del pipeline. La información de columnas se recupera de la fuente misma: vía la API de Kaggle se toma la tabla **Feature Groups** (grupo temático de cada columna) y las descripciones de **Target Variables**, y se combina con un perfilado del CSV real.

Columnas del diccionario:

| Columna | Significado |
|---|---|
| `column` | Nombre de la columna en el dataset |
| `group` | Grupo temático asignado por el autor (Transaction Identifiers, Transaction Details, Customer Profile, Velocity Features, Risk Signals, Device and Channel, Account Behaviour, Targets) |
| `dtype` | Tipo de dato inferido por pandas (`int64`, `float64`, `object`) |
| `description` | Texto de la fuente: descripción oficial para las 3 targets; para el resto, su grupo (la fuente no provee descripciones individuales) |
| `non_null` | Filas con valor no nulo en esa columna |
| `n_unique` | Cantidad de valores distintos |
| `top_values` | Los 3 valores más frecuentes — solo columnas **no numéricas** (p. ej. el país o categoría que más aparece) |
| `min`, `max`, `mean` | Mínimo, máximo y promedio — solo columnas **numéricas** |

## Libretas (marimo)

Las libretas del proyecto usan **[marimo](https://marimo.io)** en lugar de Jupyter: se guardan como `.py` plano, así que los commits solo contienen código — los outputs nunca se escriben al archivo (se regeneran al abrirla).

**Abrir una libreta** (con el venv activado):

```bash
marimo edit notebooks/0.1-am-initial-exploration.py
```

Abre el editor en `http://localhost:2718` — se usa como Jupyter: celdas de código y markdown, outputs en vivo.

Notas:

- Cada variable se define **una sola vez** en toda la libreta (ejecución reactiva); si reasignas un nombre, marimo te avisa.
- Una libreta también corre como script: `python notebooks/0.1-am-initial-exploration.py`.
- Opcional: extensión oficial de VS Code `marimo-team.vscode-marimo` para ver la libreta dentro del editor.
- No requiere kernelspec de Jupyter — marimo usa el entorno desde donde se lanza.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         credit_card_fraud_detection_dataset_2026 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── credit_card_fraud_detection_dataset_2026   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes credit_card_fraud_detection_dataset_2026 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

