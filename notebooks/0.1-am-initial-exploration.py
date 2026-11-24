import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0.1 Exploración inicial — Credit Card Fraud Detection Dataset 2026

    EDA inicial del dataset de [Kaggle](https://www.kaggle.com/datasets/mohankrishnathalla/credit-card-fraud-detection-dataset-2026): 100,000 transacciones x 43 features.

    Proyecto: detección de fraude financiero mediante el kernel de Christoffel-Darboux (ver `references/Proyecto_final__MCD.pdf`).
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd

    return pd, plt


@app.cell
def _():
    from credit_card_fraud_detection_dataset_2026.config import (
        INTERIM_DATA_DIR,
        PROJ_ROOT,
        RAW_DATA_DIR,
    )

    DATA_DICT_PATH = PROJ_ROOT / "references" / "data_dictionary.csv"
    RAW_CSV = RAW_DATA_DIR / "credit_card_fraud_detection_2026.csv"
    INTERIM_CSV = INTERIM_DATA_DIR / "mexico_numeric.csv"
    return DATA_DICT_PATH, INTERIM_CSV, RAW_CSV


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diccionario de datos

    Generado por `credit_card_fraud_detection_dataset_2026/dataset.py` a partir de la metadata del dataset en Kaggle más el perfilado de los datos (`references/data_dictionary.csv`).
    """)
    return


@app.cell
def _(DATA_DICT_PATH, pd):
    data_dictionary = pd.read_csv(DATA_DICT_PATH)
    data_dictionary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Carga del dataset (raw)
    """)
    return


@app.cell
def _(RAW_CSV, pd):
    df = pd.read_csv(RAW_CSV, parse_dates=["transaction_datetime"])
    df.head()
    return (df,)


@app.cell
def _(df):
    print(f"Shape: {df.shape}  ({df.shape[0]:,} transacciones x {df.shape[1]} columnas)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `df.info()`
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `df.describe()`
    """)
    return


@app.cell
def _(df):
    df.describe(include="all").T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matriz de correlación

    Correlación de Pearson entre columnas numéricas (covarianza estandarizada, valores en $[-1, 1]$). La matriz de covarianzas $\Sigma$ sobre las transacciones legítimas es la que usará después la distancia de Mahalanobis.
    """)
    return


@app.cell
def _(df):
    _numeric_df = df.select_dtypes(include="number")
    corr_matrix = _numeric_df.corr()
    corr_matrix
    return (corr_matrix,)


@app.cell
def _(corr_matrix, plt):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns,
        rotation=90,
        fontsize=7,
    )
    ax.set_yticks(range(len(corr_matrix.columns)), corr_matrix.columns, fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Matriz de correlación (columnas numéricas)")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Datos interim: México (columnas numéricas + `is_fraud`)

    Generado por `credit_card_fraud_detection_dataset_2026/features.py`: filtra `merchant_country == "Mexico"` del raw y conserva solo las columnas numéricas (que ya incluyen `is_fraud`, tipo `int64`) — pasos 1 y 2 del proyecto.
    """)
    return


@app.cell
def _(INTERIM_CSV, pd):
    df_mx = pd.read_csv(INTERIM_CSV)
    print(
        f"{len(df_mx):,} transacciones México | "
        f"{df_mx['is_fraud'].sum()} fraudes ({df_mx['is_fraud'].mean():.2%})"
    )
    df_mx
    return (df_mx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDA del subconjunto México
    """)
    return


@app.cell
def _(df_mx):
    df_mx.columns.tolist()
    return


@app.cell
def _(df_mx):
    df_mx.info()
    return


@app.cell
def _(df_mx):
    df_mx.describe().T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Separación 80/20 — `X_train` / `X_test`

    Paso 3 del proyecto: 20% de los renglones para pruebas finales (`X_test`), 80% para entrenamiento (`X_train`). `random_state=42` para reproducibilidad.

    Nota: el PDF pide un split simple; si se quisiera preservar la proporción de fraude en ambos conjuntos, se puede usar `stratify=df_mx["is_fraud"]`.
    """)
    return


@app.cell
def _(df_mx):
    from sklearn.model_selection import train_test_split

    X_train, X_test = train_test_split(df_mx, test_size=0.2, random_state=42)
    print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
    print(
        f"Fraude en train: {X_train['is_fraud'].mean():.2%} | "
        f"en test: {X_test['is_fraud'].mean():.2%}"
    )
    return


if __name__ == "__main__":
    app.run()
