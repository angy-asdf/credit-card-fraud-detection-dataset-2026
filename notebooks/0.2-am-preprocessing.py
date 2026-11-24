import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pandas as pd

    return pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0.2 Preprocesamiento — split 80/20, estandarización y PCA

    Continuación de `0.1-am-initial-exploration`: pasos 3 y 4 del proyecto (ver `references/Proyecto_final__MCD.pdf`) sobre el subconjunto México.
    """)
    return


@app.cell
def _():
    from credit_card_fraud_detection_dataset_2026.config import INTERIM_DATA_DIR

    INTERIM_CSV = INTERIM_DATA_DIR / "mexico_numeric.csv"
    return (INTERIM_CSV,)


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
    return (X_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paso 4.1: Estandarización

    PCA busca las direcciones de **máxima varianza** de la matriz de covarianzas. Si las columnas tienen unidades muy distintas (`credit_limit` en miles de dólares vs `used_pin` que vale 0/1), la de mayor escala domina las componentes aunque no sea la más informativa — el PCA mediría unidades, no patrones.

    `StandardScaler` deja cada columna con media 0 y varianza 1, así todas aportan por igual y el PCA refleja las correlaciones reales entre variables.

    `is_fraud` se excluye del escalado: es la etiqueta, no una feature — va *junto con* la nueva codificación, no dentro de ella.
    """)
    return


@app.cell
def _(X_train):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    feature_cols = [c for c in X_train.columns if c != "is_fraud"]
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    X_train_scaled
    return (X_train_scaled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paso 4.2: PCA (10 componentes)

    Las 10 componentes principales son las 10 direcciones ortogonales de máxima varianza del espacio estandarizado — una nueva codificación de cada transacción en 10 números. `is_fraud` se conserva junto a esa codificación como etiqueta.
    """)
    return


@app.cell
def _(X_train, X_train_scaled, pd):
    from sklearn.decomposition import PCA

    # Nos quedamos con las 10 primeras, estan ordenadas de mayor a menor
    pca = PCA(n_components=10)  
    X_train_pca = pca.fit_transform(X_train_scaled)

    pc_cols = [f"PC{i + 1}" for i in range(10)]
    X_train_pca = pd.DataFrame(X_train_pca, columns=pc_cols, index=X_train.index)
    X_train_pca["is_fraud"] = X_train["is_fraud"]  # la etiqueta va "junto con" el PCA
    X_train_pca
    return


if __name__ == "__main__":
    app.run()
