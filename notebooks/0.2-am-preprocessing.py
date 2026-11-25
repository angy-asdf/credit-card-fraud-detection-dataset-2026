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
    return X_train, X_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paso 4.1: Estandarización

    PCA busca las direcciones de **máxima varianza** de la matriz de covarianzas. Si las columnas tienen unidades muy distintas (`credit_limit` en miles de dólares vs `used_pin` que vale 0/1), la de mayor escala domina las componentes aunque no sea la más informativa — el PCA mediría unidades, no patrones.

    `StandardScaler` deja cada columna con media 0 y varianza 1, así todas aportan por igual y el PCA refleja las correlaciones reales entre variables.

    `is_fraud` se excluye del escalado: es la etiqueta, no una feature — va *junto con* la nueva codificación, no dentro de ella.

    Fuentes (justificación algebraica):

    - [scikit-learn: Importance of Feature Scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html) — muestra cómo las entradas de la matriz de covarianzas dependen de las unidades de cada variable.
    - [stats.stackexchange: PCA on correlation or covariance](https://stats.stackexchange.com/questions/53) — derivación de por qué estandarizar equivale a aplicar PCA sobre la matriz de correlaciones $R$ en vez de la de covarianzas $\Sigma$.
    """)
    return


@app.cell
def _(X_train):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    feature_cols = [c for c in X_train.columns if c != "is_fraud"]
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    X_train_scaled
    return X_train_scaled, feature_cols, scaler


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
    return PCA, X_train_pca, pca, pc_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scree plot — ¿por qué 10 componentes?

    Varianza explicada por cada una de las 27 componentes (se ajusta un PCA completo solo para visualizar; el `pca` del paso 4.2 no cambia). Las componentes salen ordenadas de mayor a menor varianza — la línea punteada marca el corte del proyecto en PC10.

    El **codo** es el punto donde la curva deja de caer rápido — a partir de ahí, cada componente extra aporta poco. Se detecta numéricamente como el punto más alejado de la recta que une los extremos de la curva.
    """)
    return


@app.cell
def _(PCA, X_train_scaled, plt):
    import numpy as np

    pca_full = PCA().fit(X_train_scaled)  # las 27, solo para graficar
    ratio = pca_full.explained_variance_ratio_
    n_pc = len(ratio)

    # Codo: punto más alejado de la recta que une el primer y último punto
    _x = np.arange(1, n_pc + 1) / n_pc
    _y = ratio / ratio[0]
    _dist = np.abs((_y[-1] - _y[0]) * _x - (_x[-1] - _x[0]) * _y) / np.sqrt(
        (_y[-1] - _y[0]) ** 2 + (_x[-1] - _x[0]) ** 2
    )
    elbow = int(np.argmax(_dist)) + 1

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.bar(range(1, n_pc + 1), ratio, label="Varianza por PC")
    _ax.plot(range(1, n_pc + 1), ratio.cumsum(), "o-", color="red", label="Acumulada")
    _ax.axvline(10.5, ls="--", color="gray", label="Corte del proyecto (10)")
    _ax.axvline(elbow, ls=":", color="green", label=f"Codo (PC{elbow})")
    _ax.set_xlabel("Componente principal")
    _ax.set_ylabel("Fracción de varianza explicada")
    _ax.legend()
    print(f"Codo en PC{elbow} | 10 PCs acumulan {ratio[:10].sum():.1%} de la varianza")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transacciones proyectadas en PC1 vs PC2

    Cada punto es una transacción de `X_train` vista en las dos primeras componentes (las de mayor varianza). Si los fraudes se separan de las legítimas en este plano, es buena señal de que la geometría de los pasos siguientes (Mahalanobis, kernel CD) puede distinguirlos.
    """)
    return


@app.cell
def _(X_train_pca, pca, plt):
    _fig, _ax = plt.subplots(figsize=(7, 6))
    _legit = X_train_pca["is_fraud"] == 0
    _ax.scatter(
        X_train_pca.loc[_legit, "PC1"],
        X_train_pca.loc[_legit, "PC2"],
        s=8,
        alpha=0.4,
        label="Legítima",
    )
    _ax.scatter(
        X_train_pca.loc[~_legit, "PC1"],
        X_train_pca.loc[~_legit, "PC2"],
        s=20,
        alpha=0.9,
        color="red",
        label="Fraude",
    )
    _ax.set_xlabel("PC1")
    _ax.set_ylabel("PC2")
    _ax.legend()
    print(f"PC1+PC2 explican {pca.explained_variance_ratio_[:2].sum():.1%} de la varianza")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ¿Qué es la SVD? (descomposición en valores singulares)

    Cualquier matriz $A$ de $m \times n$ se puede factorizar como

    $$A = USV^{\top}$$

    donde $U$ (columnas ortonormales, "vectores singulares izquierdos"), $S$ (diagonal con los **valores singulares** $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$) y $V$ (columnas ortonormales — direcciones en el espacio de las columnas de $A$). Es una generalización de la eigendescomposición que funciona para cualquier matriz, no solo cuadradas.

    **La conexión con PCA**: si $\tilde{X}$ es el train estandarizado (media 0 por columna), su covarianza muestral se puede reescribir con la SVD $\tilde{X} = USV^{\top}$ como

    $$\hat{\Sigma} = \frac{1}{n-1}\tilde{X}^{\top}\tilde{X} = \frac{1}{n-1}VS^{2}V^{\top}$$

    Es decir, la eigendescomposición de la covarianza *sale gratis* de la SVD: las **columnas de $V$ son las componentes principales** $v_i$ y la varianza de la componente $i$ es $\sigma_i^{2}/(n-1)$. Por eso `pca.fit_transform(X)` calcula internamente la SVD de $\tilde{X}$ y regresa $\tilde{X}V_{10}$ — las coordenadas ("scores") de cada punto en las 10 direcciones principales. Intuitivamente, $V$ rota los ejes del espacio de features ordenándolos de mayor a menor varianza.

    Fuentes:

    - [AMS Feature Column: *We Recommend a Singular Value Decomposition*](https://www.ams.org/publicoutreach/feature-column/fcarc-svd) — explicación visual clásica de qué hace cada factor $U$, $S$, $V$.
    - [Wikipedia: Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) — definición formal y propiedades.
    - [ESL §14.5.1](https://hastie.su.domains/Papers/ESLII.pdf) — PCA vía SVD (ya citado en el paso 5).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paso 5: Transformar `X_test`

    Aquí **no** se re-entrena nada: se aplican las funciones ya aprendidas en train (`transform`, no `fit_transform`). Si se recalcularan el escalado o el PCA con datos de test, la evaluación quedaría contaminada por información de los datos que se quieren evaluar (*leakage*).

    **Matemáticamente, ¿qué se calcula exactamente?** Sea $\tilde{X}_{tr}$ el train estandarizado y $\hat{\Sigma} = \frac{1}{n_{tr}-1}\tilde{X}_{tr}^{\top}\tilde{X}_{tr}$ su covarianza muestral. El PCA de train obtuvo los eigenvectores $v_1,\dots,v_{10}$ de $\hat{\Sigma}$ (equivalentemente, las 10 primeras columnas de $V$ en la SVD $\tilde{X}_{tr} = USV^{\top}$). Para cada renglón de test $x$:

    1. $\tilde{x} = D^{-1}(x - \bar{x}_{tr})$ — mapa afín que centra y escala columna a columna usando la media $\bar{x}_{tr}$ y las desviaciones $D = \operatorname{diag}(s_1,\dots,s_{27})$ **de train**.
    2. $x_{pca} = W_{10}^{\top}\tilde{x}$, con $W_{10} = [v_1\ \cdots\ v_{10}]$ — coordenadas del punto en la base de las 10 direcciones principales: la **proyección ortogonal** sobre el subespacio de dimensión 10 que maximiza la varianza en train.

    Fuentes:

    - [sklearn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) — fórmula de estandarización y semántica `fit`/`transform`.
    - [sklearn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — `transform` = proyección sobre los primeros $k$ vectores singulares derechos (SVD).
    - [Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, §14.5.1](https://hastie.su.domains/Papers/ESLII.pdf) — PCA vía SVD y proyección ortogonal.
    """)
    return


@app.cell
def _(X_test, feature_cols, pca, pc_cols, pd, scaler):
    X_test_scaled = scaler.transform(X_test[feature_cols])
    X_test_pca = pca.transform(X_test_scaled)

    X_test_pca = pd.DataFrame(X_test_pca, columns=pc_cols, index=X_test.index)
    X_test_pca["is_fraud"] = X_test["is_fraud"]
    print(f"X_test_pca: {X_test_pca.shape}")
    X_test_pca
    return (X_test_pca,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train vs Test en el mismo sistema de coordenadas

    El paso 5 no aprende nada nuevo: proyecta los puntos de test con **la misma** matriz $W_{10}$ calculada en train. Si el split 80/20 es representativo, la nube de test debe caer dentro de la nube de train (misma forma y rango) — esta gráfica es la verificación visual de que la proyección aprendida "generaliza" a datos nuevos.

    **`fit_transform` vs `transform`**: el resultado se parece (proyectar a 10 coordenadas), pero el paso 4 *aprendió y aplicó* — calculó la media $\bar{x}$, las desviaciones $s_j$ y las direcciones $W_{10}$ a partir de `X_train` — mientras el paso 5 solo *aplicó* esos parámetros ya aprendidos, sin recalcular nada.

    **Analogía**: el paso 4 construyó el sistema de coordenadas — definió dónde está el origen (media de train), la escala de cada eje (desviaciones de train) y hacia dónde apuntan PC1–PC10 ($W_{10}$). El paso 5 solo **expresa puntos nuevos en ese mismo sistema**, no construye uno propio. Por eso train y test se pueden graficar en los mismos ejes.

    **Consecuencia que delata la diferencia**: en `X_train_scaled` cada columna tiene media *exactamente* 0 — así se ajustó el scaler. En `X_test_scaled` las medias **no** son 0; son solo cercanas a 0 porque test se parece a train. Si se hubiera hecho `fit_transform` sobre test, se habría creado un sistema de coordenadas *distinto* (otro origen, otras direcciones): las distancias de Mahalanobis de train y test ya no serían comparables y test habría filtrado información al modelo (*leakage*).
    """)
    return


@app.cell
def _(X_test_pca, X_train_pca, plt):
    _fig, _ax = plt.subplots(figsize=(7, 6))
    _legit_tr = X_train_pca["is_fraud"] == 0
    _legit_te = X_test_pca["is_fraud"] == 0
    _ax.scatter(
        X_train_pca.loc[_legit_tr, "PC1"],
        X_train_pca.loc[_legit_tr, "PC2"],
        s=8,
        alpha=0.3,
        c="gray",
        label="Train legítima",
    )
    _ax.scatter(
        X_train_pca.loc[~_legit_tr, "PC1"],
        X_train_pca.loc[~_legit_tr, "PC2"],
        s=25,
        alpha=0.8,
        c="red",
        label="Train fraude",
    )
    _ax.scatter(
        X_test_pca.loc[_legit_te, "PC1"],
        X_test_pca.loc[_legit_te, "PC2"],
        s=14,
        alpha=0.7,
        c="steelblue",
        marker="x",
        label="Test legítima",
    )
    _ax.scatter(
        X_test_pca.loc[~_legit_te, "PC1"],
        X_test_pca.loc[~_legit_te, "PC2"],
        s=80,
        c="orange",
        marker="*",
        label="Test fraude",
    )
    _ax.set_xlabel("PC1")
    _ax.set_ylabel("PC2")
    _ax.legend()
    _fig
    return


if __name__ == "__main__":
    app.run()
