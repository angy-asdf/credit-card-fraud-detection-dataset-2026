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
    import numpy as np
    import pandas as pd

    return np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0.3 Detección de anomalías — distancia de Mahalanobis

    Continuación de `0.2-am-preprocessing`: paso 6 del proyecto (ver `references/Proyecto_final__MCD.pdf`). Carga los conjuntos ya transformados (`data/processed/`, generados por `features.py`).
    """)
    return


@app.cell
def _():
    from credit_card_fraud_detection_dataset_2026.config import (
        MODELS_DIR,
        PROCESSED_DATA_DIR,
    )

    X_TRAIN_PCA = PROCESSED_DATA_DIR / "x_train_pca.csv"
    X_TEST_PCA = PROCESSED_DATA_DIR / "x_test_pca.csv"
    return MODELS_DIR, X_TEST_PCA, X_TRAIN_PCA


@app.cell
def _():
    from credit_card_fraud_detection_dataset_2026.modeling.anomaly import (
        evaluate,
        mahalanobis_fit,
        mahalanobis_score,
        radio_n,
    )

    return evaluate, mahalanobis_fit, mahalanobis_score, radio_n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Datos procesados (PC1–PC10 + `is_fraud`)
    """)
    return


@app.cell
def _(X_TEST_PCA, X_TRAIN_PCA, pd):
    X_train_pca = pd.read_csv(X_TRAIN_PCA, index_col=0)
    X_test_pca = pd.read_csv(X_TEST_PCA, index_col=0)
    print(
        f"X_train_pca: {X_train_pca.shape} ({X_train_pca['is_fraud'].sum()} fraudes) | "
        f"X_test_pca: {X_test_pca.shape} ({X_test_pca['is_fraud'].sum()} fraudes)"
    )
    X_train_pca.head()
    return X_test_pca, X_train_pca


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Paso 6: distancia de Mahalanobis

    Primera propuesta del PDF. De `X_train_pca` se filtran las transacciones **no fraudulentas** — el PDF llama a este conjunto `X_train_legit` — y **solo sobre ellas** se estiman el centro $\mu$ y la covarianza $\Sigma$. La distancia de Mahalanobis

    $$d_M(x)^2 = (x-\mu)^{\top}\Sigma^{-1}(x-\mu)$$

    mide qué tan lejos está un punto del centro de las no fraudulentas, ponderando por la covarianza: una desviación en una dirección de mucha varianza "pesa" menos que una en una dirección de poca varianza — por eso las "bolas" $\{x : d_M(x) \le r\}$ son elipsoides, no esferas. Las PCs son incorrelacionadas por construcción, así que $\Sigma$ es casi diagonal.

    **¿En qué dimensión se calcula esta distancia?** En $\mathbb{R}^{10}$ — el espacio de las 10 componentes principales (en el ejercicio didáctico del final, en $\mathbb{R}^2$ con PC1–PC2).
    """)
    return


@app.cell
def _(X_train_pca, mahalanobis_fit, pd):
    pc_cols = [c for c in X_train_pca.columns if c != "is_fraud"]
    X_train_legit = X_train_pca.loc[X_train_pca["is_fraud"] == 0, pc_cols]

    mu, Sigma, Sigma_inv = mahalanobis_fit(X_train_legit)
    print(f"No fraudulentas train: {X_train_legit.shape} | Sigma: {Sigma.shape}")
    pd.DataFrame(Sigma, index=pc_cols, columns=pc_cols).round(2)
    return Sigma_inv, X_train_legit, mu, pc_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Radios por percentil

    El PDF pide: dado un porcentaje $n\%$, encontrar el radio $r_n$ tal que el $n\%$ de las no fraudulentas quede dentro de la bola $B_n(0) = \{x : d_M(x) \le r_n\}$ — es decir, el **percentil empírico** $n$ de las distancias de las no fraudulentas.

    **"Pero que no contenga $(n+1)\%$"**: pide el radio *mínimo* que captura $\ge n\%$ — un radio más grande capturaría de más. Con $N$ puntos se ordenan las distancias y se toma la $k$-ésima con $k = \lceil nN/100 \rceil$: entonces la fracción dentro es $k/N \ge n\%$ por construcción, y $k/N < (n+1)\%$ mientras $N \ge 100$ — con $N=1492$ se cumple para todo $n$ (para $n=99$: $1478/1492 = 99.06\% < 100\%$).
    """)
    return


@app.cell
def _(Sigma_inv, X_train_legit, mahalanobis_score, mu, pd, radio_n):
    d_legit = mahalanobis_score(X_train_legit.to_numpy(), mu, Sigma_inv)
    radios = {n: radio_n(d_legit, n) for n in [50, 75, 90, 95, 99]}

    radios_df = pd.DataFrame(
        {"r_n": radios.values()}, index=[f"{n}%" for n in radios]
    ).rename_axis("n% dentro")
    print(radios_df.round(3))
    radios_df.round(3)
    return d_legit, radios


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zonas de seguridad (con los fraudes de train)

    Los percentiles definen "qué tan típico" es un punto respecto a las no fraudulentas — de ahí salen las bandas:

    | Zona | Región | Significado |
    |---|---|---|
    | **seguridad extrema** | $d_M < r_{50}$ | Donde vive la mitad más típica de las no fraudulentas — el "núcleo" del comportamiento normal |
    | **alta** | $r_{50} \le d_M < r_{90}$ | Zona normal pero menos densa — hasta el 90% de las no fraudulentas caben aquí |
    | **baja** | $r_{90} \le d_M < r_{99}$ | Periferia: solo ~9% de las no fraudulentas llegan tan lejos |
    | **peligro** | $d_M \ge r_{99}$ | Región donde **menos del 1%** de las no fraudulentas caen → un punto ahí es estadísticamente atípico = sospechoso |

    **Qué significa "peligro"**: no "fraude seguro", sino la zona donde el comportamiento normal casi nunca llega — el detector reporta como anómalo lo que cae ahí. Una no fraudulenta en la zona de peligro se convierte en falso positivo; un fraude que cae dentro de las bolas se escapa (falso negativo).

    *Matiz*: el PDF nombra las categorías pero no fija las fronteras — la asignación con $r_{50}$/$r_{90}$/$r_{99}$ es nuestra elección de diseño (reusa los radios ya calculados). La tabla siguiente muestra cuántos fraudes y no fraudulentas de train caen por banda: la evidencia de si la partición separa.
    """)
    return


@app.cell
def _(Sigma_inv, X_train_pca, mahalanobis_score, mu, pc_cols, pd, radios):
    d_train_all = mahalanobis_score(X_train_pca[pc_cols].to_numpy(), mu, Sigma_inv)

    def _zona(d):
        if d < radios[50]:
            return "extrema"
        if d < radios[90]:
            return "alta"
        if d < radios[99]:
            return "baja"
        return "peligro"

    zonas = (
        pd.DataFrame({"d": d_train_all, "is_fraud": X_train_pca["is_fraud"]})
        .assign(zona=lambda t: t["d"].map(_zona))
        .groupby(["zona", "is_fraud"], observed=True)
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "no fraudulentas", 1: "fraudes"})
        .reindex(["extrema", "alta", "baja", "peligro"])
    )
    zonas
    return (zonas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validación con `X_test_pca`

    Regla de detección por umbral: un punto **fuera** de la bola se reporta como fraude. **FP** = no fraudulentas marcadas como fraude (falsos positivos); **FN** = fraudes dentro de la bola que se escapan (falsos negativos). Solo hay 8 fraudes en test — los FN se leen de a uno.

    Cada renglón es una **matriz de confusión en un renglón** `[TN, FP, FN, TP]` (formato del entregable), calculada con `sklearn.metrics.confusion_matrix`.
    """)
    return


@app.cell
def _(
    Sigma_inv,
    X_test_pca,
    evaluate,
    mahalanobis_score,
    mu,
    pc_cols,
    radios,
):
    d_test = mahalanobis_score(X_test_pca[pc_cols].to_numpy(), mu, Sigma_inv)
    validacion = evaluate(
        d_test,
        X_test_pca["is_fraud"] == 1,
        {f"r{n}": r for n, r in radios.items()},
    )
    validacion
    return (validacion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ejercicio didáctico: solo PC1 y PC2

    El PDF pide repetir con las dos primeras componentes y graficar las bolas — en 2D son elipses. Para un radio $r$, la elipse $\{x : (x-\mu_2)^\top\Sigma_2^{-1}(x-\mu_2) \le r^2\}$ tiene semi-ejes de longitud $r\sqrt{\lambda_i}$ apuntando en la dirección del eigenvector $v_i$ de $\Sigma_2$.
    """)
    return


@app.cell
def _(
    X_train_legit,
    X_train_pca,
    mahalanobis_fit,
    mahalanobis_score,
    np,
    plt,
    radio_n,
):
    from matplotlib.patches import Ellipse

    _pc2 = ["PC1", "PC2"]
    _legit2 = X_train_legit[_pc2].to_numpy()
    mu2, Sigma2, Sigma2_inv = mahalanobis_fit(_legit2)
    _vals, _vecs = np.linalg.eigh(Sigma2)

    _d_legit2 = mahalanobis_score(_legit2, mu2, Sigma2_inv)
    radios2 = {n: radio_n(_d_legit2, n) for n in [50, 75, 90, 95, 99]}

    _fig, _ax = plt.subplots(figsize=(7, 6))
    _ax.scatter(
        _legit2[:, 0], _legit2[:, 1], s=8, alpha=0.3, c="gray", label="No fraudulenta"
    )
    _fraud2 = X_train_pca.loc[X_train_pca["is_fraud"] == 1, _pc2]
    _ax.scatter(_fraud2["PC1"], _fraud2["PC2"], s=25, alpha=0.8, c="red", label="Fraude")
    _ang = np.degrees(np.arctan2(_vecs[1, 0], _vecs[0, 0]))
    for _n, _r in radios2.items():
        _ax.add_patch(
            Ellipse(
                mu2,
                2 * _r * np.sqrt(_vals[0]),
                2 * _r * np.sqrt(_vals[1]),
                angle=_ang,
                fill=False,
                label=f"r{_n}",
            )
        )
    _ax.set_xlabel("PC1")
    _ax.set_ylabel("PC2")
    _ax.legend(fontsize=8)
    _fig
    return Sigma2_inv, mu2, radios2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ¿Pierde eficiencia con 2 componentes?

    Misma tabla FP/FN pero calculada solo con PC1–PC2 vs las 10 componentes completas, más la varianza explicada por dos componentes (del `pca.joblib` del pipeline).
    """)
    return


@app.cell
def _(
    MODELS_DIR,
    Sigma2_inv,
    X_test_pca,
    evaluate,
    mahalanobis_score,
    mu2,
    radios2,
    validacion,
):
    import joblib

    _pca = joblib.load(MODELS_DIR / "pca.joblib")
    _var2 = _pca.explained_variance_ratio_[:2].sum()
    print(f"Varianza explicada PC1+PC2: {_var2:.1%}")

    _d_test2 = mahalanobis_score(
        X_test_pca[["PC1", "PC2"]].to_numpy(), mu2, Sigma2_inv
    )
    _val2 = evaluate(
        _d_test2,
        X_test_pca["is_fraud"] == 1,
        {f"r{n}": r for n, r in radios2.items()},
    )

    comparacion = validacion.join(_val2, lsuffix=" (10 comps)", rsuffix=" (2 comps)")
    comparacion
    return (comparacion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Eficiencia como métrica escalar

    La matriz de confusión lo muestra por conteos; para responder "¿pierde eficiencia?" con un número, `sklearn.metrics` da las métricas estándar por configuración y umbral:

    - **recall** $= TP/(TP+FN)$: fracción de fraudes detectados — la métrica que de verdad importa aquí
    - **precision** $= TP/(TP+FP)$: de lo marcado como fraude, cuánto sí era
    - **f1**: media armónica de precision y recall
    - **accuracy** $= (TP+TN)/N$: ojo — con 377/385 no fraudulentas, un clasificador que marca todo como "no fraude" ya tiene ~98% de accuracy; **engaña en datos desbalanceados**
    """)
    return


@app.cell
def _(
    Sigma2_inv,
    Sigma_inv,
    X_test_pca,
    mahalanobis_score,
    mu,
    mu2,
    pc_cols,
    pd,
    radios,
    radios2,
):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    _y = X_test_pca["is_fraud"] == 1
    _d10 = mahalanobis_score(X_test_pca[pc_cols].to_numpy(), mu, Sigma_inv)
    _d2 = mahalanobis_score(
        X_test_pca[["PC1", "PC2"]].to_numpy(), mu2, Sigma2_inv
    )

    _filas = []
    for _n in radios:
        for _cfg, _d, _r in [
            ("10 comps", _d10, radios[_n]),
            ("2 comps", _d2, radios2[_n]),
        ]:
            _pred = _d >= _r
            _filas.append(
                {
                    "config": _cfg,
                    "umbral": f"r{_n}",
                    "accuracy": accuracy_score(_y, _pred),
                    "precision": precision_score(_y, _pred, zero_division=0),
                    "recall": recall_score(_y, _pred),
                    "f1": f1_score(_y, _pred, zero_division=0),
                }
            )
    eficiencia = (
        pd.DataFrame(_filas).set_index(["config", "umbral"]).round(3)
    )
    eficiencia
    return (eficiencia,)


if __name__ == "__main__":
    app.run()
