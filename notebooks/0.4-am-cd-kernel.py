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
    from sklearn.metrics import confusion_matrix

    return confusion_matrix, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0.4 Segunda propuesta — kernel de Christoffel-Darboux

    Continuación de `0.3-am-mahalanobis`: sección 3 del proyecto (ver `references/Proyecto_final__MCD.pdf`). Carga los conjuntos procesados (`data/processed/`, generados por `features.py`).
    """)
    return


@app.cell
def _():
    from credit_card_fraud_detection_dataset_2026.config import PROCESSED_DATA_DIR

    X_TRAIN_PCA = PROCESSED_DATA_DIR / "x_train_pca.csv"
    X_TEST_PCA = PROCESSED_DATA_DIR / "x_test_pca.csv"
    return X_TEST_PCA, X_TRAIN_PCA


@app.cell
def _(X_TEST_PCA, X_TRAIN_PCA, pd):
    X_train_pca = pd.read_csv(X_TRAIN_PCA, index_col=0)
    X_test_pca = pd.read_csv(X_TEST_PCA, index_col=0)
    pc_cols = [c for c in X_train_pca.columns if c != "is_fraud"]
    X_train_legit = X_train_pca.loc[X_train_pca["is_fraud"] == 0, pc_cols]
    print(f"No fraudulentas train: {X_train_legit.shape} | test: {X_test_pca.shape}")
    return X_test_pca, X_train_legit, X_train_pca, pc_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## La re-codificación polinomial $\varphi(x)$

    En vez de medir distancias directamente en el espacio de las PCs, cada transacción $x=(x_1,x_2)$ se "levanta" al espacio de **polinomios bivariados de grado $\le 2$**:

    $$\varphi(x) = (1,\ x_1,\ x_2,\ x_1^2,\ x_1x_2,\ x_2^2) \in \mathbb{R}^6$$

    La idea: una "bola" en el espacio de $\varphi$ puede verse como una curva arbitraria en $\mathbb{R}^2$ — formas que una elipse de Mahalanobis no puede capturar. En general, para $d$ variables y grado $g$, la base tiene $\binom{d+g}{g}$ monomios.
    """)
    return


@app.cell
def _(np):
    from itertools import combinations_with_replacement

    def phi(X, degree):
        """Mapa a la base de monomios de grado total <= degree (X: N x d -> N x C(d+g, g))."""
        _X = np.atleast_2d(np.asarray(X, dtype=float))
        _cols = [np.ones(len(_X))]
        for _total in range(1, degree + 1):
            for _idx in combinations_with_replacement(range(_X.shape[1]), _total):
                _cols.append(np.prod(_X[:, _idx], axis=1))
        return np.column_stack(_cols)

    def cd_fit(legit_X, degree, lam=1e-6):
        """Ajusta el kernel CD en legit_X: devuelve M_reg^{-1} = (M_n + lam*I)^{-1}."""
        _D = phi(legit_X, degree)
        _M = _D.T @ _D / len(_D) + lam * np.eye(_D.shape[1])
        return np.linalg.inv(_M), _D

    def cd_score(x, M_inv, degree):
        """p_CD(x) = phi(x)^T M_reg^{-1} phi(x)."""
        _p = phi(x, degree)
        return np.einsum("ij,jk,ik->i", _p, M_inv, _p)

    def radio_n(valores_legit, n):
        """Nivel/radio l_n tal que n% de las no fraudulentas queda dentro,
        sin contener (n+1)%: el k-esimo valor ordenado, k = ceil(nN/100)."""
        _ds = np.sort(np.asarray(valores_legit))
        _k = int(np.ceil(n / 100 * len(_ds)))
        return float(_ds[_k - 1])

    return cd_fit, cd_score, phi, radio_n


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## La matriz de momentos empírica — ¿qué tiene que ver con el kernel?

    **Básico**: el problema de Mahalanobis es que solo puede aprender *elipses*. El kernel CD resuelve lo mismo pero en el espacio de $\varphi$: la matriz $D$ (renglones = $\varphi(x_i)$ de cada transacción no fraudulenta) guarda dónde está la masa de los datos en ese espacio lifted, y la **matriz de momentos** $M_n$ resume esa geometría.

    **Matemático**: la matriz de momentos empírica es el segundo momento de los features liftados:

    $$M_n = \frac{1}{N}D^{\top}D = \frac{1}{N}\sum_{i=1}^{N}\varphi(x_i)\varphi(x_i)^{\top} = \hat{\mathbb{E}}[\varphi\varphi^{\top}]$$

    — es como la covarianza, pero *sin centrar*. El polinomio de Christoffel-Darboux

    $$p_{CD}(x) = \varphi(x)^{\top}M_{reg}^{-1}\varphi(x) = \big\|M_{reg}^{-1/2}\varphi(x)\big\|^2$$

    es la **norma al cuadrado del vector de features blanqueado** — idéntico en forma a Mahalanobis pero con $\varphi(x)$ en lugar de $x$. Y $k(x,y) = \varphi(x)^{\top}M^{-1}\varphi(y)$ es un producto interior tras blanquear → kernel simétrico y semidefinido positivo; $p_{CD}$ es su diagonal $k(x,x)$. Su inversa $\Lambda(x) = 1/p_{CD}(x)$ es la *función de Christoffel*: grande donde hay densidad de datos, chica en regiones vírgenes → $p_{CD}$ funciona como score de rareza.

    **¿Relación con Mahalanobis?** (pregunta del PDF): si el mapa fuera *lineal*, $\varphi(x) = x - \mu$, entonces $M = \Sigma$ y $p_{CD}(x) = d_M(x)^2$ — **la distancia de Mahalanobis es el caso de grado 1 del kernel CD**. La versión polinomial es la generalización a features no lineales.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularización de Tikhonov — ¿por qué $M_{reg}$ siempre es invertible?

    $M_n$ es una matriz de Gram → sus eigenvalores son $\ge 0$ (semidefinida positiva), pero puede ser singular o mal condicionada si la base es casi linealmente dependiente sobre los datos. Al sumar $M_{reg} = M_n + \lambda I$ con $\lambda = 10^{-6}$:

    $$\text{eig}(M_{reg}) = \text{eig}(M_n) + \lambda \ge \lambda > 0$$

    Todos los eigenvalores quedan $\ge \lambda > 0$ → $M_{reg}$ es **definida positiva** → siempre invertible (y además bien condicionada). El precio: una ligera distorsión de la geometría, despreciable para $\lambda$ chica.
    """)
    return


@app.cell
def _(X_train_legit, cd_fit, np):
    _pc2 = X_train_legit[["PC1", "PC2"]].to_numpy()
    M_inv, D = cd_fit(_pc2, 2)
    M = D.T @ D / len(D)
    _eig_min = float(np.linalg.eigvalsh(M).min())
    _eig_min_reg = float(np.linalg.eigvalsh(M + 1e-6 * np.eye(M.shape[0])).min())
    print(
        f"D: {D.shape} (base R6) | eig min de M: {_eig_min:.2e} "
        f"-> con Tikhonov: {_eig_min_reg:.2e}"
    )
    return D, M_inv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Subconjuntos de nivel

    El equivalente a las "bolas" de Mahalanobis son los **subconjuntos de nivel** $S_{\lambda} = \{x : p_{CD}(x) \le \lambda\}$, con $\lambda_n$ = percentil $n$ de $p_{CD}$ sobre las no fraudulentas (misma función `radio_n` que en la libreta anterior).

    **¿$p_{CD}$ determina una distancia al origen y los subconjuntos son bolas centradas en $\mathbb{R}^2$?** (pregunta del PDF): $p_{CD}(x) = \|M^{-1/2}\varphi(x)\|^2$ sí es una norma² al origen — pero del vector *lifted* en $\mathbb{R}^6$, no de $x$. En $\mathbb{R}^2$ los subniveles de un polinomio de grado 4 son curvas arbitrarias (ni siquiera necesariamente convexas) — **no son bolas**, y justamente ahí está la ventaja sobre la elipse de Mahalanobis.
    """)
    return


@app.cell
def _(M_inv, X_train_legit, cd_score, radio_n):
    p_cd_legit = cd_score(X_train_legit[["PC1", "PC2"]].to_numpy(), M_inv, 2)
    niveles = {n: radio_n(p_cd_legit, n) for n in [50, 75, 90, 95, 99]}
    for _n, _l in niveles.items():
        print(f"lambda_{_n} = {_l:.2f}")
    return niveles, p_cd_legit


@app.cell
def _(X_train_legit, X_train_pca, M_inv, cd_score, niveles, np, plt):
    _pc2 = ["PC1", "PC2"]
    _legit2 = X_train_legit[_pc2].to_numpy()

    # p_CD sobre una rejilla para dibujar los subconjuntos de nivel
    _x1 = np.linspace(*np.percentile(_legit2[:, 0], [0.5, 99.5]), 200)
    _x2 = np.linspace(*np.percentile(_legit2[:, 1], [0.5, 99.5]), 200)
    _gx, _gy = np.meshgrid(_x1, _x2)
    _Z = cd_score(
        np.column_stack([_gx.ravel(), _gy.ravel()]), M_inv, 2
    ).reshape(_gx.shape)

    _fig, _ax = plt.subplots(figsize=(7, 6))
    _cs = _ax.contour(_gx, _gy, _Z, levels=sorted(niveles.values()))
    _ax.clabel(_cs, fmt={l: f"λ{n}" for n, l in niveles.items()}, fontsize=8)
    _ax.scatter(_legit2[:, 0], _legit2[:, 1], s=8, alpha=0.3, c="gray", label="No fraudulenta")
    _fraud2 = X_train_pca.loc[X_train_pca["is_fraud"] == 1, _pc2]
    _ax.scatter(_fraud2["PC1"], _fraud2["PC2"], s=25, alpha=0.8, c="red", label="Fraude")
    _ax.set_xlabel("PC1")
    _ax.set_ylabel("PC2")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Validación con `X_test_pca` (grado 2, PC1–PC2)

    Regla: $p_{CD}(x) > \lambda_n$ → fraude. **FP** = no fraudulentas de test fuera del subnivel; **FN** = fraudes dentro. Cada renglón es la **matriz de confusión en un renglón** `[TN, FP, FN, TP]` vía `sklearn.metrics.confusion_matrix` (formato del entregable).
    """)
    return


@app.cell
def _(X_test_pca, M_inv, cd_score, confusion_matrix, niveles, np, pd):
    _sc = cd_score(X_test_pca[["PC1", "PC2"]].to_numpy(), M_inv, 2)
    _fraud = X_test_pca["is_fraud"] == 1

    _filas = []
    for _l in niveles.values():
        _tn, _fp, _fn, _tp = confusion_matrix(_fraud, _sc > _l).ravel()
        _filas.append({"TN": _tn, "FP": _fp, "FN": _fn, "TP": _tp})
    validacion_cd2 = pd.DataFrame(
        _filas, index=[f"λ{n}" for n in niveles]
    ).rename_axis("umbral")
    validacion_cd2
    return (validacion_cd2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Base de grado 4

    **¿Quién sería la base en ese caso?** (pregunta del PDF): todos los monomios $x_1^{a}x_2^{b}$ con $a+b \le 4$ → $\binom{4+2}{2} = \mathbf{15}$ monomios: los 6 de grado ≤ 2 más $(x_1^3, x_1^2x_2, x_1x_2^2, x_2^3, x_1^4, x_1^3x_2, x_1^2x_2^2, x_1x_2^3, x_2^4)$. Más grados = fronteras de nivel aún más flexibles, con riesgo de sobreajuste a las no fraudulentas.
    """)
    return


@app.cell
def _(X_train_legit, X_train_pca, cd_fit, cd_score, np, pd, plt, radio_n):
    _pc2 = ["PC1", "PC2"]
    _legit2 = X_train_legit[_pc2].to_numpy()
    M_inv4, D4 = cd_fit(_legit2, 4)
    print(f"Base grado 4: {D4.shape[1]} monomios")

    _p4 = cd_score(_legit2, M_inv4, 4)
    niveles4 = {n: radio_n(_p4, n) for n in [50, 75, 90, 95, 99]}

    _x1 = np.linspace(*np.percentile(_legit2[:, 0], [0.5, 99.5]), 200)
    _x2 = np.linspace(*np.percentile(_legit2[:, 1], [0.5, 99.5]), 200)
    _gx, _gy = np.meshgrid(_x1, _x2)
    _Z = cd_score(np.column_stack([_gx.ravel(), _gy.ravel()]), M_inv4, 4).reshape(
        _gx.shape
    )

    _fig, _ax = plt.subplots(figsize=(7, 6))
    _cs = _ax.contour(_gx, _gy, _Z, levels=sorted(niveles4.values()))
    _ax.clabel(_cs, fmt={l: f"λ{n}" for n, l in niveles4.items()}, fontsize=8)
    _ax.scatter(_legit2[:, 0], _legit2[:, 1], s=8, alpha=0.3, c="gray", label="No fraudulenta")
    _fraud2 = X_train_pca.loc[X_train_pca["is_fraud"] == 1, _pc2]
    _ax.scatter(_fraud2["PC1"], _fraud2["PC2"], s=25, alpha=0.8, c="red", label="Fraude")
    _ax.set_xlabel("PC1")
    _ax.set_ylabel("PC2")
    _ax.legend()
    _fig
    return D4, M_inv4, niveles4


@app.cell
def _(X_test_pca, M_inv4, cd_score, confusion_matrix, niveles4, np, pd):
    _sc = cd_score(X_test_pca[["PC1", "PC2"]].to_numpy(), M_inv4, 4)
    _fraud = X_test_pca["is_fraud"] == 1

    _filas = []
    for _l in niveles4.values():
        _tn, _fp, _fn, _tp = confusion_matrix(_fraud, _sc > _l).ravel()
        _filas.append({"TN": _tn, "FP": _fp, "FN": _fn, "TP": _tp})
    validacion_cd4 = pd.DataFrame(
        _filas, index=[f"λ{n}" for n in niveles4]
    ).rename_axis("umbral")
    validacion_cd4
    return (validacion_cd4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Repitiendo con 5, 7 y 10 componentes

    Mismo ejercicio pero con $\varphi$ de grado 2 sobre las primeras $k$ PCs: la base crece a $\binom{k+2}{2}$ monomios (k=5 → 21, k=7 → 36, k=10 → 66). En más dimensiones no se puede graficar — solo la tabla FP/FN + tiempo de ejecución (insumo del reporte).
    """)
    return


@app.cell
def _(
    X_test_pca,
    X_train_legit,
    cd_fit,
    cd_score,
    confusion_matrix,
    np,
    pc_cols,
    pd,
    radio_n,
):
    import time

    _fraud = X_test_pca["is_fraud"] == 1
    _filas = []
    for _k in [5, 7, 10]:
        _cols = pc_cols[:_k]
        _t0 = time.perf_counter()
        _Mi, _D = cd_fit(X_train_legit[_cols].to_numpy(), 2)
        _p_leg = cd_score(X_train_legit[_cols].to_numpy(), _Mi, 2)
        _sc = cd_score(X_test_pca[_cols].to_numpy(), _Mi, 2)
        _t = time.perf_counter() - _t0
        for _n in [50, 75, 90, 95, 99]:
            _l = radio_n(_p_leg, _n)
            _tn, _fp, _fn, _tp = confusion_matrix(_fraud, _sc > _l).ravel()
            _filas.append(
                {
                    "comps": _k,
                    "umbral": f"λ{_n}",
                    "TN": _tn,
                    "FP": _fp,
                    "FN": _fn,
                    "TP": _tp,
                    "t (ms)": round(_t * 1000, 1),
                }
            )
    comparacion_k = pd.DataFrame(_filas).set_index(["comps", "umbral"])
    comparacion_k
    return (comparacion_k,)


if __name__ == "__main__":
    app.run()

