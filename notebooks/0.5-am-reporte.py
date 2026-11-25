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
    # 0.5 Reporte — análisis comparativo de los métodos

    Consolida los resultados de `0.3-am-mahalanobis` y `0.4-am-cd-kernel` sobre `X_test_pca`: **una sola tabla** con tiempo de ejecución, varianza explicada y la matriz de confusión en un renglón `[TN, FP, FN, TP]` por método, configuración y umbral — los insumos del reporte de una hoja (entregable 1 del `references/Proyecto_final__MCD.pdf`).
    """)
    return


@app.cell
def _():
    from credit_card_fraud_detection_dataset_2026.config import (
        FIGURES_DIR,
        MODELS_DIR,
        PROCESSED_DATA_DIR,
    )
    from credit_card_fraud_detection_dataset_2026.modeling.anomaly import (
        cd_fit,
        cd_score,
        evaluate,
        mahalanobis_fit,
        mahalanobis_score,
        radio_n,
    )

    X_TRAIN_PCA = PROCESSED_DATA_DIR / "x_train_pca.csv"
    X_TEST_PCA = PROCESSED_DATA_DIR / "x_test_pca.csv"
    return (
        FIGURES_DIR,
        MODELS_DIR,
        X_TEST_PCA,
        X_TRAIN_PCA,
        cd_fit,
        cd_score,
        evaluate,
        mahalanobis_fit,
        mahalanobis_score,
        radio_n,
    )


@app.cell
def _(MODELS_DIR, X_TEST_PCA, X_TRAIN_PCA, pd):
    import joblib

    X_train_pca = pd.read_csv(X_TRAIN_PCA, index_col=0)
    X_test_pca = pd.read_csv(X_TEST_PCA, index_col=0)
    pc_cols = [c for c in X_train_pca.columns if c != "is_fraud"]
    X_train_legit = X_train_pca.loc[X_train_pca["is_fraud"] == 0, pc_cols]
    pca = joblib.load(MODELS_DIR / "pca.joblib")
    print(
        f"legit train: {X_train_legit.shape} | test: {X_test_pca.shape} "
        f"({X_test_pca['is_fraud'].sum()} fraudes)"
    )
    return X_test_pca, X_train_legit, X_train_pca, pc_cols, pca


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tabla comparativa unificada

    Mismo experimento para todas las configuraciones: se ajusta el método solo con las no fraudulentas de train, se calculan los niveles $\lambda_{50..99}$ (percentiles del score sobre no fraudulentas) y se evalúa en `X_test_pca`. **Tiempo** = ajuste + score de train y test completos; **var %** = varianza explicada acumulada de las PCs usadas (del `pca.joblib` del pipeline).
    """)
    return


@app.cell
def _(
    X_test_pca,
    X_train_legit,
    cd_fit,
    cd_score,
    evaluate,
    mahalanobis_fit,
    mahalanobis_score,
    pc_cols,
    pca,
    pd,
    radio_n,
):
    import time

    # (metodo, n_comps, grado) — grado None = Mahalanobis
    _configs = [
        ("Mahalanobis", 10, None),
        ("Mahalanobis", 2, None),
        ("CD", 2, 2),
        ("CD", 2, 4),
        ("CD", 5, 2),
        ("CD", 7, 2),
        ("CD", 10, 2),
    ]
    _fraud = X_test_pca["is_fraud"] == 1
    _filas = []
    for _metodo, _k, _deg in _configs:
        _cols = pc_cols[:_k]
        _Xl = X_train_legit[_cols].to_numpy()
        _Xt = X_test_pca[_cols].to_numpy()
        _t0 = time.perf_counter()
        if _deg is None:
            _mu, _S, _Si = mahalanobis_fit(_Xl)
            _s_leg = mahalanobis_score(_Xl, _mu, _Si)
            _s_te = mahalanobis_score(_Xt, _mu, _Si)
            _config = f"{_k} comps"
        else:
            _Mi, _D = cd_fit(_Xl, _deg)
            _s_leg = cd_score(_Xl, _Mi, _deg)
            _s_te = cd_score(_Xt, _Mi, _deg)
            _config = f"{_k} comps, grado {_deg}"
        _t = time.perf_counter() - _t0
        _var = pca.explained_variance_ratio_[:_k].sum()
        _ev = evaluate(
            _s_te,
            _fraud,
            {f"λ{n}": radio_n(_s_leg, n) for n in [50, 75, 90, 95, 99]},
        )
        for _umb, _row in _ev.iterrows():
            _filas.append(
                {
                    "método": _metodo,
                    "config": _config,
                    "umbral": _umb,
                    **_row.to_dict(),
                    "var %": round(100 * _var, 1),
                    "t (ms)": round(_t * 1000, 1),
                }
            )
    comparativa = pd.DataFrame(_filas).set_index(["método", "config", "umbral"])
    comparativa
    return (comparativa,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resumen: mejor umbral por configuración

    Vista compacta para el reporte: por cada método/config, el máximo de fraudes detectados (TP de 8 en test) con el umbral donde se logra, y los FP que cuesta.
    """)
    return


@app.cell
def _(comparativa, pd):
    resumen = (
        comparativa.reset_index()
        .sort_values(["TP", "FP"], ascending=[False, True])
        .groupby(["método", "config"], as_index=False)
        .first()
        .set_index(["método", "config"])
        .drop(columns="umbral")
        .assign(recall=lambda t: (t["TP"] / 8).round(2))
    )
    resumen
    return (resumen,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gráficas en dos dimensiones (para el reporte)

    Regenera las tres figuras 2D y las exporta a `reports/figures/`: elipses de Mahalanobis, y subniveles del kernel CD grado 2 y grado 4.
    """)
    return


@app.cell
def _(
    FIGURES_DIR,
    X_train_legit,
    X_train_pca,
    cd_fit,
    cd_score,
    mahalanobis_fit,
    mahalanobis_score,
    np,
    plt,
    radio_n,
):
    from matplotlib.patches import Ellipse

    _pc2 = ["PC1", "PC2"]
    _legit2 = X_train_legit[_pc2].to_numpy()
    _fraud2 = X_train_pca.loc[X_train_pca["is_fraud"] == 1, _pc2]

    def _scatter(_ax):
        _ax.scatter(
            _legit2[:, 0], _legit2[:, 1], s=8, alpha=0.3, c="gray",
            label="No fraudulenta",
        )
        _ax.scatter(
            _fraud2["PC1"], _fraud2["PC2"], s=25, alpha=0.8, c="red",
            label="Fraude",
        )
        _ax.set_xlabel("PC1")
        _ax.set_ylabel("PC2")
        _ax.legend(fontsize=8)

    _niv = [50, 75, 90, 95, 99]

    # Mahalanobis: elipses
    _mu2, _S2, _S2i = mahalanobis_fit(_legit2)
    _radios = {n: radio_n(mahalanobis_score(_legit2, _mu2, _S2i), n) for n in _niv}
    _vals, _vecs = np.linalg.eigh(_S2)
    _fig, _ax = plt.subplots(figsize=(7, 6))
    _scatter(_ax)
    _ang = np.degrees(np.arctan2(_vecs[1, 0], _vecs[0, 0]))
    for _n, _r in _radios.items():
        _ax.add_patch(
            Ellipse(
                _mu2, 2 * _r * np.sqrt(_vals[0]), 2 * _r * np.sqrt(_vals[1]),
                angle=_ang, fill=False, label=f"r{_n}",
            )
        )
    _ax.set_title("Mahalanobis — elipses de nivel")
    _fig.savefig(FIGURES_DIR / "mahalanobis_elipses.png", dpi=150, bbox_inches="tight")

    # CD: contornos de p_CD en rejilla
    def _contorno(grado, nombre):
        _Mi, _D = cd_fit(_legit2, grado)
        _niveles = {n: radio_n(cd_score(_legit2, _Mi, grado), n) for n in _niv}
        _x1 = np.linspace(*np.percentile(_legit2[:, 0], [0.5, 99.5]), 200)
        _x2 = np.linspace(*np.percentile(_legit2[:, 1], [0.5, 99.5]), 200)
        _gx, _gy = np.meshgrid(_x1, _x2)
        _Z = cd_score(
            np.column_stack([_gx.ravel(), _gy.ravel()]), _Mi, grado
        ).reshape(_gx.shape)
        _fig, _ax = plt.subplots(figsize=(7, 6))
        _cs = _ax.contour(_gx, _gy, _Z, levels=sorted(_niveles.values()))
        _ax.clabel(_cs, fmt={l: f"λ{n}" for n, l in _niveles.items()}, fontsize=8)
        _scatter(_ax)
        _ax.set_title(f"Kernel CD grado {grado} — subconjuntos de nivel")
        _fig.savefig(FIGURES_DIR / nombre, dpi=150, bbox_inches="tight")
        return _fig

    _fig2 = _contorno(2, "cd_grado2_contornos.png")
    _fig4 = _contorno(4, "cd_grado4_contornos.png")
    print(f"Figuras en {FIGURES_DIR}")
    _fig4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusiones (borrador para el reporte — ajustar a gusto)

    - **Ningún umbral separa limpio**: detectar más fraudes siempre cuesta ~200 FP (más de la mitad de las no fraudulentas de test). La región de fraude se solapa con la normal en estas features — hallazgo central del dataset.
    - **Mejor recall**: kernel CD grado 2 con **7 componentes** llega a **6/8 fraudes** (vs 5/8 de Mahalanobis-10d) — la flexibilidad de los subniveles polinomiales sí captura algo que la elipse no. En λ99 todos colapsan a ~1 TP: los niveles altos son demasiado estrictos para fraudes "centrados".
    - **Varianza explicada**: con 2 comps (21.1%) todos los métodos pierden ~1-2 TP vs sus versiones con más componentes — las PCs descartadas cargan señal de fraude.
    - **Costo**: el tiempo del CD crece con la base $\binom{k+2}{2}$ (1.4 ms → 36 ms de 2 a 10 comps) — sigue siendo trivial a esta escala, pero el trade-off existe.
    - **Recomendación**: para detección operativa convendría un umbral intermedio (~λ50-λ75) aceptando FP, o combinar con reglas adicionales — ni Mahalanobis ni CD solos dan un clasificador usable con 8 fraudes de muestra.
    """)
    return


if __name__ == "__main__":
    app.run()
