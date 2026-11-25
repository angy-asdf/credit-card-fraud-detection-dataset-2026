"""Deteccion de anomalias: Mahalanobis y kernel Christoffel-Darboux.

Helpers compartidos por las libretas 0.3-0.5. Las funciones implementan
las formulas del proyecto final (distancia de Mahalanobis, radios por
percentil y densidad p_CD = phi^T M^-1 phi con regularizacion Tikhonov).
"""

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def radio_n(valores, n):
    """Radio r_n tal que n% de los valores queda dentro de B_rn(0),
    pero que no contiene (n+1)%: el k-esimo valor ordenado,
    con k = ceil(nN/100). Aplica tanto a distancias Mahalanobis
    como a valores de p_CD (niveles lambda_n)."""
    ds = np.sort(np.asarray(valores))
    k = int(np.ceil(n / 100 * len(ds)))
    return float(ds[k - 1])


def mahalanobis_fit(X):
    """Ajusta el centro mu y la covarianza Sigma de X (las transacciones
    no fraudulentas de train). Devuelve (mu, Sigma, Sigma_inv)."""
    mu = np.asarray(X).mean(axis=0)
    Sigma = np.cov(np.asarray(X), rowvar=False)
    return mu, Sigma, np.linalg.inv(Sigma)


def mahalanobis_score(X, mu, Sigma_inv):
    """Distancia Mahalanobis al cuadrado d_M(x)^2 = (x-mu)^T Sigma^-1 (x-mu)."""
    z = np.asarray(X) - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", z, Sigma_inv, z))


def phi(X, degree):
    """Base de monomios de grado <= degree: phi(x) = [1, x_i, x_i*x_j, ...].
    Genera todas las combinaciones con repeticion de columnas."""
    X = np.asarray(X)
    cols = [np.ones(len(X))]
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(X.shape[1]), d):
            cols.append(np.prod(X[:, combo], axis=1))
    return np.column_stack(cols)


def cd_fit(X_legit, degree, lam=1e-6):
    """Ajusta la matriz de momentos empirica M = (1/N) D^T D con
    regularizacion Tikhonov M_reg = M + lam*I. Devuelve (M_inv, D)."""
    D = phi(X_legit, degree)
    M = (D.T @ D) / len(D)
    M_inv = np.linalg.inv(M + lam * np.eye(len(M)))
    return M_inv, D


def cd_score(X, M_inv, degree):
    """Densidad Christoffel-Darboux p_CD(x) = phi(x)^T M^-1 phi(x).
    Valores bajos = regiones pobladas; altos = anomalos."""
    P = phi(X, degree)
    return np.einsum("ij,jk,ik->i", P, M_inv, P)


def niveles(valores_legit, ns=(50, 75, 90, 95, 99)):
    """Niveles lambda_n (o radios r_n) de los valores de referencia."""
    return {n: radio_n(valores_legit, n) for n in ns}


def evaluate(scores, y_true, niveles_dict):
    """Matriz de confusion en un renglon [TN, FP, FN, TP] por nivel:
    score > nivel se clasifica como anomalia (fraude)."""
    y_true = np.asarray(y_true, dtype=bool)
    rows = {}
    for n, level in niveles_dict.items():
        tn, fp, fn, tp = confusion_matrix(y_true, scores > level).ravel()
        rows[n] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("umbral n%")
