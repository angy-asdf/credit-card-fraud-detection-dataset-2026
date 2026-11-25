from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import typer

from credit_card_fraud_detection_dataset_2026.config import (
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

app = typer.Typer()

N_COMPONENTS = 10
RANDOM_STATE = 42


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "credit_card_fraud_detection_2026.csv",
    output_path: Path = INTERIM_DATA_DIR / "mexico_numeric.csv",
):
    """Filter transactions in Mexico and keep numeric columns plus is_fraud."""
    logger.info(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)

    df_mx = df[df["merchant_country"] == "Mexico"]
    df_numeric = df_mx.select_dtypes(include="number")  # is_fraud (int64) is included

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_numeric.to_csv(output_path, index=False)
    logger.success(f"{len(df_numeric):,} rows x {df_numeric.shape[1]} cols -> {output_path}")

    # Paso 3: split 80/20
    X_train, X_test = train_test_split(df_numeric, test_size=0.2, random_state=RANDOM_STATE)
    logger.info(f"Split 80/20 -> X_train {X_train.shape}, X_test {X_test.shape}")

    # Paso 4: StandardScaler + PCA fiteados en train (is_fraud va junto, no dentro)
    feature_cols = [c for c in X_train.columns if c != "is_fraud"]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])

    pca = PCA(n_components=N_COMPONENTS)
    pc_cols = [f"PC{i + 1}" for i in range(N_COMPONENTS)]
    X_train_pca = pd.DataFrame(
        pca.fit_transform(X_train_scaled), columns=pc_cols, index=X_train.index
    )
    X_train_pca["is_fraud"] = X_train["is_fraud"]
    logger.success(
        f"PCA {N_COMPONENTS} comps fiteado en train "
        f"({pca.explained_variance_ratio_.sum():.1%} varianza explicada)"
    )

    # Paso 5: transform en test con los parámetros aprendidos en train
    X_test_pca = pd.DataFrame(
        pca.transform(scaler.transform(X_test[feature_cols])),
        columns=pc_cols,
        index=X_test.index,
    )
    X_test_pca["is_fraud"] = X_test["is_fraud"]

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train_pca.to_csv(PROCESSED_DATA_DIR / "x_train_pca.csv")
    X_test_pca.to_csv(PROCESSED_DATA_DIR / "x_test_pca.csv")
    logger.success(f"X_train_pca {X_train_pca.shape}, X_test_pca {X_test_pca.shape} -> {PROCESSED_DATA_DIR}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(pca, MODELS_DIR / "pca.joblib")
    logger.success(f"scaler.joblib, pca.joblib -> {MODELS_DIR}")


if __name__ == "__main__":
    app()
