from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from credit_card_fraud_detection_dataset_2026.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


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


if __name__ == "__main__":
    app()
