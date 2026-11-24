from pathlib import Path
import shutil

import kagglehub
from loguru import logger
import typer

from credit_card_fraud_detection_dataset_2026.config import RAW_DATA_DIR

app = typer.Typer()

KAGGLE_DATASET = "mohankrishnathalla/credit-card-fraud-detection-dataset-2026"


@app.command()
def main(force: bool = False):
    existing = list(RAW_DATA_DIR.glob("*.csv"))
    if existing and not force:
        logger.info(
            f"Raw data already present: {[f.name for f in existing]}. Use --force to re-download."
        )
        return

    logger.info(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    cache_path = kagglehub.dataset_download(KAGGLE_DATASET)

    csv_files = list(Path(cache_path).rglob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in downloaded dataset at {cache_path}")
        raise typer.Exit(code=1)

    for csv_file in csv_files:
        dest = RAW_DATA_DIR / csv_file.name
        shutil.copy(csv_file, dest)
        logger.success(f"Saved {csv_file.name} -> {dest}")


if __name__ == "__main__":
    app()
