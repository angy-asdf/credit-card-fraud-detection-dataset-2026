from datetime import datetime, timezone
from pathlib import Path
import shutil

import kagglehub
from kagglehub.clients import build_kaggle_client
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from loguru import logger
import pandas as pd
import typer

from credit_card_fraud_detection_dataset_2026.config import PROJ_ROOT, RAW_DATA_DIR

app = typer.Typer()

KAGGLE_DATASET = "mohankrishnathalla/credit-card-fraud-detection-dataset-2026"
KAGGLE_OWNER, KAGGLE_SLUG = KAGGLE_DATASET.split("/")
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
REFERENCES_DIR = PROJ_ROOT / "references"


def download_raw_data(force: bool = False) -> None:
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


def fetch_dataset_metadata() -> tuple[str, str, int]:
    """Fetch the dataset title, markdown description and version via the Kaggle API."""
    request = ApiGetDatasetRequest()
    request.owner_slug = KAGGLE_OWNER
    request.dataset_slug = KAGGLE_SLUG
    dataset = build_kaggle_client().datasets.dataset_api_client.get_dataset(request)
    return dataset.title, dataset.description, dataset.current_version_number


def parse_feature_groups(description: str) -> dict[str, str]:
    """Parse the '## Feature Groups' markdown table into {column: group}."""
    groups = {}
    in_table = False
    for line in description.splitlines():
        if line.startswith("## "):
            if in_table:
                break
            in_table = line.strip() == "## Feature Groups"
            continue
        if not in_table:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] in ("Group", "") or set(cells[0]) <= {"-", ":"}:
            continue
        for col in cells[1].split(","):
            col = col.strip()
            if col:
                groups[col] = cells[0]
    return groups


def parse_target_descriptions(description: str) -> dict[str, str]:
    """Parse '### N. <column> (...)' blocks under '## Target Variables' into {column: text}."""
    targets = {}
    in_section = False
    current = None
    for line in description.splitlines():
        if line.startswith("## "):
            in_section = line.strip() == "## Target Variables"
            current = None
            continue
        if not in_section:
            continue
        if line.startswith("### "):
            current = line[4:].split(" ", 2)[1].strip()
            targets[current] = []
        elif current and line.strip().startswith("-"):
            targets[current].append(line.strip().lstrip("- ").replace("**", ""))
    return {col: "; ".join(parts) for col, parts in targets.items()}


def build_data_dictionary(
    df: pd.DataFrame, groups: dict[str, str], target_descs: dict[str, str]
) -> pd.DataFrame:
    """Combine source metadata (groups, target descriptions) with data profiling."""
    rows = []
    for col in df.columns:
        series = df[col]
        row = {
            "column": col,
            "group": groups.get(col, ""),
            "dtype": str(series.dtype),
            "description": target_descs.get(col, groups.get(col, "")),
            "non_null": int(series.notna().sum()),
            "n_unique": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            row["min"] = series.min()
            row["max"] = series.max()
            row["mean"] = round(series.mean(), 4)
        else:
            row["top_values"] = ", ".join(
                str(v) for v in series.value_counts().head(3).index.tolist()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row.tolist()) + " |")
    return "\n".join(lines)


def write_data_dictionary(dict_df: pd.DataFrame, title: str, version: int) -> None:
    REFERENCES_DIR.mkdir(exist_ok=True)
    csv_path = REFERENCES_DIR / "data_dictionary.csv"
    dict_df.to_csv(csv_path, index=False)

    md_path = REFERENCES_DIR / "data_dictionary.md"
    header = (
        f"# Data Dictionary — {title}\n\n"
        f"- Source: [{KAGGLE_DATASET}]({KAGGLE_URL})\n"
        f"- Dataset version: {version}\n"
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        "by `credit_card_fraud_detection_dataset_2026/dataset.py`\n\n"
    )
    md_path.write_text(header + _md_table(dict_df) + "\n", encoding="utf-8")
    logger.success(f"Data dictionary saved -> {csv_path.name}, {md_path.name}")


@app.command()
def main(force: bool = False):
    download_raw_data(force)

    raw_csvs = sorted(RAW_DATA_DIR.glob("*.csv"))
    if not raw_csvs:
        logger.error("No raw CSV found to build the data dictionary.")
        raise typer.Exit(code=1)

    logger.info("Fetching dataset metadata from Kaggle...")
    title, description, version = fetch_dataset_metadata()

    df = pd.read_csv(raw_csvs[0])
    dict_df = build_data_dictionary(
        df, parse_feature_groups(description), parse_target_descriptions(description)
    )
    write_data_dictionary(dict_df, title, version)


if __name__ == "__main__":
    app()
