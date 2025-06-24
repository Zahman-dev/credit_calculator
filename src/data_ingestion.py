"""data_ingestion.py
====================
Merkezi veri indirme ve hazırlama modülü.
* Almanya Kredi veri setini kaynak URL'den indirir.
* CSV formatına dönüştürür.
* Basit şema ve tip kontrolleri yapar.
* Başarılı olduğunda final CSV yolunu döner.

Bu modül tekrar çalıştırıldığında mevcut dosyaları atlar (idempotent). `force=True` argümanı ile yeniden indirme/işleme yapılabilir.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd
import requests

from .config import DATA_DIR, ALL_FEATURES, TARGET_COLUMN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
RAW_FILENAME = "german.data"
CSV_FILENAME = "german_credit_data.csv"
SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)


class DataIngestionError(Exception):
    """Raised when data ingestion fails."""


def download_raw_dataset(force: bool = False, data_dir: Path = DATA_DIR) -> Path:
    """Download the raw german.data file from the UCI repository.

    Parameters
    ----------
    force : bool, optional
        If ``True`` download the file even if it already exists, by default ``False``.
    data_dir : Path, optional
        Target directory, by default ``DATA_DIR`` from project config.

    Returns
    -------
    Path
        Path to the downloaded raw file.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / RAW_FILENAME

    if raw_path.exists() and not force:
        logger.info("Raw dataset already exists at %s", raw_path)
        return raw_path

    logger.info("Downloading raw dataset from %s", SOURCE_URL)
    try:
        response = requests.get(SOURCE_URL, timeout=60)
        response.raise_for_status()
        raw_path.write_bytes(response.content)
        logger.info("Raw dataset downloaded to %s (%.2f KB)", raw_path, raw_path.stat().st_size / 1024)
    except requests.RequestException as exc:
        raise DataIngestionError(f"Failed to download dataset: {exc}") from exc

    return raw_path


def parse_and_validate(raw_path: Path) -> pd.DataFrame:
    """Parse the raw file into a DataFrame and run basic validations."""
    column_names: List[str] = ALL_FEATURES + [TARGET_COLUMN]

    try:
        # The raw file is space-separated with no header.
        df = pd.read_csv(raw_path, sep=" ", header=None, names=column_names, engine="python")
    except Exception as exc:
        raise DataIngestionError(f"Failed to parse raw dataset: {exc}") from exc

    # Convert numeric columns to appropriate dtypes
    numeric_cols = [
        "Duration",
        "Credit_amount",
        "Age",
        "Installment_rate",
        "Present_residence",
        "Existing_credits",
        "Dependents",
        "Risk",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Basic validation checks
    if df.isnull().values.any():
        missing = int(df.isnull().sum().sum())
        logger.warning("Dataset contains %d missing values. They will need handling during preprocessing.", missing)

    expected_rows = 1000  # As per UCI description
    if len(df) != expected_rows:
        logger.warning("Expected %d rows but parsed %d rows", expected_rows, len(df))

    if not set(df.columns) == set(column_names):
        raise DataIngestionError("Column mismatch after parsing. Validation failed.")

    return df


def save_dataframe(df: pd.DataFrame, data_dir: Path = DATA_DIR) -> Path:
    """Save DataFrame to CSV and return the path."""
    csv_path = data_dir / CSV_FILENAME
    df.to_csv(csv_path, index=False)
    logger.info("Processed dataset saved to %s", csv_path)
    return csv_path


def download_and_prepare_german_credit_dataset(force: bool = False, data_dir: Path = DATA_DIR) -> Path:
    """High-level helper to ensure dataset is available locally as CSV.

    This is the main function that should be imported by other modules.
    """
    csv_path = data_dir / CSV_FILENAME
    if csv_path.exists() and not force:
        logger.info("Prepared CSV dataset already exists at %s", csv_path)
        return csv_path

    raw_path = download_raw_dataset(force=force, data_dir=data_dir)
    df = parse_and_validate(raw_path)
    return save_dataframe(df, data_dir=data_dir)


if __name__ == "__main__":
    try:
        path = download_and_prepare_german_credit_dataset()
        print(f"🎉 Dataset ready at {path}")
    except DataIngestionError as err:
        print(f"❌ Data ingestion failed: {err}") 