"""test_data_ingestion.py
Unit tests for the src.data_ingestion module.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_ingestion import parse_and_validate
from src.config import ALL_FEATURES, TARGET_COLUMN


def _build_sample_row() -> list[str]:
    """Build a single synthetic raw-data row matching German dataset schema."""
    return [
        "A11",  # Checking_account (cat)
        "12",  # Duration (num)
        "A34",  # Credit_history (cat)
        "A43",  # Purpose (cat)
        "5000",  # Credit_amount (num)
        "A61",  # Savings_account (cat)
        "A73",  # Employment (cat)
        "3",  # Installment_rate (num)
        "A93",  # Personal_status_sex (cat)
        "A101",  # Other_debtors (cat)
        "2",  # Present_residence (num)
        "A121",  # Property (cat)
        "35",  # Age (num)
        "A143",  # Other_installment_plans (cat)
        "A152",  # Housing (cat)
        "1",  # Existing_credits (num)
        "A173",  # Job (cat)
        "1",  # Dependents (num)
        "A192",  # Telephone (cat)
        "A201",  # Foreign_worker (cat)
        "1",  # Risk (target)
    ]


def test_parse_and_validate_with_small_sample(tmp_path: Path) -> None:
    """parse_and_validate should correctly parse a minimal sample file."""
    # Build dummy raw dataset with two rows
    rows = [_build_sample_row(), _build_sample_row()]
    raw_file = tmp_path / "sample.data"
    raw_file.write_text("\n".join(" ".join(r) for r in rows) + "\n")

    df = parse_and_validate(raw_file)

    # Basic structural assertions
    expected_cols = ALL_FEATURES + [TARGET_COLUMN]
    assert list(df.columns) == expected_cols
    assert len(df) == 2

    # Numeric columns converted to numeric dtype
    assert pd.api.types.is_numeric_dtype(df["Duration"])  # type: ignore[arg-type, attr-defined]
    assert pd.api.types.is_numeric_dtype(df["Credit_amount"])  # type: ignore[arg-type, attr-defined]
