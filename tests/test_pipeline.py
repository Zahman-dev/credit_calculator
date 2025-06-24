"""test_pipeline.py
Tests for the preprocessing pipeline.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.pipeline import DataPreprocessor
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def _sample_dataframe() -> pd.DataFrame:
    """Return a minimal DataFrame with required columns."""
    data: dict[str, list] = {
        "Duration": [12, 24],
        "Credit_amount": [5000, 10000],
        "Age": [35, 45],
        "Installment_rate": [2, 4],
        "Present_residence": [1, 3],
        "Existing_credits": [1, 2],
        "Dependents": [1, 1],
        "Checking_account": ["A11", "A12"],
        "Credit_history": ["A34", "A33"],
        "Purpose": ["A43", "A42"],
        "Savings_account": ["A61", "A62"],
        "Employment": ["A73", "A74"],
        "Personal_status_sex": ["A93", "A92"],
        "Other_debtors": ["A101", "A102"],
        "Property": ["A121", "A122"],
        "Other_installment_plans": ["A143", "A141"],
        "Housing": ["A152", "A151"],
        "Job": ["A173", "A172"],
        "Telephone": ["A192", "A191"],
        "Foreign_worker": ["A201", "A202"],
    }
    return pd.DataFrame(data)


def test_preprocessor_shape_and_inverse() -> None:
    df = _sample_dataframe()
    y = np.array([0, 1])

    pre = DataPreprocessor()
    X_trans = pre.fit_transform(df, y)

    # Expect preserved sample size
    assert X_trans.shape[0] == df.shape[0]  # type: ignore[index]

    # After one-hot encoding categorical variables, feature count > original columns
    assert X_trans.shape[1] > len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)  # type: ignore[index]

    # Test inverse transform of target
    assert pre.inverse_transform_target(np.array([0, 1])).tolist() == ["Good", "Bad"] 