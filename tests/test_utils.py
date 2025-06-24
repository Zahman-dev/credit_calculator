"""test_utils.py
Unit tests for utility functions.
"""
from __future__ import annotations

import numpy as np

from src.utils import calculate_business_metrics


def test_calculate_business_metrics_basic() -> None:
    """Verify metric calculations on a simple confusion-matrix scenario."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])

    metrics = calculate_business_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.5
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 1 