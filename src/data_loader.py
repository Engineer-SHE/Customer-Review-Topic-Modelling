"""Utilities for loading review datasets."""

from __future__ import annotations

import pandas as pd

def load_reviews(csv_path: str) -> pd.DataFrame:
    """Load review data from a CSV file.

    Args:
        csv_path: Path to the CSV file containing review data.

    Returns:
        DataFrame with the review content. Expects a ``review_text`` column.

    Raises:
        ValueError: If the expected ``review_text`` column is missing.
    """

    df = pd.read_csv(csv_path)
    if "review_text" not in df.columns:
        raise ValueError("Input dataset must include a 'review_text' column")
    return df
