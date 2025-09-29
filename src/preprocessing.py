"""Text preprocessing utilities."""

from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_resource(resource: str) -> None:
    """Ensure the requested NLTK resource is available."""

    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1])


# Guarantee required corpora are available when the module is imported.
_ensure_nltk_resource("corpora/stopwords")
_ensure_nltk_resource("corpora/wordnet")

_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS = set(stopwords.words("english"))

_CLEAN_PATTERN = re.compile(r"[^a-zA-Z\s]")


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, and lemmatize a text string."""

    text = text.lower()
    text = _CLEAN_PATTERN.sub(" ", text)
    tokens = [
        _LEMMATIZER.lemmatize(token)
        for token in text.split()
        if token not in _STOP_WORDS and len(token) > 2
    ]
    return " ".join(tokens)


def preprocess_reviews(reviews: Iterable[str]) -> List[str]:
    """Preprocess an iterable of review texts."""

    return [clean_text(review) for review in reviews]


def add_clean_text_column(df: pd.DataFrame, source_column: str = "review_text") -> pd.DataFrame:
    """Add a ``clean_text`` column to the DataFrame."""

    df = df.copy()
    df["clean_text"] = preprocess_reviews(df[source_column].fillna(""))
    return df
