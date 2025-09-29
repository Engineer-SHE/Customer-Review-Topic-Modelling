"""Unit tests for text preprocessing utilities."""

from __future__ import annotations

from preprocessing import clean_text, preprocess_reviews


def test_clean_text_removes_punctuation_and_stopwords() -> None:
    text = "The Shipping!!! was FAST and AMAZING???"
    cleaned = clean_text(text)
    assert "the" not in cleaned
    assert "shipping" in cleaned
    assert "fast" in cleaned
    assert "amazing" in cleaned
    assert "!" not in cleaned


def test_preprocess_reviews_handles_multiple_entries() -> None:
    reviews = ["Great taste", "Bad packaging"]
    processed = preprocess_reviews(reviews)
    assert len(processed) == 2
    assert processed[0] != processed[1]
