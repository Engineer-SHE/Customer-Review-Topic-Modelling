"""Topic modelling pipeline using scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


@dataclass
class TopicModelConfig:
    """Configuration options for the topic model."""

    n_topics: int = 5
    max_features: int = 1000
    max_df: float = 0.95
    min_df: int = 2
    use_tfidf: bool = False


class TopicModeler:
    """Fit LDA topic models to review corpora."""

    def __init__(self, config: TopicModelConfig | None = None) -> None:
        self.config = config or TopicModelConfig()
        self.vectorizer: CountVectorizer | TfidfVectorizer | None = None
        self.model: LatentDirichletAllocation | None = None

    def _build_vectorizer(self) -> CountVectorizer | TfidfVectorizer:
        if self.config.use_tfidf:
            return TfidfVectorizer(
                max_df=self.config.max_df,
                min_df=self.config.min_df,
                max_features=self.config.max_features,
            )
        return CountVectorizer(
            max_df=self.config.max_df,
            min_df=self.config.min_df,
            max_features=self.config.max_features,
        )

    def fit(self, documents: List[str]) -> None:
        """Fit the topic model on the documents."""

        self.vectorizer = self._build_vectorizer()
        matrix = self.vectorizer.fit_transform(documents)
        self.model = LatentDirichletAllocation(
            n_components=self.config.n_topics,
            random_state=42,
            learning_method="batch",
        )
        self.model.fit(matrix)

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents into topic distributions."""

        if not self.model or not self.vectorizer:
            raise RuntimeError("Model must be fitted before calling transform().")
        matrix = self.vectorizer.transform(documents)
        return self.model.transform(matrix)

    def get_top_words(self, n_words: int = 10) -> List[List[str]]:
        """Return the top ``n_words`` for each topic."""

        if not self.model or not self.vectorizer:
            raise RuntimeError("Model must be fitted before retrieving topics.")
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        topics: List[List[str]] = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            topics.append(feature_names[top_indices].tolist())
        return topics

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit the model and return topic distributions."""

        self.fit(documents)
        return self.transform(documents)

    def describe_topics(self, n_words: int = 10) -> List[Tuple[int, List[str]]]:
        """Convenience wrapper returning topic indices with top words."""

        topics = self.get_top_words(n_words=n_words)
        return list(enumerate(topics, start=1))
