"""Helpers for presenting topic modelling results."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def format_topics(topics: Sequence[Sequence[str]]) -> List[str]:
    """Format topics for display in the console."""

    formatted = []
    for idx, words in enumerate(topics, start=1):
        formatted.append(f"Topic {idx}: {', '.join(words)}")
    return formatted


def plot_topic_distributions(distributions: np.ndarray, title: str = "Topic Distribution per Document") -> None:
    """Plot stacked bar chart of topic distributions."""

    n_docs, n_topics = distributions.shape
    ind = np.arange(n_docs)
    bottom = np.zeros(n_docs)

    plt.figure(figsize=(10, 6))
    for topic_idx in range(n_topics):
        plt.bar(ind, distributions[:, topic_idx], bottom=bottom, label=f"Topic {topic_idx + 1}")
        bottom += distributions[:, topic_idx]

    plt.xlabel("Document")
    plt.ylabel("Topic Weight")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def print_topics(formatted_topics: Iterable[str]) -> None:
    """Print formatted topics to stdout."""

    for line in formatted_topics:
        print(line)
