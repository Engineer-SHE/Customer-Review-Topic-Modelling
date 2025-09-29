"""Command line entry point for topic modelling over review data."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import load_reviews
from preprocessing import add_clean_text_column
from topic_model import TopicModelConfig, TopicModeler
from visualization import format_topics, print_topics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer Review Topic Modelling")
    parser.add_argument("dataset", type=Path, help="Path to the CSV file containing reviews.")
    parser.add_argument("--topics", type=int, default=5, help="Number of topics to extract.")
    parser.add_argument(
        "--use-tfidf",
        action="store_true",
        help="Use TF-IDF vectorization instead of raw counts.",
    )
    parser.add_argument("--max-features", type=int, default=1000, help="Maximum vocabulary size.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for tokens.")
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.95,
        help="Ignore terms that appear in more than this proportion of documents.",
    )
    parser.add_argument(
        "--top-words",
        type=int,
        default=10,
        help="Number of keywords to show for each topic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_reviews(str(args.dataset))
    df = add_clean_text_column(df)

    config = TopicModelConfig(
        n_topics=args.topics,
        max_features=args.max_features,
        max_df=args.max_df,
        min_df=args.min_df,
        use_tfidf=args.use_tfidf,
    )

    modeler = TopicModeler(config)
    modeler.fit(df["clean_text"].tolist())

    formatted = format_topics(modeler.get_top_words(n_words=args.top_words))
    print_topics(formatted)


if __name__ == "__main__":
    main()
