# Customer Review Topic Modelling

Discover common themes in customer feedback using unsupervised learning. This project provides a lightweight pipeline to clean product or service reviews, train a Latent Dirichlet Allocation (LDA) model, and inspect the resulting topics.

## Project Structure

```
.
├── data/
│   └── sample_reviews.csv   # Example dataset of Amazon/Yelp reviews
├── src/
│   ├── data_loader.py       # CSV loading utilities
│   ├── preprocessing.py     # Text cleaning helpers
│   ├── topic_model.py       # Topic modelling abstractions
│   ├── visualization.py     # Utilities for presenting topic outputs
│   └── main.py              # Command line entry point
├── tests/
│   └── test_preprocessing.py
├── README.md
├── requirements.txt
└── LICENSE
```

## Getting Started

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   python -m nltk.downloader stopwords wordnet
   ```

3. Run the topic modeller on the sample dataset:

   ```bash
   python -m src.main data/sample_reviews.csv --topics 3 --top-words 8
   ```

   Example output:

   ```
   Topic 1: shipping, fast, packaging, great, taste, solid
   Topic 2: delicious, desserts, ambiance, brunch, fantastic, fresh
   Topic 3: wait, drinks, staff, service, slow, table
   ```

4. Plot topic distributions (optional):

   ```python
   from data_loader import load_reviews
   from preprocessing import add_clean_text_column
   from topic_model import TopicModeler
   from visualization import plot_topic_distributions

   df = add_clean_text_column(load_reviews("data/sample_reviews.csv"))
   modeler = TopicModeler()
   distributions = modeler.fit_transform(df["clean_text"].tolist())
   plot_topic_distributions(distributions)
   ```

## Testing

Run unit tests with:

```bash
pytest
```

## Extending the Project

- Swap the sample dataset with your own CSV that contains at least a `review_text` column.
- Experiment with different numbers of topics, vectorizers, or preprocessing rules.
- Integrate other visualizations like pyLDAvis for interactive topic exploration.
