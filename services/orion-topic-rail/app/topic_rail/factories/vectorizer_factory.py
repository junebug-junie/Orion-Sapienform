from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer

from app.settings import Settings


def build_vectorizer(settings: Settings) -> CountVectorizer:
    return CountVectorizer(
        ngram_range=settings.ngram_range,
        stop_words=settings.stopwords,
        min_df=2,
        max_df=0.95,
    )
