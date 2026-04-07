import sys

sys.path.insert(0, "services/orion-topic-foundry")

from app.topic_engine import _build_vectorizer


def test_build_vectorizer_english_stop_words_and_extras_are_applied():
    vectorizer = _build_vectorizer(
        {
            "vectorizer_stop_words": "english",
            "stop_words_extra": "orion,juniper",
            "vectorizer_min_df": 1,
            "vectorizer_max_df": 0.9,
            "vectorizer_max_features": 2500,
            "vectorizer_ngram_min": 1,
            "vectorizer_ngram_max": 2,
        }
    )

    params = vectorizer.get_params()
    stop_words = params["stop_words"]
    assert isinstance(stop_words, list)
    assert "the" in stop_words
    assert "orion" in stop_words
    assert "juniper" in stop_words
    assert params["ngram_range"] == (1, 2)
    assert params["max_df"] == 0.9
    assert params["min_df"] == 1
    assert params["max_features"] == 2500
