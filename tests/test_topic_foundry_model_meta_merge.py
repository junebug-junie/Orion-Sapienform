import sys

sys.path.insert(0, "services/orion-topic-foundry")

from app.services.training import _compose_model_meta


def test_compose_model_meta_includes_model_spec_params_and_run_params():
    model_row = {
        "model_meta": {"representation": "ctfidf", "vectorizer_stop_words": "english"},
        "model_spec": {
            "params": {
                "vectorizer_max_df": 0.7,
                "top_n_words": 9,
            }
        },
    }

    merged = _compose_model_meta(
        model_row=model_row,
        run_model_meta={"representation": "mmr", "hdbscan_min_samples": 3},
        run_model_params={"vectorizer_min_df": 2, "top_n_words": 11},
        mode_params={"seed_topic_list": [["foo", "bar"]]},
    )

    assert merged["representation"] == "mmr"
    assert merged["vectorizer_stop_words"] == "english"
    assert merged["vectorizer_max_df"] == 0.7
    assert merged["vectorizer_min_df"] == 2
    assert merged["top_n_words"] == 11
    assert merged["seed_topic_list"] == [["foo", "bar"]]
