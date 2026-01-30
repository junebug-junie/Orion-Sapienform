import os
import sys
import types

SERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("TOPIC_RAIL_MODEL_VERSION", "topic-rail-test")

if "bertopic" not in sys.modules:
    bertopic_module = types.ModuleType("bertopic")

    class DummyBERTopic:  # minimal stub for import-time usage
        def __init__(self, *args, **kwargs):
            pass

        def save(self, *args, **kwargs):
            return None

        @classmethod
        def load(cls, *args, **kwargs):
            return cls()

    bertopic_module.BERTopic = DummyBERTopic
    sys.modules["bertopic"] = bertopic_module

if "hdbscan" not in sys.modules:
    hdbscan_module = types.ModuleType("hdbscan")

    class DummyHDBSCAN:  # minimal stub
        def __init__(self, *args, **kwargs):
            pass

    hdbscan_module.HDBSCAN = DummyHDBSCAN
    sys.modules["hdbscan"] = hdbscan_module

if "umap" not in sys.modules:
    umap_module = types.ModuleType("umap")

    class DummyUMAP:  # minimal stub
        def __init__(self, *args, **kwargs):
            pass

    umap_module.UMAP = DummyUMAP
    sys.modules["umap"] = umap_module

if "sklearn" not in sys.modules:
    sklearn_module = types.ModuleType("sklearn")
    feature_module = types.ModuleType("sklearn.feature_extraction")
    text_module = types.ModuleType("sklearn.feature_extraction.text")

    class DummyCountVectorizer:  # minimal stub
        def __init__(self, *args, **kwargs):
            pass

    text_module.CountVectorizer = DummyCountVectorizer
    feature_module.text = text_module
    sklearn_module.feature_extraction = feature_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.feature_extraction"] = feature_module
    sys.modules["sklearn.feature_extraction.text"] = text_module
