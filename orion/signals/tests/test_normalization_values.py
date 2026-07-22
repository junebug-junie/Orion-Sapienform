from orion.signals.normalization import NormalizationContext


def test_get_value_defaults_to_none():
    ctx = NormalizationContext()
    assert ctx.get_value("chat_stance", "missing") is None


def test_set_then_get_value_roundtrips():
    ctx = NormalizationContext()
    ctx.set_value("chat_stance", "embedding:sess-1", [0.1, 0.2])
    assert ctx.get_value("chat_stance", "embedding:sess-1") == [0.1, 0.2]


def test_values_are_scoped_per_organ():
    ctx = NormalizationContext()
    ctx.set_value("chat_stance", "k", "a")
    ctx.set_value("other_organ", "k", "b")
    assert ctx.get_value("chat_stance", "k") == "a"
    assert ctx.get_value("other_organ", "k") == "b"


def test_delete_value_removes_key():
    ctx = NormalizationContext()
    ctx.set_value("chat_stance", "embedding:sess-1", [0.1, 0.2])
    ctx.delete_value("chat_stance", "embedding:sess-1")
    assert ctx.get_value("chat_stance", "embedding:sess-1") is None


def test_delete_value_on_missing_key_is_a_no_op():
    ctx = NormalizationContext()
    ctx.delete_value("chat_stance", "never-set")  # must not raise
