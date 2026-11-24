from orion.reverie.semantic_lift import infra_vocabulary_hit


def test_machinery_words_hit() -> None:
    for text in ("the coalition keeps winning", "an open loop on the substrate", "attended node n-1",
                 "Stability Score dropped", "projection_id 7"):
        assert infra_vocabulary_hit(text), text


def test_material_words_do_not_hit() -> None:
    for text in ("Why did nobody pass the walkway between 07:00 and 08:00?",
                 "Do long turns happen more than I bet they will?", "", None):
        assert not infra_vocabulary_hit(text), text  # type: ignore[arg-type]
