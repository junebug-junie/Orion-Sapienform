from __future__ import annotations


def assert_fields_within_budget(fields: dict[str, tuple[str, int]]) -> None:
    """Raise ``compactor_output_over_budget:<field>`` when any value exceeds max chars."""
    for name, (value, max_chars) in fields.items():
        if len(value or "") > int(max_chars):
            raise ValueError(f"compactor_output_over_budget:{name}")
