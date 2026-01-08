from __future__ import annotations

from typing import Dict, Type

from .base import BaseVerb


class VerbRegistry:
    def __init__(self) -> None:
        self._verbs: Dict[str, Type[BaseVerb]] = {}

    def register(self, trigger: str, verb_cls: Type[BaseVerb]) -> None:
        if trigger in self._verbs:
            raise ValueError(f"Verb already registered: {trigger}")
        self._verbs[trigger] = verb_cls

    def get(self, trigger: str) -> Type[BaseVerb] | None:
        return self._verbs.get(trigger)

    def all(self) -> Dict[str, Type[BaseVerb]]:
        return dict(self._verbs)


registry = VerbRegistry()


def verb(trigger: str):
    def decorator(cls: Type[BaseVerb]) -> Type[BaseVerb]:
        registry.register(trigger, cls)
        setattr(cls, "trigger", trigger)
        return cls

    return decorator
