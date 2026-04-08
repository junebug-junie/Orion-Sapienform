__all__ = ["AutonomyVerificationHarness", "GraphDBClient", "ScenarioFixture", "load_scenarios", "write_report"]


def __getattr__(name: str):
    if name in set(__all__):
        from .verification import (
            AutonomyVerificationHarness,
            GraphDBClient,
            ScenarioFixture,
            load_scenarios,
            write_report,
        )

        mapping = {
            "AutonomyVerificationHarness": AutonomyVerificationHarness,
            "GraphDBClient": GraphDBClient,
            "ScenarioFixture": ScenarioFixture,
            "load_scenarios": load_scenarios,
            "write_report": write_report,
        }
        return mapping[name]
    raise AttributeError(name)
