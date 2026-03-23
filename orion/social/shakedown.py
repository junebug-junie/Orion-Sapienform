from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from orion.schemas.social_scenario import SocialScenarioFixtureV1
from orion.schemas.social_shakedown import SocialShakedownFixV1, SocialShakedownIssueV1

from .scenario_replay import DEFAULT_SCENARIO_PACK, SocialScenarioReplayHarness, load_scenarios

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHAKEDOWN_PACK = ROOT / "tests" / "fixtures" / "social_room" / "shakedown_issues.json"


class SocialRoomShakedownWorkflow:
    def __init__(self, *, harness: SocialScenarioReplayHarness | None = None) -> None:
        self.harness = harness or SocialScenarioReplayHarness()

    def run(
        self,
        *,
        issues: Sequence[SocialShakedownIssueV1],
        fixes: Sequence[SocialShakedownFixV1],
        scenarios: Sequence[SocialScenarioFixtureV1] | None = None,
    ) -> dict[str, object]:
        scenario_list = list(scenarios or load_scenarios(DEFAULT_SCENARIO_PACK))
        scenario_map = {item.scenario_id: item for item in scenario_list}
        results: list[dict[str, object]] = []
        verified = 0
        missing_links: list[str] = []

        for issue in issues:
            linked_fixes = [item for item in fixes if item.issue_id == issue.issue_id]
            scenario_id = issue.linked_regression_scenario or issue.scenario_id
            scenario = scenario_map.get(scenario_id or "") if scenario_id else None
            evaluation = self.harness.run_scenario(scenario) if scenario is not None else None
            if scenario_id and scenario is None:
                missing_links.append(issue.issue_id)
            issue_verified = bool(evaluation and evaluation.passed and linked_fixes and all(item.status in {"tuned", "verified"} for item in linked_fixes))
            if issue_verified:
                verified += 1
            results.append(
                {
                    "issue": issue.model_dump(mode="json"),
                    "fixes": [item.model_dump(mode="json") for item in linked_fixes],
                    "evaluation": evaluation.model_dump(mode="json") if evaluation is not None else None,
                    "verified": issue_verified,
                }
            )

        return {
            "summary": {
                "issue_count": len(issues),
                "verified_count": verified,
                "open_issue_ids": [item.issue_id for item in issues if item.fix_status == "open"],
                "missing_regression_links": missing_links,
            },
            "results": results,
        }


def load_shakedown_pack(
    path: Path | str = DEFAULT_SHAKEDOWN_PACK,
    *,
    only_issue_ids: Iterable[str] | None = None,
) -> tuple[list[SocialShakedownIssueV1], list[SocialShakedownFixV1]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    issues = [SocialShakedownIssueV1.model_validate(item) for item in raw.get("issues") or []]
    fixes = [SocialShakedownFixV1.model_validate(item) for item in raw.get("fixes") or []]
    if only_issue_ids is None:
        return issues, fixes
    allowed = set(only_issue_ids)
    filtered_issues = [item for item in issues if item.issue_id in allowed]
    filtered_issue_ids = {item.issue_id for item in filtered_issues}
    filtered_fixes = [item for item in fixes if item.issue_id in filtered_issue_ids]
    return filtered_issues, filtered_fixes
