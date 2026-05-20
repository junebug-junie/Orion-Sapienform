from __future__ import annotations

from pathlib import Path

import pytest

from orion.knowledge_forge.review import apply_pending_patch, list_pending_patches


def test_list_pending_patches_finds_fixture(tmp_path: Path) -> None:
    pending = tmp_path / "reviews" / "pending"
    pending.mkdir(parents=True)
    patch = pending / "review-test-001.patch.md"
    patch.write_text(
        """---
patch_id: review:test:001
target: specs/design/test.yaml
action: create
status: pending
---

```yaml
type: spec
id: spec:test:001
status: draft
component: test
requirements: []
```
""",
        encoding="utf-8",
    )
    found = list_pending_patches(tmp_path)
    assert len(found) == 1
    assert found[0].patch_id == "review:test:001"


def test_apply_pending_patch_writes_target(tmp_path: Path) -> None:
    pending = tmp_path / "reviews" / "pending"
    accepted = tmp_path / "reviews" / "accepted"
    pending.mkdir(parents=True)
    accepted.mkdir(parents=True)
    patch_path = pending / "review-test-002.patch.md"
    patch_path.write_text(
        """---
patch_id: review:test:002
target: specs/design/test.yaml
action: create
status: pending
---

```yaml
type: spec
id: spec:test:002
status: draft
component: test
requirements: ["one"]
non_goals: []
acceptance_tests: []
source_claims: []
```
""",
        encoding="utf-8",
    )
    apply_pending_patch(tmp_path, "review:test:002")
    target = tmp_path / "specs" / "design" / "test.yaml"
    assert target.is_file()
    assert not patch_path.exists()
    assert (accepted / "review-test-002.patch.md").is_file()


def test_apply_rejects_execution_ready_target(tmp_path: Path) -> None:
    pending = tmp_path / "reviews" / "pending"
    pending.mkdir(parents=True)
    patch_path = pending / "review-exec-ready.patch.md"
    patch_path.write_text(
        """---
patch_id: review:test:exec-ready
target: specs/execution_ready/evil.yaml
action: create
status: pending
---

```yaml
type: spec
id: spec:test:evil
status: execution_ready
component: test
requirements: ["x"]
acceptance_tests: ["y"]
```
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="execution_ready"):
        apply_pending_patch(tmp_path, "review:test:exec-ready")


def test_apply_rejects_target_outside_corpus(tmp_path: Path) -> None:
    pending = tmp_path / "reviews" / "pending"
    pending.mkdir(parents=True)
    patch_path = pending / "review-escape.patch.md"
    patch_path.write_text(
        """---
patch_id: review:test:escape
target: ../../outside.yaml
action: create
status: pending
---

```yaml
type: spec
id: spec:test:escape
status: draft
component: test
requirements: []
```
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes corpus"):
        apply_pending_patch(tmp_path, "review:test:escape")
