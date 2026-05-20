from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orion.knowledge_forge.api_compile import (
    _claim_included,
    build_context_pack_output_path,
    compile_context_pack_api_v1,
)
from orion.knowledge_forge.models import (
    ClaimV1,
    ContextPackTargetV1,
    ContextPackV1,
    DecisionV1,
    SourceV1,
    SpecV1,
)
from orion.knowledge_forge.review import apply_pending_patch, list_pending_patches
from orion.knowledge_forge.sources import ingest_source
from orion.knowledge_forge.store import KnowledgeStore, DocModel
from orion.knowledge_forge.yaml_doc import load_yaml_doc

from app.api_schemas import (
    ClaimSummaryV1,
    ContextPackCompileRequestV1,
    ContextPackCompileResultV1,
    ContextPackSummaryV1,
    DecisionSummaryV1,
    KnowledgeForgeStatusV1,
    ReviewSummaryV1,
    SearchHitV1,
    SourceIngestRequestV1,
    SourceIngestResultV1,
    SourceSummaryV1,
    SpecSummaryV1,
)
from app.settings import Settings

_MODEL_BY_TYPE: dict[str, type[DocModel]] = {
    "source": SourceV1,
    "claim": ClaimV1,
    "decision": DecisionV1,
    "spec": SpecV1,
    "context_pack": ContextPackV1,
}


class KnowledgeForgeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.knowledge_forge_repo_root
        self.store = KnowledgeStore(self.root)
        self.warnings: list[str] = []
        self.reload()

    def reload(self) -> None:
        self.warnings = []
        self.store.by_id.clear()
        self.store.paths_by_id.clear()
        if not self.root.is_dir():
            self.warnings.append(f"corpus root not found: {self.root}")
            return
        for path in sorted(self.root.rglob("*.yaml")):
            if "reviews/" in path.as_posix():
                continue
            try:
                doc = load_yaml_doc(path)
                typed = self._coerce(doc)
                if typed.id in self.store.by_id:
                    self.warnings.append(f"duplicate id {typed.id} in {path}")
                    continue
                self.store.by_id[typed.id] = typed
                self.store.paths_by_id[typed.id] = path
            except Exception as exc:  # noqa: BLE001 — malformed files become warnings
                self.warnings.append(f"{path}: {exc}")

    def status(self) -> KnowledgeForgeStatusV1:
        claims = self.store.claims()
        specs = self.store.specs()
        return KnowledgeForgeStatusV1(
            generated_at=datetime.now(timezone.utc),
            repo_root=str(self.root),
            source_count=len(self._sources()),
            claim_count=len(claims),
            accepted_claim_count=sum(1 for c in claims if c.status.value == "accepted"),
            disputed_claim_count=sum(1 for c in claims if c.status.value == "disputed"),
            stale_claim_count=sum(1 for c in claims if c.status.value == "stale"),
            spec_count=len(specs),
            execution_ready_spec_count=sum(1 for s in specs if s.status.value == "execution_ready"),
            decision_count=len(self._decisions()),
            pending_review_count=len(list_pending_patches(self.root)),
            context_pack_count=len(self._context_packs()),
            warnings=list(self.warnings),
            enabled=self.settings.knowledge_forge_enabled,
            write_enabled=self.settings.knowledge_forge_write_enabled,
        )

    def list_claims(self) -> list[ClaimSummaryV1]:
        return [self._claim_summary(claim) for claim in sorted(self.store.claims(), key=lambda c: c.id)]

    def search_claims(self, query: str) -> list[ClaimSummaryV1]:
        return [
            self._claim_summary(hit)
            for hit in self._search_claim_docs(query)
        ]

    def search(self, query: str) -> list[SearchHitV1]:
        needle = query.casefold()
        hits: list[SearchHitV1] = []
        for claim in self._search_claim_docs(query):
            hits.append(
                SearchHitV1(
                    kind="claim",
                    id=claim.id,
                    label=claim.statement,
                    status=claim.status.value,
                    path=self._rel_path(claim.id),
                    score=self._score_text(needle, claim.id, claim.statement, self._rel_path(claim.id)),
                )
            )
        for spec in self.store.specs():
            title = _spec_title(spec)
            body = " ".join(spec.requirements)
            path = self._rel_path(spec.id)
            score = self._score_text(needle, spec.id, title, spec.component, body, path)
            if score > 0:
                hits.append(
                    SearchHitV1(
                        kind="spec",
                        id=spec.id,
                        label=title,
                        status=spec.status.value,
                        path=path,
                        score=score,
                    )
                )
        for source in self._sources():
            score = self._score_text(needle, source.id, source.path)
            if score > 0:
                hits.append(
                    SearchHitV1(
                        kind="source",
                        id=source.id,
                        label=source.path,
                        path=source.path,
                        score=score,
                    )
                )
        hits.sort(key=lambda h: (-h.score, h.kind, h.id))
        return hits[: self.settings.knowledge_forge_max_search_results]

    def _search_claim_docs(self, query: str) -> list[ClaimV1]:
        needle = query.casefold()
        scored: list[tuple[int, ClaimV1]] = []
        for claim in self.store.claims():
            path = self._rel_path(claim.id)
            body = self._read_doc_body(claim.id)
            score = self._score_text(needle, claim.id, claim.statement, path, body)
            if score > 0:
                scored.append((score, claim))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [claim for _, claim in scored[: self.settings.knowledge_forge_max_search_results]]

    def list_specs(self) -> list[SpecSummaryV1]:
        return [self._spec_summary(spec) for spec in sorted(self.store.specs(), key=lambda s: s.id)]

    def get_spec(self, spec_id: str) -> SpecSummaryV1 | None:
        doc = self.store.get(spec_id)
        if not isinstance(doc, SpecV1):
            return None
        return self._spec_summary(doc)

    def list_decisions(self) -> list[DecisionSummaryV1]:
        return [self._decision_summary(decision) for decision in sorted(self._decisions(), key=lambda d: d.id)]

    def list_context_packs(self) -> list[ContextPackSummaryV1]:
        return [
            self._context_pack_summary(pack)
            for pack in sorted(self._context_packs(), key=lambda p: p.id)
        ]

    def get_context_pack(self, pack_id: str) -> ContextPackSummaryV1 | None:
        doc = self.store.get(pack_id)
        if not isinstance(doc, ContextPackV1):
            return None
        return self._context_pack_summary(doc)

    def list_sources(self) -> list[SourceSummaryV1]:
        return [self._source_summary(source) for source in sorted(self._sources(), key=lambda s: s.id)]

    def ingest_source(self, request: SourceIngestRequestV1) -> SourceIngestResultV1:
        try:
            result = ingest_source(
                self.root,
                source_path=Path(request.path),
                source_id=request.source_id,
                kind=request.kind,
                write_review=request.write_review,
                dry_run=request.dry_run,
                write_enabled=self.settings.knowledge_forge_write_enabled,
                store=self.store,
            )
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        if result.review_path or result.source_path:
            self.reload()
        return SourceIngestResultV1(
            source_id=result.source_id,
            status=result.status,
            source_path=result.source_path,
            review_path=result.review_path,
            proposed_claims=result.proposed_claims,
            possibly_affected_specs=result.possibly_affected_specs,
            warnings=result.warnings,
            content=result.content,
        )

    def list_pending_reviews(self) -> list[ReviewSummaryV1]:
        return [
            ReviewSummaryV1(
                review_id=patch.patch_id,
                target=patch.target,
                action=patch.action,
                path=str(patch.path.relative_to(self.root)),
            )
            for patch in list_pending_patches(self.root)
        ]

    def compile_context_pack(self, request: ContextPackCompileRequestV1) -> ContextPackCompileResultV1:
        included_specs: list[str] = []
        for spec_id in request.spec_ids:
            doc = self.store.get(spec_id)
            if isinstance(doc, SpecV1):
                included_specs.append(spec_id)

        claim_id_set: set[str] = set(request.claim_ids)
        for spec_id in included_specs:
            spec = self.store.get(spec_id)
            if isinstance(spec, SpecV1):
                claim_id_set.update(spec.source_claims)

        included_claims: list[str] = []
        excluded_claims: list[str] = []
        for claim_id in sorted(claim_id_set):
            doc = self.store.get(claim_id)
            if not isinstance(doc, ClaimV1):
                continue
            if _claim_included(
                doc,
                include_disputed=request.include_disputed,
                include_stale=request.include_stale,
            ):
                included_claims.append(claim_id)
            else:
                excluded_claims.append(claim_id)

        content, compile_warnings = compile_context_pack_api_v1(
            self.store,
            task=request.task,
            target=request.target.value,
            spec_ids=request.spec_ids,
            claim_ids=request.claim_ids,
            include_disputed=request.include_disputed,
            include_stale=request.include_stale,
        )
        warnings = list(compile_warnings)
        output_path: Path | None = None
        pack_id = f"ctx:generated:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        if request.write_file:
            if not self.settings.knowledge_forge_write_enabled:
                warnings.append("write disabled: KNOWLEDGE_FORGE_WRITE_ENABLED is false")
            else:
                output_path = build_context_pack_output_path(
                    self.root,
                    target=request.target.value,
                    task=request.task,
                    timestamp=datetime.now(timezone.utc),
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(content, encoding="utf-8")
                pack_id = f"ctx:file:{output_path.stem}"
        return ContextPackCompileResultV1(
            pack_id=pack_id,
            path=str(output_path.relative_to(self.root)) if output_path else None,
            target=request.target.value,
            task=request.task,
            included_specs=included_specs,
            included_claims=included_claims,
            excluded_claims=excluded_claims,
            warnings=warnings,
            content=content,
        )

    def accept_review(self, review_id: str) -> str:
        target = apply_pending_patch(self.root, review_id)
        self.reload()
        return str(target.relative_to(self.root))

    def reject_review(self, review_id: str) -> str:
        rejected_dir = self.root / "reviews" / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        for patch in list_pending_patches(self.root):
            if patch.patch_id != review_id:
                continue
            dest = rejected_dir / patch.path.name
            shutil.move(str(patch.path), str(dest))
            return str(dest.relative_to(self.root))
        raise FileNotFoundError(f"pending review not found: {review_id}")

    def _coerce(self, doc: dict[str, Any]) -> DocModel:
        doc_type = doc.get("type")
        model = _MODEL_BY_TYPE.get(str(doc_type))
        if model is None:
            raise ValueError(f"unknown document type: {doc_type}")
        return model.model_validate(doc)  # type: ignore[return-value]

    def _decisions(self) -> list[DecisionV1]:
        return [d for d in self.store.by_id.values() if isinstance(d, DecisionV1)]

    def _context_packs(self) -> list[ContextPackV1]:
        return [d for d in self.store.by_id.values() if isinstance(d, ContextPackV1)]

    def _sources(self) -> list[SourceV1]:
        return [d for d in self.store.by_id.values() if isinstance(d, SourceV1)]

    def _rel_path(self, doc_id: str) -> str | None:
        path = self.store.paths_by_id.get(doc_id)
        if path is None:
            return None
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def _read_doc_body(self, doc_id: str) -> str:
        path = self.store.paths_by_id.get(doc_id)
        if path is None or not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _score_text(self, needle: str, *parts: str | None) -> int:
        score = 0
        for part in parts:
            if not part:
                continue
            folded = part.casefold()
            if needle == folded:
                score = max(score, 100)
            elif needle in folded:
                score = max(score, 50 if len(parts) <= 2 else 10)
        return score

    def _claim_summary(self, claim: ClaimV1) -> ClaimSummaryV1:
        return ClaimSummaryV1(
            claim_id=claim.id,
            statement=claim.statement,
            status=claim.status.value,
            source_refs=list(claim.source_refs),
            supports=list(claim.supports),
            contradicts=list(claim.contradicts),
            used_by=list(claim.used_by),
            path=self._rel_path(claim.id),
        )

    def _spec_summary(self, spec: SpecV1) -> SpecSummaryV1:
        return SpecSummaryV1(
            spec_id=spec.id,
            title=_spec_title(spec),
            status=spec.status.value,
            component=spec.component,
            requirements=list(spec.requirements),
            non_goals=list(spec.non_goals),
            acceptance_tests=list(spec.acceptance_tests),
            source_claims=list(spec.source_claims),
            likely_files=list(spec.likely_files),
            known_traps=list(spec.known_traps),
            path=self._rel_path(spec.id),
        )

    def _decision_summary(self, decision: DecisionV1) -> DecisionSummaryV1:
        return DecisionSummaryV1(
            id=decision.id,
            status=decision.status.value,
            decision=decision.decision,
            rationale=decision.rationale,
            path=self._rel_path(decision.id),
        )

    def _context_pack_summary(self, pack: ContextPackV1) -> ContextPackSummaryV1:
        return ContextPackSummaryV1(
            id=pack.id,
            target=_api_target_from_model(pack.target),
            task=pack.task,
            included_specs=list(pack.included_specs),
            included_claim_ids=list(pack.included_claim_ids),
            path=self._rel_path(pack.id),
        )

    def _source_summary(self, source: SourceV1) -> SourceSummaryV1:
        return SourceSummaryV1(
            id=source.id,
            kind=source.kind.value,
            path=source.path,
            trust_level=source.trust_level.value,
        )


def _api_target_from_model(target: ContextPackTargetV1) -> str:
    if target == ContextPackTargetV1.orion_agent:
        return "orion"
    return target.value


def _spec_title(spec: SpecV1) -> str:
    if spec.requirements:
        return spec.requirements[0]
    return spec.component
