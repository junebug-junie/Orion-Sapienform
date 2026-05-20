from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.api_schemas import (
    ClaimSummaryV1,
    ContextPackCompileRequestV1,
    ContextPackCompileResultV1,
    ContextPackSummaryV1,
    DecisionSummaryV1,
    KnowledgeForgeStatusV1,
    ReviewSummaryV1,
    SearchHitV1,
    SourceSummaryV1,
    SpecSummaryV1,
)
from app.service import KnowledgeForgeService
from app.settings import settings

router = APIRouter(prefix="/v1", tags=["knowledge-forge-v1"])

_service: KnowledgeForgeService | None = None


def get_service() -> KnowledgeForgeService:
    global _service
    if _service is None:
        _service = KnowledgeForgeService(settings)
    return _service


def reset_service(service: KnowledgeForgeService | None = None) -> None:
    global _service
    _service = service


def require_enabled(service: KnowledgeForgeService = Depends(get_service)) -> KnowledgeForgeService:
    if not settings.knowledge_forge_enabled:
        raise HTTPException(status_code=503, detail="Knowledge Forge is disabled")
    return service


def require_operator(
    x_knowledge_forge_token: str | None = Header(default=None, alias="X-Knowledge-Forge-Token"),
) -> None:
    expected = settings.knowledge_forge_operator_token
    if not expected:
        return
    if x_knowledge_forge_token != expected:
        raise HTTPException(status_code=401, detail="invalid operator token")


@router.get("/status", response_model=KnowledgeForgeStatusV1)
def get_status(service: KnowledgeForgeService = Depends(require_enabled)) -> KnowledgeForgeStatusV1:
    return service.status()


@router.get("/claims", response_model=list[ClaimSummaryV1])
def list_claims(service: KnowledgeForgeService = Depends(require_enabled)) -> list[ClaimSummaryV1]:
    return service.list_claims()


@router.get("/claims/search", response_model=list[ClaimSummaryV1])
def search_claims(
    q: str = Query(..., min_length=1),
    service: KnowledgeForgeService = Depends(require_enabled),
) -> list[ClaimSummaryV1]:
    return service.search_claims(q)


@router.get("/search", response_model=list[SearchHitV1])
def search_all(
    q: str = Query(..., min_length=1),
    service: KnowledgeForgeService = Depends(require_enabled),
) -> list[SearchHitV1]:
    return service.search(q)


@router.get("/specs", response_model=list[SpecSummaryV1])
def list_specs(service: KnowledgeForgeService = Depends(require_enabled)) -> list[SpecSummaryV1]:
    return service.list_specs()


@router.get("/specs/{spec_id}", response_model=SpecSummaryV1)
def get_spec(spec_id: str, service: KnowledgeForgeService = Depends(require_enabled)) -> SpecSummaryV1:
    spec = service.get_spec(spec_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"spec not found: {spec_id}")
    return spec


@router.get("/decisions", response_model=list[DecisionSummaryV1])
def list_decisions(service: KnowledgeForgeService = Depends(require_enabled)) -> list[DecisionSummaryV1]:
    return service.list_decisions()


@router.get("/context-packs", response_model=list[ContextPackSummaryV1])
def list_context_packs(
    service: KnowledgeForgeService = Depends(require_enabled),
) -> list[ContextPackSummaryV1]:
    return service.list_context_packs()


@router.get("/context-packs/{pack_id}", response_model=ContextPackSummaryV1)
def get_context_pack(
    pack_id: str,
    service: KnowledgeForgeService = Depends(require_enabled),
) -> ContextPackSummaryV1:
    pack = service.get_context_pack(pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail=f"context pack not found: {pack_id}")
    return pack


@router.post("/context-packs/compile", response_model=ContextPackCompileResultV1)
def compile_context_pack(
    request: ContextPackCompileRequestV1,
    service: KnowledgeForgeService = Depends(require_enabled),
) -> ContextPackCompileResultV1:
    return service.compile_context_pack(request)


@router.get("/reviews/pending", response_model=list[ReviewSummaryV1])
def list_pending_reviews(
    service: KnowledgeForgeService = Depends(require_enabled),
) -> list[ReviewSummaryV1]:
    return service.list_pending_reviews()


@router.post("/reviews/{review_id}/accept")
def accept_review(
    review_id: str,
    _: None = Depends(require_operator),
    service: KnowledgeForgeService = Depends(require_enabled),
) -> dict[str, str]:
    try:
        target = service.accept_review(review_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"review_id": review_id, "target": target}


@router.post("/reviews/{review_id}/reject")
def reject_review(
    review_id: str,
    _: None = Depends(require_operator),
    service: KnowledgeForgeService = Depends(require_enabled),
) -> dict[str, str]:
    try:
        path = service.reject_review(review_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"review_id": review_id, "path": path}


@router.get("/sources", response_model=list[SourceSummaryV1])
def list_sources(service: KnowledgeForgeService = Depends(require_enabled)) -> list[SourceSummaryV1]:
    return service.list_sources()
