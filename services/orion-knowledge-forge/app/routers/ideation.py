from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from app.api_schemas import IdeationRunRequestV1, IdeationRunResultV1
from app.ideation.runner import IdeationRunner
from app.routers.v1 import get_service, require_enabled
from app.service import KnowledgeForgeService
from app.settings import Settings

router = APIRouter(prefix="/v1/ideation", tags=["knowledge-forge-ideation"])


def get_runner(service: KnowledgeForgeService = Depends(require_enabled)) -> IdeationRunner:
    return IdeationRunner(Settings(), service)


def require_operator_for_write(
    request: IdeationRunRequestV1,
    x_knowledge_forge_token: str | None = Header(default=None, alias="X-Knowledge-Forge-Token"),
) -> None:
    if not request.write_review:
        return
    expected = Settings().knowledge_forge_operator_token
    if not expected:
        return
    if x_knowledge_forge_token != expected:
        raise HTTPException(status_code=401, detail="invalid operator token")


@router.post("/run", response_model=IdeationRunResultV1)
async def run_ideation(
    request: IdeationRunRequestV1,
    runner: IdeationRunner = Depends(get_runner),
    _write_auth: None = Depends(require_operator_for_write),
) -> IdeationRunResultV1:
    if not runner.settings.knowledge_forge_ideation_enabled:
        raise HTTPException(status_code=503, detail="Knowledge Forge ideation is disabled")
    try:
        return await runner.run(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
