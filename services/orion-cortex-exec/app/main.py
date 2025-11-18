# orion-cortex-exec/app/main.py

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import PlanExecutionRequest, PlanExecutionResult
from .router import PlanRouter
from .config import settings

logger = logging.getLogger("orion-cortex")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Orion Cognition Cortex",
    description="Thin execution client for Orion's cognitive plans.",
    version="0.1.0",
)

# CORS if you want to call this from Hub UI directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

plan_router = PlanRouter()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "node": settings.NODE_NAME,
    }


@app.post("/execute-plan", response_model=PlanExecutionResult)
async def execute_plan(req: PlanExecutionRequest):
    """
    Execute a full cognitive plan produced by the semantic planner.
    """
    logger.info(
        f"Received plan for verb={req.plan.verb_name}, "
        f"request_id={req.args.request_id}"
    )
    result = await plan_router.execute_plan(req)
    return result
