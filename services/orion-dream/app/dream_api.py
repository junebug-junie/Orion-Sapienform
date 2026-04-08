# ==================================================
# dream_api.py
# ==================================================
from fastapi import APIRouter
from datetime import date
from .wake_readout import build_readout

router = APIRouter()

@router.get("/wakeup/today")
def wake_today():
    return build_readout(date.today())
