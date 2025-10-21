# ==================================================
# dream_api.py
# ==================================================
from fastapi import FastAPI
from datetime import date
from .wake_readout import build_readout

app = FastAPI(title="Orion Dream API", version="1.0.0")

@app.get("/dreams/wakeup/today")
def wake_today():
    return build_readout(date.today())
