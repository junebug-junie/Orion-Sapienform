from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    from ..context import ctx

    state = ctx.state_store.load()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "armed": state.armed, "mode": state.mode},
    )
