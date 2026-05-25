"""FastAPI app."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from sla.api import routes_domain, routes_generation, routes_runtime

app = FastAPI(title="Self-Study System", version="0.1.0")

app.include_router(routes_domain.router)
app.include_router(routes_runtime.router)
app.include_router(routes_generation.router)


@app.get("/", response_class=FileResponse)
def root():
    # P1b-i2: single-user -> home page = library page (the original JSON root has no external consumers, see P1b decision b)
    return FileResponse(
        Path(__file__).parent.parent.parent.parent / "web" / "library.html",
        media_type="text/html",
    )


@app.get("/health")
def health():
    return {"status": "ok"}
