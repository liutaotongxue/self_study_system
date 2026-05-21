"""FastAPI app。"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from sla.api import routes_domain, routes_generation, routes_runtime

app = FastAPI(title="Self-Learning Agent", version="0.1.0")

app.include_router(routes_domain.router)
app.include_router(routes_runtime.router)
app.include_router(routes_generation.router)


@app.get("/", response_class=FileResponse)
def root():
    # P1b-i2:单用户 → 首页=书库页(原 JSON root 无外部消费者,见 P1b 决策 b)
    return FileResponse(
        Path(__file__).parent.parent.parent.parent / "web" / "library.html",
        media_type="text/html",
    )


@app.get("/health")
def health():
    return {"status": "ok"}
