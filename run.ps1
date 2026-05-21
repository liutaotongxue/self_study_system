#!/usr/bin/env pwsh
# Windows PowerShell 等效 run.sh:启动 uvicorn(产品:无 --reload;绑 127.0.0.1 不暴露网络)。
$ErrorActionPreference = 'Stop'
Write-Host "[run] http://localhost:8000/   (Ctrl-C 停)"
uv run uvicorn sla.api.app:app --host 127.0.0.1 --port 8000
