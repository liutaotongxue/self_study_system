#!/usr/bin/env pwsh
# Windows PowerShell 等效 setup.sh:一次性安装(产品 setup;dev 工具见 README)。
# 幂等:可重跑(uv sync 幂等;.env 仅缺失时建,不覆盖;alembic upgrade 幂等)。
$ErrorActionPreference = 'Stop'

# 1/3 检查 uv 已装
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "需要 uv —— 装:https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

Write-Host "[setup] 1/3 uv sync(依赖,Python>=3.11)..."
uv sync

Write-Host "[setup] 2/3 .env ..."
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "[setup]   已建 .env —— 必须编辑它填 API key,否则无法生成图谱"
} else {
    Write-Host "[setup]   .env 已存在,跳过(不覆盖你的 key)"
}

Write-Host "[setup] 3/3 alembic upgrade head(建/迁移本地 DB)..."
uv run alembic upgrade head

Write-Host ""
Write-Host "[setup] 完成。下一步:"
Write-Host "  1) 若刚建 .env:编辑 .env 填 ANTHROPIC_API_KEY / GOOGLE_API_KEY"
Write-Host "  2) .\run.ps1   然后浏览器开 http://localhost:8000/"
