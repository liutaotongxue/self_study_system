#!/usr/bin/env bash
# 启动(产品:无 --reload;绑 127.0.0.1 —— 无登录本地单用户,不暴露网络)。
# dev 想热重载见 README ## Development。
set -euo pipefail
echo "[run] http://localhost:8000/   (Ctrl-C 停)"
exec uv run uvicorn sla.api.app:app --host 127.0.0.1 --port 8000
