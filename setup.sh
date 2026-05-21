#!/usr/bin/env bash
# 一次性安装(产品 setup:无 fixture / 无 smoke —— 那是 dev,见 README ## Development)。
# 幂等:可重跑(uv sync 幂等;.env 仅缺失时建,不覆盖;alembic upgrade 幂等)。
set -euo pipefail

command -v uv >/dev/null 2>&1 || {
  echo "ERROR: 需要 uv —— 装:https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1; }

echo "[setup] 1/3 uv sync(依赖,Python>=3.11)..."
uv sync

echo "[setup] 2/3 .env ..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "[setup]   已建 .env —— 必须编辑它填 ANTHROPIC_API_KEY,否则无法生成图谱"
else
  echo "[setup]   .env 已存在,跳过(不覆盖你的 key)"
fi

echo "[setup] 3/3 alembic upgrade head(建/迁移本地 DB)..."
uv run alembic upgrade head

echo
echo "[setup] 完成。下一步:"
echo "  1) 若刚建 .env:编辑 .env 填 ANTHROPIC_API_KEY"
echo "  2) bash run.sh   然后浏览器开 http://localhost:8000/"
