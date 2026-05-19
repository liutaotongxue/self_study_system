# Self-Learning Agent

Clone 下来本地跑、在网页上把 PDF 教材学成知识图谱 / 笔记 / 题目的**单用户**工具。
项目历史与设计决策见 `RETROSPECTIVE.md`。

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- 一个 Anthropic API key

## Setup(只跑一次)

```bash
git clone <repo-url> && cd self-learning-agent
bash setup.sh
# 按提示编辑 .env,填 ANTHROPIC_API_KEY
```

## Run

```bash
bash run.sh
# 浏览器打开 http://localhost:8000/
```

之后全部操作(上传书、生成图谱、学习)都在网页里;终端只用于上面这一次安装。

> 数据是你本地的:`app.db` 等不入 git(`.gitignore` 已含 `*.db`),各人一份本地实例。

## 重要提示

- **ANTHROPIC_API_KEY 与 Claude Max 订阅无关**。Max 覆盖 claude.ai 和 Claude Code,API 调用单独计费。
- **不要把 `app.db` / `.env` commit 到 git**。`.gitignore` 已配置。
- **LangGraph / LangChain 版本 pin 死**。只用 LangGraph + 少量 LangChain Core,不引入 LangChain 主仓。

## Development

贡献者 / 开发用(产品用户不需要):

```bash
uv sync
cp .env.example .env            # 填 ANTHROPIC_API_KEY
uv run alembic upgrade head
uv run python scripts/load_fixture.py          # 灌金标 fixture chunks(仅 dev)
uv run python scripts/smoke_test_anthropic.py  # 冒烟:Claude API 通路
uv run python scripts/smoke_test_langchain.py
uv run uvicorn sla.api.app:app --reload        # 热重载(改码自动重启)
```

验收 / 测试:

```bash
curl http://localhost:8000/documents/1/chunks   # fixture 灌入后应见 chunks
curl http://localhost:8000/tasks                 # tasks endpoint 可用
uv run pytest tests/                             # 含 Eval-1 / O3 / O4 回归
```

金标 fixture:`fixtures/sample_chapter.json` = Sutton & Barto《RL: An Introduction》
(2nd ed. in-progress)§1.3 的 8 段 chunks(`ch1.3`,书页 7-8),仅供 dev。
# self_study_system
