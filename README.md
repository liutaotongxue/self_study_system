# Self-Learning Agent

把 PDF 教材本地一键学成**知识图谱 / Markdown 笔记**的单用户 Web 工具。
clone 下来,装依赖,跑起来,在浏览器里上传 PDF → 标章节 → 一键生成可视化 KG + 可阅读 Notes。

---

## 它能做什么

- 📚 **上传 PDF**:你的教材本地保存(不入 git)
- 🗂 **抽目录**:你选目录所在页,LLM 一发抽出全书章节结构
- 📑 **章节 OCR**:你选某节内容页,视觉模型(Gemini 2.5-flash)OCR 出 markdown
- 📝 **生成笔记**:Claude(sonnet-4-6)按内容生成 markdown 学习笔记(支持 LaTeX 公式)
- 🕸 **构建知识图谱**:从笔记抽出 concepts + relations,vis-network 可视化
- 📥 **导出 Notes**:一键下载 zip(一节一 `.md`,Obsidian / Typora 友好)

每一步都在网页里点;终端只为安装。

---

## 前置要求

| 工具 | 用途 |
|---|---|
| Python ≥ 3.11 | 后端运行时 |
| [uv](https://docs.astral.sh/uv/) | 依赖管理 |
| Anthropic API key | S2 抽结构 / S5 笔记 + KG agent([注册](https://console.anthropic.com/)) |
| Google API key (Gemini) | S4 视觉 OCR([注册](https://aistudio.google.com/app/apikey)) |
| 一台现代浏览器 | viewer 前端 |

> ⚠️ **Anthropic API key 与 Claude Max / Pro 订阅完全无关**。Max 覆盖 claude.ai 和 Claude Code 网页/桌面端,API 调用按 token 单独计费。

---

## 一次性安装

```bash
git clone https://github.com/liutaotongxue/self_study_system.git
cd self_study_system
bash setup.sh
# 按提示编辑 .env,填 ANTHROPIC_API_KEY 和 GOOGLE_API_KEY
```

`setup.sh` 会:
1. `uv sync` 安装依赖
2. 复制 `.env.example` → `.env`(若不存在)
3. `alembic upgrade head` 建本地 SQLite DB

---

## 启动

```bash
bash run.sh
# 浏览器打开 http://localhost:8000/
```

`run.sh` 绑 `127.0.0.1:8000`,**不暴露网络**(单机单用户)。

---

## 使用流程

| 步 | 在哪 | 做什么 |
|---|---|---|
| 1 | 主页 `/` | 上传 PDF(只登记,不解析) |
| 2 | viewer `/index.html?doc=N` → 章节状态 | 输入目录所在页(如 `7-10`)→ LLM 抽全书结构 |
| 3 | 章节状态页 | 给某节填内容页(如 `28-35`)→ 缩略图预览确认 → Gemini OCR + 切块 |
| 4 | 章节状态页 | 点「生成」→ Claude 出 markdown 笔记 + KG |
| 5 | viewer | 看图、点节点读笔记、导出 Notes zip |

---

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI + SQLAlchemy 2.0 + Alembic + SQLite |
| LLM 编排 | LangGraph 1.x + langchain-anthropic 1.x + langchain-google-genai 4.x |
| 视觉 OCR | Gemini 2.5-flash(`safety_settings=BLOCK_NONE`) |
| 文本/KG | Claude Sonnet 4.6 |
| PDF 处理 | pymupdf |
| 前端 | 原生 HTML/JS + marked(markdown) + KaTeX(数学公式) + vis-network(KG) |

零打包 / 零构建步骤——前端就是 3 个静态 HTML 文件。

---

## 项目结构

```
self-learning-agent/
├── setup.sh / run.sh           安装 + 启动
├── pyproject.toml              依赖声明
├── alembic/ + alembic.ini      DB schema 迁移链
├── src/sla/                    后端源码
│   ├── api/                    FastAPI 路由
│   ├── models/                 SQLAlchemy 模型
│   ├── harness/                LangGraph agent harness + KG 抽取
│   ├── parsing/                PDF 渲染 / TOC / 内容 OCR
│   └── runtime/                runner / task / event
├── web/                        前端(index.html / library.html / process.html)
├── scripts/                    CLI 工具(ingest / study_book / build_kg / ...)
└── tests/                      pytest 测试
```

数据在本地:`app.db`(SQLite) + 用户原 PDF 的绝对路径,都不入 git。

---

## 开发

```bash
# 热重载(改码自动重启)
uv run uvicorn sla.api.app:app --reload

# 运行测试(71 tests)
uv run pytest tests/

# DB 迁移(改 model 后)
uv run alembic revision --autogenerate -m "..."
uv run alembic upgrade head
```

主要 CLI 工具(都在 `scripts/`):

| 脚本 | 作用 |
|---|---|
| `ingest_pdf.py` | 离线把 PDF 灌入 DB(网页端等价于上传 + S2 + S3) |
| `study_book.py` | 离线给章节生成 markdown 笔记 |
| `build_kg.py` | 离线给章节抽 KG(支持 `--clear` 防累积) |
| `run_generation.py` | UI 触发 generation job 的执行器(后台 Popen) |

---

## 卸载

项目完全自包含,卸载只需删目录:

```bash
rm -rf self_study_system
```

所有数据(`app.db`、`.env`、`.venv/`、`uploads/`)都在项目目录内,一并清除。**不会**残留任何系统级文件、服务或全局包。

可选额外清理:

- 不再用任何 LLM 项目 → 到 [Anthropic Console](https://console.anthropic.com/) / [Google AI Studio](https://aistudio.google.com/app/apikey) 撤销 API key
- 不再用任何 uv 项目 → `uv cache clean` 或 `rm -rf ~/.cache/uv`

---

## License

[MIT](LICENSE) © 2026 liutao

---

## 致谢

参考过的工作:
- [Nous Research Hermes Agent](https://github.com/NousResearch/hermes-agent) — LangGraph harness 设计参考
- [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) — content filter 诊断时直读 native API
