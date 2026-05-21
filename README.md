# Self-Learning Agent

把 PDF 教材本地一键学成**知识图谱 / Markdown 笔记**的单用户 Web 工具。
clone 下来,装依赖,跑起来,在浏览器里上传 PDF → 标章节 → 一键生成可视化 KG + 可阅读 Notes。

---

## 它能做什么

- 📚 **上传 PDF**:你的教材本地保存(不入 git)
- 🗂 **抽目录**:你选目录所在页,LLM 一发抽出全书章节结构
- 📑 **章节 OCR**:你选某节内容页,视觉 LLM 把扫描页 OCR 成 markdown
- 📝 **生成笔记**:LLM 按内容生成 markdown 学习笔记(支持 LaTeX 公式)
- 🕸 **构建知识图谱**:从笔记抽出 concepts + relations,vis-network 可视化
- 📥 **导出 Notes**:一键下载 zip(一节一 `.md`,Obsidian / Typora 友好)

每一步都在网页里点;终端只为安装。

---

## 前置要求

| 工具 | 用途 |
|---|---|
| Python ≥ 3.11 | 后端运行时 |
| [uv](https://docs.astral.sh/uv/) | 依赖管理 |
| 一个或多个 LLM provider 的 API key | 详见下方 |
| 一台现代浏览器 | viewer 前端 |

> **Windows 用户**:本项目提供 PowerShell(`.ps1`)启动脚本,**无需 Git Bash 或 WSL**。首次使用请先看下方「[Windows 首次准备](#windows-首次准备)」段(3 个小步骤)。

项目通过 LangChain 抽象层接入 LLM,在 `.env` 配置你要用的 provider key:

- `ANTHROPIC_API_KEY` —— [获取](https://console.anthropic.com/)
- `GOOGLE_API_KEY` —— [获取](https://aistudio.google.com/app/apikey)

> ⚠️ **API key 用于按 token 计费的 API 调用,与各家的网页/桌面聊天订阅产品(若有)完全无关**——后者覆盖不到 API。

---

## Windows 首次准备

> macOS / Linux 用户跳过这段。

### 1. 安装 uv

在 PowerShell 跑(无需管理员):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

装完**关掉重开 PowerShell**(让 PATH 生效),`uv --version` 能输出版本号即成功。

### 2. 解锁脚本执行策略

Windows 默认禁止跑 `.ps1` 脚本。打开 PowerShell 跑一次(只需一次):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

提示输入 `Y` 确认。这只影响**当前用户**,允许本地脚本和签名脚本运行,不影响系统其他用户。

### 3. (可选)装 PowerShell 7+

Windows 自带 PowerShell 5.1 已经够用。如果你想要更好的 Unicode 支持、更现代的语法,可以从 [aka.ms/powershell](https://aka.ms/powershell) 装 PowerShell 7+。

### Windows 常见报错速查

| 报错 | 原因 | 修法 |
|---|---|---|
| `uv: 不是内部或外部命令` | uv 没装 / PATH 没生效 | 关掉重开 PowerShell;或回 1 步重装 |
| `因为在此系统上禁止运行脚本` | ExecutionPolicy 未解锁 | 跑第 2 步命令 |
| `ModuleNotFoundError: No module named 'pymupdf'`(或其他模块) | 直接跑了 `.\run.ps1` 跳过 setup | 先跑 `.\setup.ps1` |
| 控制台中文乱码 | PowerShell 输出编码非 UTF-8 | 跑 `chcp 65001` 切 UTF-8,或用 PowerShell 7+ |
| `git pull` 提示 lockfile 冲突 | 多机协作时 uv.lock 冲突 | 跑 `uv sync` 重新解析即可 |

---

## 一次性安装

```bash
git clone https://github.com/liutaotongxue/self_study_system.git
cd self_study_system
```

**macOS / Linux**:

```bash
bash setup.sh
```

**Windows (PowerShell)**:

```powershell
.\setup.ps1
```

无论哪个,都会:
1. `uv sync` 安装依赖
2. 复制 `.env.example` → `.env`(若不存在)
3. `alembic upgrade head` 建本地 SQLite DB

跑完按提示编辑 `.env`,填 `ANTHROPIC_API_KEY` 和 `GOOGLE_API_KEY`。

---

## 启动

**macOS / Linux**:

```bash
bash run.sh
```

**Windows (PowerShell)**:

```powershell
.\run.ps1
```

打开浏览器 `http://localhost:8000/`。绑 `127.0.0.1:8000`,**不暴露网络**(单机单用户)。

---

## 使用流程

| 步 | 在哪 | 做什么 |
|---|---|---|
| 1 | 主页 `/` | 上传 PDF(只登记,不解析) |
| 2 | viewer `/index.html?doc=N` → 章节状态 | 输入目录所在页(如 `7-10`)→ LLM 抽全书结构 |
| 3 | 章节状态页 | 给某节填内容页(如 `28-35`)→ 缩略图预览确认 → 视觉 OCR + 切块 |
| 4 | 章节状态页 | 点「生成」→ LLM 出 markdown 笔记 + KG |
| 5 | viewer | 看图、点节点读笔记、导出 Notes zip |

---

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI + SQLAlchemy 2.0 + Alembic + SQLite |
| LLM 编排 | LangGraph + LangChain(多 provider 抽象) |
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
| `ingest_pdf.py` | 离线把 PDF 灌入 DB(网页端等价于上传 + 抽目录) |
| `study_book.py` | 离线给章节生成 markdown 笔记 |
| `build_kg.py` | 离线给章节抽 KG(支持 `--clear` 防累积) |
| `run_generation.py` | UI 触发 generation job 的执行器(后台 Popen) |

---

## 卸载

项目完全自包含,卸载只需删目录:

```bash
# macOS / Linux / Git Bash
rm -rf self_study_system

# Windows PowerShell
Remove-Item -Recurse -Force self_study_system
```

所有数据(`app.db`、`.env`、`.venv/`、`uploads/`)都在项目目录内,一并清除。**不会**残留任何系统级文件、服务或全局包。

可选额外清理:

- 不再用任何 LLM 项目 → 登录你 API key 所属 provider 的 console 撤销 key(防泄漏被刷费)
- 不再用任何 uv 项目 → `uv cache clean` 或 `rm -rf ~/.cache/uv`

---

## License

[MIT](LICENSE) © 2026 liutao

---

## 致谢

参考过的工作:
- [Nous Research Hermes Agent](https://github.com/NousResearch/hermes-agent) — LangGraph harness 设计参考
