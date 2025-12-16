# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Paper2Slides 是一个将研究论文、报告和文档自动转换为专业幻灯片和海报的工具。它使用 RAG（检索增强生成）技术和 LLM 进行内容提取、规划和图像生成。

## 常用命令

### 环境设置
```bash
# 创建并激活 conda 环境
conda create -n paper2slides python=3.12 -y
conda activate paper2slides

# 安装依赖
pip install -r requirements.txt
```

### 命令行使用
```bash
# 基础用法 - 从论文生成幻灯片
python -m paper2slides --input paper.pdf --output slides --length medium

# 生成海报（自定义风格）
python -m paper2slides --input paper.pdf --output poster --style "minimalist with blue theme" --density medium

# 快速模式（跳过 RAG 索引）
python -m paper2slides --input paper.pdf --output slides --fast

# 并行生成（使用 2 个 worker）
python -m paper2slides --input paper.pdf --output slides --parallel 2

# 列出所有已处理的输出
python -m paper2slides --list

# 从特定阶段重新开始
python -m paper2slides --input paper.pdf --output slides --from-stage plan

# 生成透明背景图片
python -m paper2slides --input paper.pdf --output slides --transparent-bg
```

### Web 界面
```bash
# 启动所有服务（后端 + 前端）
./scripts/start.sh

# 或分别启动
./scripts/start_backend.sh  # 后端 API (默认端口 8001)
./scripts/start_frontend.sh # 前端 (默认端口 5173)

# 停止所有服务
./scripts/stop.sh
```

### 前端开发
```bash
cd frontend
npm install          # 安装依赖
npm run dev          # 开发模式
npm run build        # 构建生产版本
npm run preview      # 预览构建结果
```

## 核心架构

### 四阶段流水线

Paper2Slides 通过 4 个阶段处理文档，每个阶段都有检查点保存：

1. **RAG 阶段** (`paper2slides/core/stages/rag_stage.py`)
   - 解析文档并构建 RAG 索引
   - 使用 `raganything/` 模块进行多模态文档解析
   - 输出：`checkpoint_rag.json` 和 `rag_output/` 索引

2. **摘要阶段** (`paper2slides/core/stages/summary_stage.py`)
   - 提取文档结构、图表、表格
   - 使用 `summary/` 模块的提取器
   - 输出：`checkpoint_summary.json` 和 `summary.md`

3. **规划阶段** (`paper2slides/core/stages/plan_stage.py`)
   - 生成内容布局和幻灯片/海报组织策略
   - 使用 `generator/content_planner.py`
   - 输出：`checkpoint_plan.json`

4. **生成阶段** (`paper2slides/core/stages/generate_stage.py`)
   - 渲染最终的高质量图像
   - 使用 `generator/image_generator.py`
   - 支持并行生成（`--parallel` 参数）
   - 输出：PNG/JPG 图像和 PDF（幻灯片模式）

### 流水线编排

- **入口点**：`paper2slides/main.py` (CLI) 和 `api/server.py` (Web API)
- **核心编排**：`paper2slides/core/pipeline.py` 中的 `run_pipeline()` 函数
- **状态管理**：`paper2slides/core/state.py` 处理检查点的加载/保存
- **路径管理**：`paper2slides/core/paths.py` 管理输出目录结构

### 检查点与恢复

系统在每个阶段自动保存进度：
- 中断后重新运行相同命令会自动恢复
- 使用 `--from-stage` 强制从特定阶段重新开始
- 检查点文件位于 `outputs/<project>/<content_type>/<mode>/`

### 模式差异

- **Normal 模式**：完整的 RAG 索引 + 深度文档分析（适合长文档、多文件）
- **Fast 模式** (`--fast`)：跳过 RAG 索引，直接 LLM 查询（适合短文档、快速预览）

## 配置说明

### 环境变量

在 `paper2slides/.env` 中配置（参考 `paper2slides/.env.example`）：

```bash
# RAG LLM API
RAG_LLM_API_KEY=""
RAG_LLM_BASE_URL=""

# 图像生成
IMAGE_GEN_PROVIDER="openrouter"  # 或 "google"
IMAGE_GEN_API_KEY=""
IMAGE_GEN_BASE_URL=""
IMAGE_GEN_MODEL=""               # 默认 google/gemini-3-pro-image-preview
IMAGE_GEN_RESPONSE_MIME_TYPE="image/png"
GOOGLE_GENAI_BASE_URL=""
```

### 输出目录结构

```
outputs/
├── <project_name>/
│   ├── <content_type>/          # paper 或 general
│   │   ├── <mode>/              # fast 或 normal
│   │   │   ├── checkpoint_rag.json
│   │   │   ├── checkpoint_summary.json
│   │   │   ├── summary.md
│   │   │   └── <config_name>/   # 例如 slides_doraemon_medium
│   │   │       ├── state.json
│   │   │       ├── checkpoint_plan.json
│   │   │       └── <timestamp>/
│   │   │           ├── slide_01.png
│   │   │           └── slides.pdf
│   │   └── rag_output/          # RAG 索引存储
```

## 关键模块

### RAG 与文档解析
- `paper2slides/raganything/`：多模态 RAG 处理
  - `raganything.py`：RAG 处理器主类
  - `parser.py`：文档解析器
  - `modalprocessors.py`：多模态处理器
  - 依赖 `lightrag-hku` 和 `mineru` 库

### 内容提取
- `paper2slides/summary/`：内容提取模块
  - `paper.py`：论文结构提取
  - `extractors/figure_extractor.py`：图片提取
  - `extractors/table_extractor.py`：表格提取
  - `models.py`：数据模型定义

### 内容生成
- `paper2slides/generator/`：内容规划与图像生成
  - `content_planner.py`：幻灯片/海报内容规划
  - `image_generator.py`：图像生成（支持并行）
  - `config.py`：生成配置（风格、长度、密度）

### Prompts
- `paper2slides/prompts/`：LLM 提示词模板
  - `content_planning.py`：内容规划提示词
  - `image_generation.py`：图像生成提示词
  - `paper_extraction.py`：论文提取提示词

### Web API
- `api/server.py`：FastAPI 后端
  - 文件上传处理
  - 会话管理（支持取消）
  - 进度跟踪
  - 静态文件服务

### 前端
- `frontend/src/`：React + Vite + TailwindCSS
  - 文件上传界面
  - 实时进度显示
  - 结果预览

## 开发注意事项

### 图像生成
- 默认使用 `gemini-3-pro-image-preview` (OpenRouter)
- 可切换到 Google Gemini API (`IMAGE_GEN_PROVIDER=google`)
- 透明背景模式 (`--transparent-bg`) 强制使用 PNG 格式
- 并行生成使用 `ThreadPoolExecutor`，通过 `--parallel N` 控制 worker 数量

### 会话管理
- Web API 使用 `SessionManager` 跟踪运行中的会话
- 支持取消正在运行的任务
- CLI 模式不使用会话管理

### 错误处理
- 每个阶段的错误会保存到 `state.json` 的 `error` 字段
- 失败的阶段状态标记为 `"failed"`
- 可以通过 `--from-stage` 从失败的阶段重新开始

### 日志
- 使用 Python `logging` 模块
- `--debug` 参数启用 DEBUG 级别日志
- Web 后端日志保存到 `logs/backend.log`