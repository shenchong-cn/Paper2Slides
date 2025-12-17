# SVG 透明背景方案设计文档

## 文档信息
- **创建日期**: 2025-12-17
- **版本**: v1.4（评审修订：对齐当前仓库 API/CLI/Stage 真实接口；补齐“模型选择/可视参考/安全净化白名单/可选依赖与测试目录”落地细节；修正章节编号）
- **作者**: Claude Code
- **GitHub 仓库**: https://github.com/shenchong-cn/Paper2Slides
- **评审对齐代码版本**: `0b98f41`
- **相关 GitHub Issue / PR**:
  - Issue: （建议创建，如 `SVG output: transparent background`，待填 `#123`）
  - PR: （落地实现时创建，待填 `#456`）
- **相关问题**: PNG 透明背景效果不佳，尝试改用 SVG 矢量格式作为替代输出
- **修订历史**:
  - v1.0: 初始版本
  - v1.1: 对齐现有代码结构、补齐兼容/安全/测试闭环
  - v1.2: 补充依赖声明、明确文本换行策略、完善 LLM API 适配代码、明确 both 模式定义
  - v1.3: 补充 GitHub 追踪信息；明确与现有 `paper2slides/main.py`、`paper2slides/core/stages/generate_stage.py` 的对齐点；补齐 SVG→PNG 依赖可行性说明与可测试落地方案
  - v1.4: 对齐当前仓库 `api/server.py`（`/api/chat`）与 CLI；补齐 provider/模型能力前置条件；补齐 SVG 净化白名单与测试目录落地建议；修正编号

## 0. 目标与范围

### 0.1 目标

将图像生成从 PNG 位图格式改为 SVG 矢量格式，实现：
- 真正的透明背景（无需后处理去白底/去面板）
- 更稳定的边缘质量（避免 PNG 抗锯齿白边）
- 任意缩放不失真（矢量）
- 在“纯矢量为主”场景下文件体积更小（但嵌入大图片时可能更大）
- Office 2016+ 可用（需限制在 PPT 支持的 SVG 子集）

### 0.1.1 与现有“透明背景 PNG 改进方案”的关系

仓库已有 `docs/transparent_background_improvement_design.md`，其方向是“继续产出 PNG，但尽可能让 alpha 真实、去掉浅色面板作为兜底”。本 SVG 方案属于**替代/补充输出链路**：
- **适合**：文字/形状/简单图表为主、用户希望导入 PPT 后可无限缩放、且可接受“PPT 支持的 SVG 子集”限制。
- **不适合**：大量照片/复杂渐变/滤镜效果、或者必须依赖 `slides.pdf` 且不希望引入 SVG→PNG 栅格化依赖的场景。

### 0.2 PNG 方案的问题

现有 PNG 方案存在以下问题：
1. **模型生成不可控**: 即使优化提示词，模型仍可能生成白色内容卡片
2. **后处理复杂**: 需要复杂的算法去除浅色面板，容易误删内容
3. **边缘质量差**: 抗锯齿产生半透明像素，在有色背景上出现白边
4. **缩放失真**: 位图放大会模糊，缩小会丢失细节
5. **文件体积大**: 高分辨率 PNG 文件较大

### 0.3 SVG 方案的优势

| 特性 | PNG 方案 | SVG 方案 |
|------|---------|---------|
| 透明背景 | 需要后处理 | 原生支持 |
| 边缘质量 | 有锯齿/白边 | 完美平滑 |
| 缩放 | 失真 | 无损 |
| 文件大小 | 较大 | 较小 |
| 可编辑性 | 困难 | 容易 |
| PPT 兼容性 | 一般 | 取决于 SVG 子集（可做到更好） |

### 0.4 现状与差异清单（对齐当前仓库，避免“按文档编码即偏离”）

为了确保本文档可直接驱动实现，需要先明确当前仓库的真实接口与本文档新增点的差异：

1. **CLI 现状**：`paper2slides/main.py` 当前只有 `--transparent-bg`（PNG 真透明改进链路），尚无 `--format/--image-format`（SVG 输出）参数。
2. **Web API 现状**：`api/server.py` 的入口是 `POST /api/chat`（Form + 文件上传），并在服务端构建 `config` 传入流水线；不存在 `POST /api/generate`。
3. **Stage 保存逻辑现状**：`paper2slides/core/stages/generate_stage.py` 的 `save_image_callback` 在 `transparent_bg=true` 时会**强制 `.png` 扩展名**，并且 `ext_map` 不包含 `.svg`；因此 SVG 输出必须显式改造这段逻辑，不能依赖透明 PNG 分支。
4. **PDF 现状**：`paper2slides/generator/image_generator.py:save_images_as_pdf()` 只能处理位图 bytes，且 `RGBA -> RGB` 会把透明区域合成到黑底；SVG 若要生成 `slides.pdf`，必须先 SVG→PNG，并且建议提供“合成底色”选项（默认白底更接近 PPT 观感）。

## 1. 技术方案

### 1.1 方案概述

采用 **LLM 直接生成 SVG 代码（文本输出）** 的方案：
1. 修改提示词，要求模型输出 SVG 代码而非图像
2. 验证和清理 SVG 代码
3. 可选：转换为 PNG 作为备份格式

### 1.1.1 与当前仓库实现对齐（必须明确）

当前仓库 `paper2slides/generator/image_generator.py` 的生成接口是“调用模型 → 返回二进制 image bytes + mime_type”，并在 `paper2slides/core/stages/generate_stage.py` 保存文件、在 slides 模式下合成 `slides.pdf`（仅支持位图）。因此 SVG 方案需要在设计上明确：

1. **模型返回形式**：SVG 以“文本”返回（推荐），而不是依赖模型直接返回 `image/svg+xml`（不同 provider 能力不一致）。
2. **输出闭环**：若输出为 `.svg`，是否同时导出 `.png`（用于 `slides.pdf` 以及旧链路兼容）必须在配置与 CLI 中定义；否则需要显式关闭/跳过 PDF 合成。
3. **Stage 保存逻辑**：当前 `paper2slides/core/stages/generate_stage.py` 通过 `save_callback` 边生成边落盘，并基于 `mime_type→ext` 做扩展名映射；要支持 `.svg`，需要：
   - 将 `image/svg+xml` 映射到 `.svg`；
   - 若 `slides` 模式仍需 `slides.pdf`，必须保证参与 `save_images_as_pdf()` 的是位图（例如由 SVG 栅格化得到的 PNG），否则 PDF 合成会失败。
4. **OpenRouter/Google 差异**：当前 OpenRouter 路径只从 `message.images` 取图像数据；要支持 SVG 文本，需要新增“从 `message.content` 取文本”的解析分支（或新建文本调用函数）。

### 1.1.2 模型与 Provider 配置策略（建议明确）

SVG 生成本质是“文本生成”（可能带参考图），与当前“图像生成”链路使用的 `IMAGE_GEN_*` 环境变量耦合较强。

**实现前置条件（必须写清，否则落地会踩坑）**：

- SVG 输出依赖“文本生成”能力：需要一个**文本/聊天模型**；当前默认值可能指向“图像模型”（例如 OpenRouter 路径默认 `google/gemini-3-pro-image-preview`）。若直接复用默认模型，可能导致返回结构不稳定或不输出可用 SVG。
- 若要把 figures 作为“可视参考”传入 SVG 生成，则文本模型必须支持**视觉输入**（vision）。否则应提供开关禁用参考图片，仅保留 captions/表格文本，避免调用失败。

**本次实施决策：默认复用 `IMAGE_GEN_*` 配置，但允许引入可选的 `SVG_GEN_*` 覆盖**：
- 默认：复用 `IMAGE_GEN_PROVIDER/IMAGE_GEN_API_KEY/IMAGE_GEN_BASE_URL/IMAGE_GEN_MODEL`
- 可选覆盖（建议后续加，避免干扰 PNG 路径）：`SVG_GEN_MODEL`、`SVG_GEN_MAX_TOKENS`、`SVG_GEN_USE_REFERENCE_IMAGES`

实现注意事项（必须写进代码逻辑而非靠使用者配置）：
- SVG 文本调用需要**强制按文本响应解析**（不可依赖全局 `IMAGE_GEN_RESPONSE_MIME_TYPE`），否则会影响现有 PNG 行为或导致 provider 路径分叉。
- Google 路径需设置 `generationConfig.responseMimeType="text/plain"` 并从 `parts[].text` 读取输出（可复用现有 `requests` 调用形态）。

（后续可选优化）如需要更强隔离，可再引入 `SVG_GEN_*`（provider/model/max_tokens）做解耦。

### 1.2 核心流程

```
用户请求
  ↓
生成内容规划（现有流程）
  ↓
构建 SVG 生成提示词
  ↓
LLM 生成 SVG 代码
  ↓
验证 SVG 语法
  ↓
清理和优化 SVG
  ↓
保存 SVG 文件
  ↓
可选：导出 PNG 备份
```

### 1.3 SVG 生成提示词设计

```python
SVG_GENERATION_PROMPT = """
CRITICAL REQUIREMENT - Generate SVG Code:

1. OUTPUT FORMAT:
   - Output valid SVG code (XML format)
   - Use viewBox for sizing (e.g., viewBox="0 0 1920 1080")
   - For PPT import stability, also set width/height to match the viewBox (e.g., width="1920" height="1080")
   - NO background rectangle - transparent by default
   - Use UTF-8 encoding with proper XML declaration
   - Keep to a PPT-safe SVG subset (PowerPoint supports only part of SVG)

2. CONTENT RENDERING:
   - Use <text> elements for all text content
   - Use <rect>, <circle>, <path> for shapes and diagrams
   - Use <image> ONLY when necessary, and ONLY as base64 data URI (no external URLs)
   - Use <g> for grouping related elements
   - Prefer inline presentation attributes (fill/stroke/font-size/...) over <style> to maximize PPT compatibility

3. READABILITY STRATEGY:
   - Text with stroke for visibility: stroke="black" stroke-width="2" fill="white"
   - Prefer “double text” outline instead of filters (more PPT-compatible):
       1) outline text: fill="none" stroke="black" stroke-width="3"
       2) main text: fill="white" stroke="none"
   - Use web-safe fonts or specify fallbacks: font-family="Arial, sans-serif"
   - Ensure sufficient contrast and font weight

4. STRUCTURE:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080">
     <defs>
       <!-- Optional: gradients/patterns (avoid filters for PPT compatibility) -->
     </defs>
     <!-- Content elements -->
   </svg>
   ```

5. WHAT TO AVOID:
   - NO background <rect> filling the entire canvas
   - NO JavaScript, NO <script>, NO event handlers (onload/onclick/...)
   - NO <foreignObject>, NO external CSS, NO external image links
   - Avoid SVG filters (<filter>) and advanced features that PowerPoint may not render correctly
   - NO invalid XML characters or unclosed tags

6. EXAMPLE:
   ```xml
   <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080">
     <text x="960" y="200" font-size="72" font-weight="bold"
           text-anchor="middle" fill="white" stroke="black" stroke-width="3">
       Title Text
     </text>
     <rect x="400" y="300" width="1120" height="600"
           fill="none" stroke="#4A90E2" stroke-width="4" rx="10"/>
   </svg>
   ```
"""
```

### 1.3.1 PPT 兼容子集约束（建议固化为白名单）

PowerPoint 对 SVG 的支持是子集，设计上应“先收敛可用子集，再逐步放开”。建议在提示词与净化逻辑中同时约束：
- **推荐保留**：`<svg>` `<g>` `<rect>` `<circle>` `<ellipse>` `<line>` `<polyline>` `<polygon>` `<path>` `<text>` `<tspan>`（以及少量 `<defs>`：`<linearGradient>` `<radialGradient>` `<stop>`，如确有需要）。
- **建议禁用**：`<script>`、事件属性（`onload` 等）、`<foreignObject>`、外链资源（`href="http..."`）、`<filter>`/`<fe*>`、动画（SMIL）、外部 CSS。

> 目标不是“SVG 能力最强”，而是“PPT 中渲染稳定、可预测”。

### 1.3.2 文本换行与布局（必须可测试）

SVG 的 `<text>` 默认不会像 HTML 一样自动换行；若不规定换行策略，长段落会溢出并导致不可控。

**选定策略：提示词强制换行**

采用"提示词强制换行"方案，理由如下：
1. **更可控**：模型在生成时就考虑布局，避免后处理的不确定性
2. **更高效**：无需额外的解析和重写步骤
3. **更易测试**：可以通过提示词调整来优化效果

具体规则（在提示词中明确）：
- 标题文本：单行，最大宽度 80% viewBox 宽度
- 正文文本：使用 `<tspan>` 分行，每行最大宽度 70% viewBox 宽度
- 列表项：每项一个 `<text>` 元素，自动换行时使用 `<tspan>`
- 最大行数限制：标题 2 行，正文段落 10 行

示例提示词片段：
```
For text content:
- Titles: Single line or max 2 lines using <tspan>, each line max 80% of viewBox width
- Body text: Use <tspan dy="1.2em"> for line breaks, max 70% of viewBox width per line
- Lists: Each item as separate <text>, use <tspan> if item is too long
```

### 1.4 SVG 验证和清理

```python
def validate_and_clean_svg(svg_code: str) -> str:
    """
    验证 + 清理 + 安全净化 SVG 代码（设计稿伪代码，需与实际实现对齐）

    目标：
    - 可靠提取 SVG（支持 markdown 代码块/前后噪声）
    - XML 解析通过；根节点必须是 <svg>
    - 补齐 viewBox（必要时）
    - 移除“全画布背景矩形/面板”（包括嵌套在 <g> 内的 rect）
    - 移除高风险与低兼容元素（脚本、外链、foreignObject 等）
    """
    import re
    import xml.etree.ElementTree as ET

    # 1) 提取 SVG（markdown 代码块 + 容错截取首个 <svg>... </svg>）
    svg_match = re.search(r"```(?:xml|svg)?\s*\n(.*?)\n```", svg_code, re.DOTALL)
    if svg_match:
        svg_code = svg_match.group(1)
    if "<svg" in svg_code and "</svg>" in svg_code:
        svg_code = svg_code[svg_code.find("<svg"): svg_code.rfind("</svg>") + len("</svg>")]

    # 2) XML 解析
    # 安全兜底：拒绝 DTD/ENTITY（防止 XML 实体膨胀类 DoS），实现时可：
    # - 直接在解析前检测并拒绝包含 <!DOCTYPE / <!ENTITY 的输入
    # - 或引入 defusedxml 作为依赖（更稳健）
    try:
        root = ET.fromstring(svg_code.strip())
    except ET.ParseError as e:
        raise ValueError(f"Invalid SVG XML: {e}")

    # 3) 根节点必须是 svg（兼容带命名空间与不带命名空间）
    root_tag = root.tag.split("}", 1)[-1].lower()
    if root_tag != "svg":
        raise ValueError("Root element is not <svg>")

    # 兜底补齐 xmlns（用于规范化输出；查找元素时不要依赖是否带命名空间）
    if "xmlns" not in root.attrib:
        root.attrib["xmlns"] = "http://www.w3.org/2000/svg"

    # 4) viewBox 兜底（PPT/缩放需要）
    if "viewBox" not in root.attrib:
        width = root.attrib.get("width", "1920")
        height = root.attrib.get("height", "1080")
        # width/height 可能带单位（px），实现时需做解析/净化
        root.attrib["viewBox"] = f"0 0 {width} {height}"

    # 5) 安全净化（示意）：移除脚本/外链/事件属性/foreignObject
    # 实现建议（需要落地到可测试的确定性规则）：
    # - tag 白名单 + 属性白名单（按 tag 分组更精细）
    # - 统一移除事件属性（onload/onclick/...）
    # - 对 <image> 仅允许 data: URI，且限制 mimeType（image/png|image/jpeg）与体积上限
    # - 对 path/points 等长字符串设置长度上限，避免极端输入导致内存/性能问题

    # 6) 移除“全画布背景 rect”（必须支持嵌套删除）
    parent_map = {c: p for p in root.iter() for c in p}
    vb = root.attrib.get("viewBox", "0 0 1920 1080").split()
    vb_w, vb_h = (vb[2], vb[3]) if len(vb) == 4 else ("1920", "1080")

    def _is_full_canvas_rect(el: ET.Element) -> bool:
        if el.tag.split("}", 1)[-1].lower() != "rect":
            return False
        return (
            el.attrib.get("x", "0") in ("0", "0px")
            and el.attrib.get("y", "0") in ("0", "0px")
            and el.attrib.get("width") in (vb_w, f"{vb_w}px")
            and el.attrib.get("height") in (vb_h, f"{vb_h}px")
        )

    for el in list(root.iter()):
        if _is_full_canvas_rect(el):
            parent = parent_map.get(el)
            if parent is not None:
                parent.remove(el)

    return ET.tostring(root, encoding="unicode")
```

**建议固化为确定性白名单（可单测）**：

- **允许的 tag（PPT 友好子集）**：`svg, g, rect, circle, ellipse, line, polyline, polygon, path, text, tspan, defs, linearGradient, radialGradient, stop`
- **明确禁止的 tag**：`script, foreignObject, style, iframe, object` 以及所有动画相关元素（SMIL）
- **统一移除的属性**：
  - 所有事件属性：`onload, onclick, ...`（所有 `on*`）
  - 所有外链引用：`href/xlink:href` 只允许 `data:`；若包含 `http(s):`、`file:`、`javascript:` 直接拒绝或移除
  - `style`（建议移除，逼迫使用 inline attributes；若保留则需解析并做属性级白名单）
- **推荐允许的属性（按需扩展，优先最小集）**：
  - `<svg>`：`xmlns, width, height, viewBox, preserveAspectRatio`
  - 形状/路径：`x, y, x1, y1, x2, y2, width, height, rx, ry, cx, cy, r, d, points, fill, fill-opacity, stroke, stroke-width, stroke-opacity, opacity, transform`
  - 文本：`x, y, fill, fill-opacity, stroke, stroke-width, stroke-opacity, opacity, font-family, font-size, font-weight, text-anchor`
  - `<image>`（若允许）：`x, y, width, height, href/xlink:href(data:...), preserveAspectRatio`（并限制 data URI 体积与 mimeType）

> 评审补充：白名单必须在代码中体现为“确定性规则”，这样才能写出稳定的单元测试；否则每次净化策略变更都会产生不可控回归。

### 1.5 SVG 转 PNG（可选备份）

**依赖要求：**

SVG→PNG 建议作为**可选依赖**引入，避免影响“只用 PNG 生成”的基础安装：

- 推荐：新增 `requirements-svg.txt`（或类似命名），在需要 `--format svg/both` 且 `svg_export_png=true` 的环境中再安装
- 不建议：直接写入 `requirements.txt`，因为 `cairosvg` 在部分平台可能额外依赖系统库（cairo/pango），容易造成安装失败

可选依赖（二选一；仍需在目标运行环境验证）：

```python
# 方案1：cairosvg（渲染质量好；但在部分环境可能依赖系统 cairo 库）
cairosvg>=2.7.0

# 方案2：svglib（备选；常见依赖包括 lxml/reportlab，通常有 wheels 但仍需验证环境）
svglib>=1.5.0
```

**实现代码：**

```python
def svg_to_png(svg_path: str, png_path: str, width: int = 1920, height: int = 1080):
    """
    将 SVG 转换为 PNG（用于：
    - 生成 slides.pdf（当前实现仅支持位图）
    - 兼容旧版 Office / 不支持 SVG 的场景
    ）
    """
    try:
        import cairosvg
        cairosvg.svg2png(
            url=svg_path,
            write_to=png_path,
            output_width=width,
            output_height=height,
            background_color=None  # 透明背景
        )
    except ImportError:
        # 备选方案：使用 svglib + reportlab
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            drawing = svg2rlg(svg_path)
            renderPM.drawToFile(drawing, png_path, fmt='PNG', bg=0x00000000)
        except ImportError:
            raise ImportError(
                "SVG to PNG conversion requires either 'cairosvg' or 'svglib'. "
                "Install with: pip install cairosvg>=2.7.0"
            )
```

## 2. 实施方案

### 2.1 配置选项

扩展 `GenerationConfig` 类（当前文件为 `paper2slides/generator/config.py`，已存在 `transparent_bg` 等字段）：

```python
@dataclass
class GenerationConfig:
    # 现有字段...

    # 输出格式配置
    output_format: str = "png"  # "svg" | "png" | "both"（默认保持现有 PNG 行为）
    svg_export_png: bool = True  # SVG 模式下是否同时导出 PNG
    svg_viewbox_width: int = 1920
    svg_viewbox_height: int = 1080
```

**配置交互约束（明确定义）：**

| output_format | 行为 | 生成文件 | slides.pdf |
|---------------|------|----------|------------|
| `png` | 现有位图生成链路 | `.png` | ✅ 支持 |
| `svg` | 生成 SVG，根据 `svg_export_png` 决定是否导出 PNG | `.svg` + 可选 `.png` | ✅ 支持（需 PNG） |
| `both` | **明确定义**：保存 `.svg` + 从 SVG 栅格化得到 `.png`（确保一致性） | `.svg` + `.png` | ✅ 支持 |

**详细说明：**
- `output_format=png`：走现有位图生成链路；如需透明背景仍使用 `transparent_bg` 与后处理。
- `output_format=svg`：
  - 生成 `.svg` 文件
  - 若 `svg_export_png=true`，则从 SVG 栅格化得到 `.png`（透明 alpha），并用于生成 `slides.pdf`
  - 若 `svg_export_png=false`，则仅生成 `.svg`，slides 模式下跳过 `slides.pdf` 生成并给出提示
- `output_format=both`：
  - **明确定义**：保存 `.svg` + 从 SVG 栅格化得到 `.png`
  - 确保 SVG 和 PNG 内容一致（PNG 由 SVG 生成，而非独立生成）
  - 便于生成 `slides.pdf` 且提供 SVG 源文件

**与现有 `transparent_bg` 的关系（避免歧义，建议实现中写死规则）**：
- `transparent_bg` 及其高级参数仅对 `output_format=png` 生效（即“位图生成 + 后处理”链路）。
- 当 `output_format in {svg, both}` 且导出 PNG 时，PNG 来自 SVG 栅格化，天然具备 alpha；此时应忽略 `transparent_bg` 的后处理参数（必要时在日志中提示“已忽略”）。

### 2.2 命令行参数

```bash
# 基础用法（默认 PNG，保持现有行为）
python -m paper2slides --input paper.pdf --output slides

# 明确指定格式
python -m paper2slides --input paper.pdf --output slides --format svg
python -m paper2slides --input paper.pdf --output slides --format png
python -m paper2slides --input paper.pdf --output slides --format both

# SVG 特定选项
python -m paper2slides --input paper.pdf --output slides --format svg --no-png-export
python -m paper2slides --input paper.pdf --output slides --format svg --viewbox 1920x1080
```

### 2.3 代码修改位置

#### 文件1: `paper2slides/generator/config.py`

```python
@dataclass
class GenerationConfig:
    # ... 现有字段

    # 输出格式配置（新增）
    output_format: str = "png"  # "svg" | "png" | "both"（默认保持现有 PNG 行为）
    svg_export_png: bool = True
    svg_viewbox_width: int = 1920
    svg_viewbox_height: int = 1080
```

> 评审补充：当前 `GenerationConfig.to_dict()` 仅序列化了少量字段；若后续有任何模块依赖 `GenerationInput.to_dict()`（例如用于调试/记录），需要同步把新增字段（以及透明背景高级选项）纳入 `to_dict()`，以免“配置在日志/检查点中丢失”造成排障困难。

#### 文件2: `paper2slides/generator/image_generator.py`

**修改点1**: 添加 SVG 生成与清理（文本返回）

目标是尽量保持现有“返回 bytes + mime_type”的结构：SVG 作为 UTF-8 文本 bytes 返回，并标注 `mime_type="image/svg+xml"`，由 `generate_stage.py` 负责落盘。

```python
# 示例省略 import：from typing import List
def _generate_svg_bytes(self, prompt: str, reference_images: List[dict]) -> tuple[bytes, str]:
    full_prompt = f"{prompt}\n\n{SVG_GENERATION_PROMPT}"
    svg_text = self._call_model_for_text(full_prompt, reference_images)
    cleaned = validate_and_clean_svg(svg_text)
    return cleaned.encode("utf-8"), "image/svg+xml"
```

**修改点2**: 与现有 poster/slides 生成流程对齐

- 在 `_generate_poster()` / `_generate_slides()` 内，根据 `config.output_format` 分支调用 `_generate_svg_bytes()`（SVG）或现有 `_call_model()`（PNG/JPEG/WEBP）。
- 对 slides 的并行路径（ThreadPoolExecutor）同样需要按 `output_format` 分支，保证行为一致。
- 评审补充（slides 一致性）：当前代码会把“第 2 张 slide”作为 `style_ref_image`（base64 位图）注入后续 slide 生成提示以维持风格一致性。若切换为 SVG 文本生成，需明确二选一：
  - **继续使用位图参考**：对第 2 张 SVG 先栅格化出 PNG，再作为 `style_ref_image` 传入后续生成（实现简单、但引入 SVG→PNG 依赖与额外耗时）。
  - **改为文本参考**：将第 2 张的“SVG 片段/配色/字体定义”抽取为文本约束（更轻量，但需要更强提示词设计与稳定性验证）。

**修改点3**: `both` 与 `slides.pdf` 的闭环建议放在 `generate_stage.py`

由于当前 `save_images_as_pdf()` 只能处理位图：
- `output_format=svg && svg_export_png=true`：在 `generate_stage.py` 落盘 SVG 后，再调用 `svg_to_png()` 生成 PNG（用于 PDF 与降级）。
- `output_format=svg && svg_export_png=false`：slides 模式应跳过 `slides.pdf`，并在日志/返回值中提示。

#### 文件3: `paper2slides/main.py`

```python
# 添加命令行参数
parser.add_argument('--format', choices=['svg', 'png', 'both'], default='png',
                    help='Output format (default: png)')
parser.add_argument('--no-png-export', action='store_true',
                    help='Do not export PNG when using SVG format')
parser.add_argument('--viewbox', type=str, default='1920x1080',
                    help='SVG viewBox size (default: 1920x1080)')

# 解析 viewbox
width, height = map(int, args.viewbox.split('x'))

# 构建配置
config = GenerationConfig(
    output_format=args.format,
    svg_export_png=not args.no_png_export,
    svg_viewbox_width=width,
    svg_viewbox_height=height,
    # ... 其他参数
)
```

### 2.4 LLM API 适配

由于 SVG 生成需要文本输出而非图像输出，需要适配 LLM API（建议在 `ImageGenerator` 内新增 `_call_model_for_text()`）：

**新增方法：`_call_model_for_text()`**

```python
def _call_model_for_text(self, prompt: str, reference_images: List[dict]) -> str:
    """
    调用 LLM 生成文本（用于 SVG 代码生成）

    Args:
        prompt: 包含 SVG 生成要求的提示词
        reference_images: 参考图片列表（figures/tables）

    Returns:
        str: LLM 返回的文本内容（应包含 SVG 代码）
    """
    provider = os.getenv("IMAGE_GEN_PROVIDER", "openrouter")

    if provider == "openrouter":
        return self._call_openrouter_for_text(prompt, reference_images)
    elif provider == "google":
        return self._call_google_for_text(prompt, reference_images)
    else:
        raise ValueError(f"Unsupported provider for text generation: {provider}")
```

**1) OpenRouter 适配：**

```python
def _call_openrouter_for_text(self, prompt: str, reference_images: List[dict]) -> str:
    """OpenRouter 文本生成（用于 SVG）"""
    # 构建消息：与现有 `_call_model_openrouter()` 的 reference_images 结构对齐（base64 + mime_type）
    content = [{"type": "text", "text": prompt}]
    for img in reference_images:
        if img.get("base64") and img.get("mime_type"):
            fig_id = img.get("figure_id", "Figure")
            caption = img.get("caption", "")
            label = f"[{fig_id}]: {caption}" if caption else f"[{fig_id}]"
            content.append({"type": "text", "text": label})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img['mime_type']};base64,{img['base64']}"}
            })

    # 调用 API（不设置 IMAGE_GEN_RESPONSE_MIME_TYPE，使用默认文本响应）
    response = self.client.chat.completions.create(
        model=self.model,
        messages=[{"role": "user", "content": content}],
        max_tokens=8000,  # SVG 代码可能较长
        # 说明：是否需要 `extra_body={"modalities":["text"]}` 取决于 OpenRouter/模型；
        # 实现时可与现有 `_call_model_openrouter()` 一致，并通过配置开关控制。
    )

    # 从 message.content 读取文本
    return response.choices[0].message.content
```

**2) Google Gemini 适配：**

```python
def _call_google_for_text(self, prompt: str, reference_images: List[dict]) -> str:
    """Google Gemini 文本生成（用于 SVG）"""
    # 评审建议：为减少新增依赖，优先复用仓库现有的 `requests` 调用方式（见 `_call_model_google()`）
    # 核心差异：generationConfig.responseMimeType = "text/plain"，并从 parts[].text 读取输出。
    model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
    url = f"{self.google_api_base_url}/{model_name}:generateContent"

    parts = [{"text": prompt}]
    for img in reference_images:
        if img.get("base64") and img.get("mime_type"):
            fig_id = img.get("figure_id", "Figure")
            caption = img.get("caption", "")
            label = f"[{fig_id}]: {caption}" if caption else f"[{fig_id}]"
            parts.append({"text": label})
            parts.append({
                "inlineData": {
                    "mimeType": img["mime_type"],
                    "data": img["base64"],
                }
            })

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "text/plain",
            "maxOutputTokens": 8000,
        },
    }

    resp = requests.post(url, params={"key": self.api_key}, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    parts_out = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts_out)
```

**关键点：**
- OpenRouter：从 `response.choices[0].message.content` 读取文本
- Google Gemini：设置 `response_mime_type="text/plain"`，从 `parts[].text` 拼接
- 参考图片输入方式保持不变（OpenRouter 用 `image_url`，Gemini 用 `inlineData` 的 base64）

### 2.5 输出与 PDF 行为（必须在实现/文档中写清）

当前 `paper2slides/generator/image_generator.py` 的 `save_images_as_pdf()` 只能处理位图（PIL 打开 bytes 后转 RGB），因此：

**generate_stage.py 修改要点：**

评审补充：当前 `paper2slides/core/stages/generate_stage.py` 采用“生成时 save_callback 立即落盘”的写法。

**本次实施决策：保留 `save_callback` 方案（最小改动）**：
- callback：按 `mime_type` 写 `.svg/.png/.jpg/.webp`
- stage：生成完成后，基于 `images` 列表组装“用于 PDF 的位图列表”；若遇到 SVG 且 `svg_export_png=true`，则对已落盘的 `.svg` 栅格化出 `.png` 再读回 bytes；最后调用 `save_images_as_pdf()`

```python
async def run_generate_stage(base_dir: Path, config_dir: Path, config: Dict) -> Dict:
    # ... 现有代码 ...

    # 生成图像（注意：当前实现签名为 generator.generate(plan, gen_input, ...)）
    images = generator.generate(plan, gen_input, max_workers=config.get("max_workers", 1), save_callback=save_image_callback)

    # 保存图像并处理 SVG
    output_dir = get_output_dir(config_dir)
    saved_files = []
    images_for_pdf = []  # 用于生成 PDF 的位图（GeneratedImage）

    for img in images:
        if img.mime_type == "image/svg+xml":
            if config.get("svg_export_png", True):
                svg_path = output_dir / f"{img.section_id}.svg"  # callback 已落盘
                png_path = output_dir / f"{img.section_id}.png"
                svg_to_png(str(svg_path), str(png_path), width=..., height=...)
                saved_files.append(svg_path)
                saved_files.append(png_path)
                images_for_pdf.append(GeneratedImage(img.section_id, png_path.read_bytes(), "image/png"))
            else:
                logger.warning("SVG without PNG export: skip slides.pdf")
        else:
            images_for_pdf.append(img)
            # callback 已落盘：可选记录到 saved_files（由实现决定）

    # 生成 PDF（仅 slides 模式且有 PNG 文件）
    if config.get("output_type") == "slides" and images_for_pdf:
        pdf_path = output_dir / "slides.pdf"
        save_images_as_pdf(images_for_pdf, str(pdf_path))
        saved_files.append(pdf_path)

    return {"saved_files": [str(f) for f in saved_files]}
```

> 评审补充：`save_images_as_pdf()` 会把 RGBA 转成 RGB（PDF 不支持 alpha），默认会把透明区域“压到黑色/深色底”。如果希望 `slides.pdf` 作为更接近 PPT 观感的预览，建议在实现中把 RGBA 先合成到白底（或可配置底色）再转 RGB。

**行为总结：**

| output_format | svg_export_png | 生成文件 | slides.pdf | 说明 |
|---------------|----------------|----------|------------|------|
| `png` | N/A | `.png` | ✅ | 现有行为 |
| `svg` | `true` | `.svg` + `.png` | ✅ | PNG 由 SVG 栅格化生成 |
| `svg` | `false` | `.svg` | ❌ | 仅 SVG，给出警告 |
| `both` | N/A（强制 true） | `.svg` + `.png` | ✅ | PNG 由 SVG 栅格化生成 |

### 2.6 Web API 适配（对齐当前 `api/server.py`）

当前 Web 入口为 `POST /api/chat`（Form + 文件上传），服务端在 `generate_slides_with_pipeline()` 内构建 `config` 并调用 `run_pipeline()`。因此 SVG 相关参数需要：

- 在 `/api/chat` 的参数列表中新增 Form 字段（建议）：
  - `output_format: str = Form("png")`（`png|svg|both`）
  - `svg_export_png: Optional[str] = Form(None)`（`true|false`，默认 true）
  - `svg_viewbox: str = Form("1920x1080")`
- 在 `config = {...}` 中透传到流水线：
  - `"output_format": output_format`
  - `"svg_export_png": parsed_bool(svg_export_png, default=True)`
  - `"svg_viewbox_width"/"svg_viewbox_height": parse_viewbox(svg_viewbox)`

静态文件服务当前通过 `app.mount("/outputs", StaticFiles(...))` 暴露输出；前端应使用返回的 `relative_path` 拼接 `/outputs/{relative_path}` 进行预览/下载。

**前端适配要点：**

1. **上传界面**：添加格式选择下拉框（PNG / SVG / Both）
2. **预览 SVG**：
   ```html
   <!-- 推荐：使用 <img> 标签（安全，不执行脚本） -->
   <img src="/outputs/{relative_path}" alt="Preview" />

   <!-- 避免：<object> 或 <iframe>（可能执行脚本） -->
   ```
3. **下载**：提供 `.svg` 和 `.png` 的独立下载链接
4. **安全提示**：前端展示的 SVG 必须是服务端已净化的版本

**安全注意事项：**
- 服务端返回 SVG 时，设置正确的 Content-Type: `image/svg+xml`
- （建议）仅对 `*.svg` 响应添加安全响应头：`Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline';`
- 确保 SVG 已通过 `validate_and_clean_svg()` 净化

## 3. 优势分析

### 3.1 技术优势

1. **透明背景原生支持**
   - SVG 默认透明，无需任何后处理
   - 不存在白边、锯齿等问题

2. **完美的边缘质量**
   - 矢量图形，边缘永远平滑
   - 文字渲染清晰锐利

3. **任意缩放**
   - 放大不模糊，缩小不失真
   - 适配任何屏幕分辨率

4. **文件体积小**
   - 纯矢量（文字/形状/路径）通常比位图更小
   - 若大量嵌入 `<image>` base64，文件可能反而变大，需要在提示词与实现中约束
   - 减少存储和传输成本

5. **可编辑性强**
   - 用户可以直接编辑 SVG 代码
   - 支持后期调整颜色、文字、布局

### 3.2 用户体验优势

1. **PPT 兼容性更好**
   - PowerPoint 原生支持 SVG（Office 2016+），但支持的是 SVG 子集
   - 需要在提示词与净化中限制：避免滤镜/外链/foreignObject 等，以减少渲染差异

2. **打印质量更高**
   - 矢量图形打印不失真
   - 适合制作海报和印刷品

3. **加载速度更快**
   - 文件小，加载快
   - 渲染性能好

## 4. 风险与挑战

### 4.1 技术风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| LLM 生成无效 SVG | 高 | 中 | 严格验证 + 自动修复 + 降级到 PNG |
| 复杂图表难以生成 | 中 | 高 | 提供图表模板 + 混合模式（图表用图片嵌入） |
| 字体兼容性问题 | 中 | 中 | 使用 web-safe 字体 + 字体嵌入 |
| PPT 渲染差异（尤其 `<text>` 的描边/换行/基线） | 中 | 中 | 约束子集（少用 stroke/filter/基线属性）；必要时将关键文字转路径（牺牲可编辑性） |
| `<image>` base64 在 PPT 中兼容性不确定 | 中 | 中 | 默认禁用或严格限制 `<image>`；复杂图表降级为 PNG 并作为独立资产输出 |
| 旧版 Office 不支持 | 低 | 低 | 同时导出 PNG 备份 |
| SVG 安全与渲染风险（Web/浏览器） | 中 | 中 | 生成后做安全净化（移除脚本/外链/事件/foreignObject），再提供下载/预览 |
| SVG→PNG 栅格化依赖不稳定（cairo/lxml/平台差异） | 中 | 中 | 将 SVG→PNG 设为可选依赖；在 CI/发布环境验证；失败时允许跳过 PDF 或降级 PNG 直出 |

### 4.2 实施风险

1. **LLM 能力限制**
   - 当前图像生成模型主要输出位图
   - 需要切换到文本生成模型
   - 可能需要多次迭代优化提示词

2. **复杂内容生成**
   - 复杂图表、照片等难以用 SVG 表达
   - 解决方案：混合模式（SVG + base64 嵌入图片）

3. **向后兼容**
   - 现有 PNG 流程需要保留
   - 需要支持格式切换

### 4.3 缓解策略

1. **渐进式迁移**
   - 先支持简单的文字幻灯片
   - 逐步扩展到复杂图表
   - 保留 PNG 作为备选

2. **混合模式**
   - SVG 用于文字、形状、简单图表
   - 复杂图表和照片用 base64 嵌入
   - 最佳平衡质量和兼容性

3. **自动降级与错误处理**

**错误处理策略：**

```python
def _generate_svg_bytes(self, prompt: str, reference_images: List[dict]) -> tuple[bytes, str]:
    """生成 SVG，失败时降级到 PNG"""
    try:
        # 尝试生成 SVG
        full_prompt = f"{prompt}\n\n{SVG_GENERATION_PROMPT}"
        svg_text = self._call_model_for_text(full_prompt, reference_images)

        # 验证和清理
        cleaned = validate_and_clean_svg(svg_text)

        # 成功
        logger.info("SVG generated successfully")
        return cleaned.encode("utf-8"), "image/svg+xml"

    except ValueError as e:
        # SVG 验证失败
        logger.warning(f"SVG validation failed: {e}, falling back to PNG")
        return self._generate_png_fallback(prompt, reference_images)

    except Exception as e:
        # 其他错误（API 调用失败、模型返回非文本等）
        logger.error(f"SVG generation failed: {e}, falling back to PNG")
        return self._generate_png_fallback(prompt, reference_images)

def _generate_png_fallback(self, prompt: str, reference_images: List[dict]) -> tuple[bytes, str]:
    """降级到 PNG 生成"""
    # 移除 SVG 特定的提示词，使用原有的图像生成提示词
    png_prompt = prompt  # 或者重新构建适合位图生成的提示词
    return self._call_model(png_prompt, reference_images)
```

**降级触发条件：**
1. LLM 返回的内容不包含有效的 SVG 代码
2. SVG XML 解析失败
3. SVG 验证失败（根节点不是 `<svg>`、缺少必要属性等）
4. API 调用失败或超时

**用户通知：**
- 在日志中记录降级原因
- 在 Web API 响应中添加 `warnings` 字段
- CLI 模式下输出警告信息

## 5. 实施计划

### 5.1 开发阶段

**阶段1: 基础框架** (1天)
- [ ] 扩展配置类，添加 SVG 相关字段
- [ ] 添加命令行参数
- [ ] 实现 SVG 验证和清理函数

**阶段2: SVG 生成** (2天)
- [ ] 设计 SVG 生成提示词（包含文本换行规则）
- [ ] 实现 `_call_model_for_text()` 方法
- [ ] 实现 `_generate_svg_bytes()` 方法
- [ ] 适配 OpenRouter 和 Google Gemini API
- [ ] 测试简单幻灯片生成

**阶段3: 格式转换** (1天)
- [ ] 添加依赖到 requirements.txt（cairosvg 或 svglib）
- [ ] 实现 SVG 转 PNG 功能
- [ ] 修改 generate_stage.py 支持 SVG 保存和 PNG 导出
- [ ] 支持多格式输出（svg/png/both）
- [ ] 测试格式切换

**阶段4: Web API 和优化** (1天)
- [ ] 修改 api/server.py 支持 SVG 参数
- [ ] 前端添加格式选择界面
- [ ] 优化 SVG 代码质量
- [ ] 处理边缘情况和错误降级
- [ ] 端到端测试
- [ ] 文档更新

**总计**: 5天

### 5.2 验收标准

1. **功能完整性**
   - 能生成有效的 SVG 文件
   - SVG 背景透明（无全画布背景面板）
   - 支持 SVG/PNG/Both 三种模式（需明确 both 的 PNG 来源）
   - SVG 可在 PowerPoint 2016+ 中正常显示（以“PPT 支持的 SVG 子集”为准）

2. **质量标准**
   - SVG 语法正确，可被解析
   - 文字清晰可读（优先用描边/双层文字，避免依赖滤镜）
   - 常见图形边缘平滑（矢量路径）
   - 文件大小合理（建议设软阈值；含嵌入图片时允许更大并给出告警）

3. **兼容性**
   - PowerPoint 2016+ 支持（限制在 SVG 子集）
   - 现代浏览器支持（同时注意 SVG 的安全净化）
   - 向后兼容 PNG 模式（必要时自动降级）

### 5.3 测试建议（最小可行）

单元测试（无需真实调用外部模型，可用固定样例字符串；建议优先用标准库 `unittest`，避免引入额外测试依赖）：
1. `validate_and_clean_svg()`：
   - markdown 代码块/前后噪声提取
   - 带/不带 xmlns 的 `<svg>`
   - 背景 `<rect>` 在根节点与嵌套 `<g>` 两种情况均可移除
   - 禁止元素与属性（`<script>`、`onload`、外链 `href="http..."`）被移除/拒绝
   - 拒绝 `<!DOCTYPE` / `<!ENTITY`（防止实体膨胀/DoS）
   - `<image>` 仅允许 `data:` 且 mimeType 白名单与体积上限生效
2. SVG→PNG（若启用）：对给定 SVG 样例栅格化输出 PNG，并检查 alpha 存在（背景透明）。

建议的落地方式：
- 新增 `tests/test_svg_sanitize.py`：覆盖 `validate_and_clean_svg()` 的提取/净化/背景移除/安全拒绝用例。
- 新增 `tests/test_svg_to_png.py`：若 `cairosvg`/`svglib` 不可用则跳过；可用时输出 1 张小 PNG 并断言 alpha 通道存在。
- 运行命令：`python -m unittest discover -s tests`

端到端测试（建议保留最小人工验收步骤）：
- slides 模式：生成 `.svg` + `.png`，确认 `slides.pdf` 成功生成；导入 PowerPoint 验收（不同模板浅色/深色背景）。

## 6. 评审结论与实施前决策点

**结论**：方案整体方向正确且可行；在当前仓库结构下可以按本文档落地编码。为保证“可实现、可验证、可回滚”，实现前需把关键决策点写死，并把净化/落盘/降级路径做成可单测的确定性逻辑。

**必须先定的决策点（不建议实现时再临时决定）**：
1. `generate_stage.py` 落盘方式：保留 `save_callback` 还是改为 stage 统一落盘（见 2.5）。
2. SVG 文本生成的配置来源：复用 `IMAGE_GEN_*` 还是引入 `SVG_GEN_*`（见 1.1.2）。
3. slides 一致性：`style_ref_image` 继续用位图参考还是改为文本参考（见 2.3 修改点2）。
4. SVG→PNG 栅格化选型：`cairosvg` / `svglib` / 其他（并在目标运行环境验证依赖可用性）。
5. `slides.pdf` 预览底色：透明 PNG 在 PDF 中是合成白底还是黑底（见 2.5 评审补充）。

## 7. 后续优化

### 7.1 短期优化
- 图表模板库（预定义常用图表的 SVG 模板）
- 字体嵌入（确保跨平台一致性）
- 动画支持（SVG 支持 SMIL 动画）

### 7.2 长期优化
- 交互式编辑器（Web UI 中直接编辑 SVG）
- 主题系统（预定义配色方案）
- 批量优化（压缩 SVG 代码）

## 8. 附录

### 8.1 SVG vs PNG 对比

| 特性 | SVG | PNG |
|------|-----|-----|
| 文件类型 | 矢量 | 位图 |
| 透明背景 | 原生支持 | 需要 alpha 通道 |
| 缩放 | 无损 | 失真 |
| 文件大小 | 小（文本） | 大（像素） |
| 编辑 | 容易 | 困难 |
| 浏览器支持 | 优秀 | 优秀 |
| Office 支持 | 2016+ | 全版本 |
| 复杂图像 | 不适合 | 适合 |

### 8.2 SVG 最佳实践

1. **同时设置 viewBox + width/height（PPT 导入更稳定）**
   ```xml
   <svg width="1920" height="1080" viewBox="0 0 1920 1080">  <!-- 推荐（PPT/浏览器都稳定） -->
   <svg viewBox="0 0 1920 1080">  <!-- 备选：浏览器通常没问题，但某些软件默认尺寸不可控 -->
   ```

2. **文字描边提高可读性**
    ```xml
    <!-- 推荐：双层文字（更接近 PPT 兼容子集） -->
    <text fill="none" stroke="black" stroke-width="3">Text</text>
    <text fill="white" stroke="none">Text</text>
    ```

3. **避免滤镜，用简单元素做阴影/层次**
   ```xml
   <!-- 用“复制 + 位移 + 低透明度”模拟阴影（比 filter 更兼容） -->
   <text x="102" y="202" fill="black" opacity="0.35">Text</text>
   <text x="100" y="200" fill="white">Text</text>
   ```

4. **优化路径数据**
   - 使用相对坐标（小写命令）
   - 移除不必要的精度
   - 合并相邻路径

### 8.3 常见问题

**Q: SVG 在旧版 PowerPoint 中无法显示怎么办？**
A: 使用 `--format both` 同时生成 SVG 和 PNG，或使用 `--format png` 降级到 PNG。

**Q: 复杂图表和照片如何处理？**
A: 使用混合模式，将图片以 base64 嵌入 SVG 的 `<image>` 元素中。

**Q: SVG 文件太大怎么办？**
A: 使用 SVGO 等工具压缩，或减少路径精度。

**Q: 字体在不同电脑上显示不一致？**
A: 使用 web-safe 字体（Arial, Helvetica 等），或将字体转换为路径。

---

**文档结束**
