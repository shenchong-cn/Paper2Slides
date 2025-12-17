# SVG 透明背景方案设计文档

## 文档信息
- **创建日期**: 2025-12-17
- **版本**: v1.1（评审修订：对齐现有代码结构、补齐兼容/安全/测试闭环）
- **作者**: Claude Code
- **相关Issue**: PNG 透明背景效果不佳，改用 SVG 矢量格式

## 0. 目标与范围

### 0.1 目标

将图像生成从 PNG 位图格式改为 SVG 矢量格式，实现：
- 真正的透明背景（无需后处理去白底/去面板）
- 更稳定的边缘质量（避免 PNG 抗锯齿白边）
- 任意缩放不失真（矢量）
- 在“纯矢量为主”场景下文件体积更小（但嵌入大图片时可能更大）
- Office 2016+ 可用（需限制在 PPT 支持的 SVG 子集）

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
3. **OpenRouter/Google 差异**：当前 OpenRouter 路径只从 `message.images` 取图像数据；要支持 SVG 文本，需要新增“从 `message.content` 取文本”的解析分支（或新建文本调用函数）。

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
   - Use viewBox for responsive sizing (e.g., viewBox="0 0 1920 1080")
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

建议策略（两者择一并在实现中定死）：
1. **提示词强制换行**：要求模型对每个文本块输出 `<tspan>` 分行（给出最大行宽/最大行数规则）。
2. **实现侧自动换行**：在生成后解析 SVG，对目标 `<text>` 节点进行“按空格/标点分词 + 近似宽度估算”的强制换行并重写为 `<tspan>`。

无论采用哪种，都应提供可自动化测试的“确定性规则”（同一输入应生成同一行切分），否则难以回归。

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
    # 实现建议：tag 白名单 + 属性白名单；对 <image> 仅允许 data: URI。

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

### 1.5 SVG 转 PNG（可选备份）

```python
def svg_to_png(svg_path: str, png_path: str, width: int = 1920, height: int = 1080):
    """
    将 SVG 转换为 PNG（用于：
    - 生成 slides.pdf（当前实现仅支持位图）
    - 兼容旧版 Office / 不支持 SVG 的场景
    ）

    注意：仓库当前 requirements.txt 未包含下述依赖，若启用该路径需补齐依赖并在文档中写清。
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
        # 备选方案：使用 svglib + reportlab（需要额外依赖 svglib）
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(svg_path)
        renderPM.drawToFile(drawing, png_path, fmt='PNG', bg=0x00000000)
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

配置交互约束（建议写死在实现与 CLI help 中，避免歧义）：
- `output_format=png`：走现有位图生成链路；如需透明背景仍使用 `transparent_bg` 与后处理。
- `output_format=svg`：生成 `.svg`；若 `svg_export_png=true`，则从 SVG 栅格化得到 `.png`（透明 alpha），并用于生成 `slides.pdf`。
- `output_format=both`：建议定义为“保存 `.svg` + 从 SVG 栅格化得到 `.png`”，确保一致且便于生成 `slides.pdf`。

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

#### 文件2: `paper2slides/generator/image_generator.py`

**修改点1**: 添加 SVG 生成与清理（文本返回）

目标是尽量保持现有“返回 bytes + mime_type”的结构：SVG 作为 UTF-8 文本 bytes 返回，并标注 `mime_type="image/svg+xml"`，由 `generate_stage.py` 负责落盘。

```python
def _generate_svg_bytes(self, prompt: str, reference_images: list[dict]) -> tuple[bytes, str]:
    full_prompt = f"{prompt}\n\n{SVG_GENERATION_PROMPT}"
    svg_text = self._call_model_for_text(full_prompt, reference_images)
    cleaned = validate_and_clean_svg(svg_text)
    return cleaned.encode("utf-8"), "image/svg+xml"
```

**修改点2**: 与现有 poster/slides 生成流程对齐

- 在 `_generate_poster()` / `_generate_slides()` 内，根据 `config.output_format` 分支调用 `_generate_svg_bytes()`（SVG）或现有 `_call_model()`（PNG/JPEG/WEBP）。
- 对 slides 的并行路径（ThreadPoolExecutor）同样需要按 `output_format` 分支，保证行为一致。

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

1) **OpenRouter**：当前实现只解析 `message.images`。需要新增：
- 当 `IMAGE_GEN_RESPONSE_MIME_TYPE` 为 `text/plain`（或 `output_format=svg`）时，从 `response.choices[0].message.content` 读取文本。
- 参考图片输入方式不变（仍可传 figures 作为 `image_url`）。

2) **Google Gemini**：当前实现会解析 `inlineData` 或“base64 in text”。SVG 模式应：
- 将 `generationConfig.responseMimeType` 设为 `text/plain`（并在 prompt 中要求直接输出 `<svg ...>` 文本）。
- 从 `candidates[0].content.parts[].text` 拼接得到 SVG 文本。

### 2.5 输出与 PDF 行为（必须在实现/文档中写清）

当前 `paper2slides/generator/image_generator.py` 的 `save_images_as_pdf()` 只能处理位图（PIL 打开 bytes 后转 RGB），因此：
- `output_format=svg && svg_export_png=false`：slides 模式应跳过 `slides.pdf` 生成，或给出明确错误提示。
- `output_format=svg && svg_export_png=true`：用 SVG 栅格化得到 PNG 后再生成 PDF（确保闭环）。
- `output_format=both`：建议定义为“保存 SVG + 从 SVG 栅格化得到 PNG”，避免同一页出现两套不一致的视觉结果。

同时需要在 `paper2slides/core/stages/generate_stage.py` 中补齐：
- 将 `image/svg+xml` 映射为 `.svg` 扩展名并正确落盘。
- 若启用 `svg_export_png`，在落盘后执行 `svg_to_png()` 并将生成的 PNG 纳入 `slides.pdf` 合成输入。
- Web/API 展示：如需在前端预览 SVG，务必使用“已净化的 SVG”，并优先用 `<img src="...">` 展示（避免 `<object>`/`<iframe>` 执行外部资源）。

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
| 旧版 Office 不支持 | 低 | 低 | 同时导出 PNG 备份 |
| SVG 安全与渲染风险（Web/浏览器） | 中 | 中 | 生成后做安全净化（移除脚本/外链/事件/foreignObject），再提供下载/预览 |

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

3. **自动降级**
   - SVG 生成失败时自动切换到 PNG
   - 提供用户选择权

## 5. 实施计划

### 5.1 开发阶段

**阶段1: 基础框架** (1天)
- [ ] 扩展配置类，添加 SVG 相关字段
- [ ] 添加命令行参数
- [ ] 实现 SVG 验证和清理函数

**阶段2: SVG 生成** (2天)
- [ ] 设计 SVG 生成提示词
- [ ] 实现 `_generate_svg` 方法
- [ ] 适配 LLM 文本生成 API
- [ ] 测试简单幻灯片生成

**阶段3: 格式转换** (1天)
- [ ] 实现 SVG 转 PNG 功能
- [ ] 支持多格式输出（svg/png/both）
- [ ] 测试格式切换

**阶段4: 优化和测试** (1天)
- [ ] 优化 SVG 代码质量
- [ ] 处理边缘情况
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

单元测试（无需真实调用外部模型，可用固定样例字符串）：
1. `validate_and_clean_svg()`：
   - markdown 代码块/前后噪声提取
   - 带/不带 xmlns 的 `<svg>`
   - 背景 `<rect>` 在根节点与嵌套 `<g>` 两种情况均可移除
   - 禁止元素与属性（`<script>`、`onload`、外链 `href="http..."`）被移除/拒绝
2. SVG→PNG（若启用）：对给定 SVG 样例栅格化输出 PNG，并检查 alpha 存在（背景透明）。

端到端测试（建议保留最小人工验收步骤）：
- slides 模式：生成 `.svg` + `.png`，确认 `slides.pdf` 成功生成；导入 PowerPoint 验收（不同模板浅色/深色背景）。

## 6. 后续优化

### 6.1 短期优化
- 图表模板库（预定义常用图表的 SVG 模板）
- 字体嵌入（确保跨平台一致性）
- 动画支持（SVG 支持 SMIL 动画）

### 6.2 长期优化
- 交互式编辑器（Web UI 中直接编辑 SVG）
- 主题系统（预定义配色方案）
- 批量优化（压缩 SVG 代码）

## 7. 附录

### 7.1 SVG vs PNG 对比

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

### 7.2 SVG 最佳实践

1. **使用 viewBox 而非固定尺寸**
   ```xml
   <svg viewBox="0 0 1920 1080">  <!-- 推荐 -->
   <svg width="1920" height="1080">  <!-- 不推荐 -->
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

### 7.3 常见问题

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
