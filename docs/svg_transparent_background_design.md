# SVG 透明背景方案设计文档

## 文档信息
- **创建日期**: 2025-12-17
- **版本**: v1.0
- **作者**: Claude Code
- **相关Issue**: PNG 透明背景效果不佳，改用 SVG 矢量格式

## 0. 目标与范围

### 0.1 目标

将图像生成从 PNG 位图格式改为 SVG 矢量格式，实现：
- ✓ 真正的透明背景（无需后处理）
- ✓ 完美的边缘质量（无锯齿、无白边）
- ✓ 任意缩放不失真
- ✓ 文件体积更小
- ✓ 更好的 PPT 兼容性

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
| PPT 兼容性 | 一般 | 优秀 |

## 1. 技术方案

### 1.1 方案概述

采用 **LLM 直接生成 SVG 代码** 的方案：
1. 修改提示词，要求模型输出 SVG 代码而非图像
2. 验证和清理 SVG 代码
3. 可选：转换为 PNG 作为备份格式

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

2. CONTENT RENDERING:
   - Use <text> elements for all text content
   - Use <rect>, <circle>, <path> for shapes and diagrams
   - Use <image> for embedded images (base64 or external URLs)
   - Use <g> for grouping related elements
   - Apply proper styling via style attributes or <style> section

3. READABILITY STRATEGY:
   - Text with stroke for visibility: stroke="black" stroke-width="2" fill="white"
   - Or use <filter> for drop shadows: <feDropShadow dx="2" dy="2" stdDeviation="3"/>
   - Use web-safe fonts or specify fallbacks: font-family="Arial, sans-serif"
   - Ensure sufficient contrast and font weight

4. STRUCTURE:
   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080">
     <defs>
       <!-- Filters, gradients, patterns -->
     </defs>
     <!-- Content elements -->
   </svg>
   ```

5. WHAT TO AVOID:
   - ❌ NO background <rect> filling the entire canvas
   - ❌ NO raster image generation (this is SVG, not PNG)
   - ❌ NO JavaScript or external dependencies
   - ❌ NO invalid XML characters or unclosed tags

6. EXAMPLE:
   ```xml
   <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080">
     <text x="960" y="200" font-size="72" font-weight="bold"
           text-anchor="middle" fill="white" stroke="black" stroke-width="3">
       Title Text
     </text>
     <rect x="400" y="300" width="1120" height="600"
           fill="#4A90E2" stroke="#2E5C8A" stroke-width="4" rx="10"/>
   </svg>
   ```
"""
```

### 1.4 SVG 验证和清理

```python
def validate_and_clean_svg(svg_code: str) -> str:
    """验证和清理 SVG 代码"""
    import xml.etree.ElementTree as ET
    import re

    # 1. 提取 SVG 代码（如果 LLM 输出包含 markdown 代码块）
    svg_match = re.search(r'```(?:xml|svg)?\s*\n(.*?)\n```', svg_code, re.DOTALL)
    if svg_match:
        svg_code = svg_match.group(1)

    # 2. 验证 XML 语法
    try:
        root = ET.fromstring(svg_code)
    except ET.ParseError as e:
        raise ValueError(f"Invalid SVG XML: {e}")

    # 3. 确保有 xmlns 命名空间
    if 'xmlns' not in root.attrib:
        root.attrib['xmlns'] = 'http://www.w3.org/2000/svg'

    # 4. 确保有 viewBox（如果没有则根据 width/height 添加）
    if 'viewBox' not in root.attrib:
        width = root.attrib.get('width', '1920')
        height = root.attrib.get('height', '1080')
        root.attrib['viewBox'] = f"0 0 {width} {height}"

    # 5. 移除背景矩形（如果存在）
    for rect in root.findall('.//{http://www.w3.org/2000/svg}rect'):
        # 检查是否是全画布背景
        vb = root.attrib.get('viewBox', '0 0 1920 1080').split()
        if (rect.attrib.get('x', '0') == '0' and
            rect.attrib.get('y', '0') == '0' and
            rect.attrib.get('width') == vb[2] and
            rect.attrib.get('height') == vb[3]):
            root.remove(rect)

    # 6. 转换回字符串
    return ET.tostring(root, encoding='unicode')
```

### 1.5 SVG 转 PNG（可选备份）

```python
def svg_to_png(svg_path: str, png_path: str, width: int = 1920, height: int = 1080):
    """将 SVG 转换为 PNG（使用 cairosvg 或 Pillow）"""
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
        # 备选方案：使用 Pillow + svglib
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        drawing = svg2rlg(svg_path)
        renderPM.drawToFile(drawing, png_path, fmt='PNG', bg=0x00000000)
```

## 2. 实施方案

### 2.1 配置选项

扩展 `GenerationConfig` 类：

```python
@dataclass
class GenerationConfig:
    # 现有字段...

    # 输出格式配置
    output_format: str = "svg"  # "svg" | "png" | "both"
    svg_export_png: bool = True  # SVG 模式下是否同时导出 PNG
    svg_viewbox_width: int = 1920
    svg_viewbox_height: int = 1080
```

### 2.2 命令行参数

```bash
# 基础用法（默认 SVG）
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
    output_format: str = "svg"  # "svg" | "png" | "both"
    svg_export_png: bool = True
    svg_viewbox_width: int = 1920
    svg_viewbox_height: int = 1080
```

#### 文件2: `paper2slides/generator/image_generator.py`

**修改点1**: 添加 SVG 生成方法

```python
def _generate_svg(self, prompt: str) -> str:
    """生成 SVG 代码"""
    # 构建完整提示词
    full_prompt = f"{prompt}\n\n{SVG_GENERATION_PROMPT}"

    # 调用 LLM（使用文本生成而非图像生成）
    response = self.llm_client.generate_text(
        prompt=full_prompt,
        max_tokens=8000,  # SVG 代码可能较长
        temperature=0.7
    )

    # 验证和清理
    svg_code = validate_and_clean_svg(response)

    return svg_code
```

**修改点2**: 修改生成流程

```python
def generate_slide(self, slide_data: dict, index: int) -> str:
    """生成单个幻灯片"""
    prompt = self._build_slide_prompt(slide_data, index)

    if self.config.output_format == "svg":
        # 生成 SVG
        svg_code = self._generate_svg(prompt)
        svg_path = self._save_svg(svg_code, index)

        # 可选：导出 PNG
        if self.config.svg_export_png:
            png_path = svg_path.replace('.svg', '.png')
            svg_to_png(svg_path, png_path)

        return svg_path

    elif self.config.output_format == "png":
        # 原有 PNG 生成流程
        return self._generate_png(prompt, index)

    else:  # "both"
        svg_path = self._generate_svg(prompt)
        png_path = self._generate_png(prompt, index)
        return svg_path
```

**修改点3**: 添加 SVG 保存方法

```python
def _save_svg(self, svg_code: str, index: int) -> str:
    """保存 SVG 文件"""
    output_dir = self.paths.get_output_dir()
    svg_path = output_dir / f"slide_{index:02d}.svg"

    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(svg_code)

    logger.info(f"Saved SVG: {svg_path}")
    return str(svg_path)
```

#### 文件3: `paper2slides/main.py`

```python
# 添加命令行参数
parser.add_argument('--format', choices=['svg', 'png', 'both'], default='svg',
                    help='Output format (default: svg)')
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

由于 SVG 生成需要文本输出而非图像输出，需要适配 LLM API：

```python
class ImageGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config

        # 图像生成客户端（用于 PNG）
        self.image_client = self._init_image_client()

        # 文本生成客户端（用于 SVG）
        self.text_client = self._init_text_client()

    def _init_text_client(self):
        """初始化文本生成客户端"""
        # 使用相同的 API 配置，但调用文本生成端点
        return LLMClient(
            api_key=os.getenv("IMAGE_GEN_API_KEY"),
            base_url=os.getenv("IMAGE_GEN_BASE_URL"),
            model=os.getenv("IMAGE_GEN_MODEL", "google/gemini-2.0-flash-exp"),
        )
```

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
   - 文本描述的图形，通常比位图小 5-10 倍
   - 减少存储和传输成本

5. **可编辑性强**
   - 用户可以直接编辑 SVG 代码
   - 支持后期调整颜色、文字、布局

### 3.2 用户体验优势

1. **PPT 兼容性更好**
   - PowerPoint 原生支持 SVG（Office 2016+）
   - 插入后可直接编辑和调整

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
   - ✓ 能生成有效的 SVG 文件
   - ✓ SVG 背景透明
   - ✓ 支持 SVG/PNG/Both 三种模式
   - ✓ SVG 可在 PowerPoint 中正常显示

2. **质量标准**
   - ✓ SVG 语法正确，无错误
   - ✓ 文字清晰可读
   - ✓ 边缘平滑无锯齿
   - ✓ 文件大小合理（< 500KB/页）

3. **兼容性**
   - ✓ PowerPoint 2016+ 支持
   - ✓ 现代浏览器支持
   - ✓ 向后兼容 PNG 模式

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
   <text fill="white" stroke="black" stroke-width="2">Text</text>
   ```

3. **使用 <defs> 定义可复用元素**
   ```xml
   <defs>
     <filter id="shadow">
       <feDropShadow dx="2" dy="2" stdDeviation="3"/>
     </filter>
   </defs>
   <text filter="url(#shadow)">Text</text>
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
