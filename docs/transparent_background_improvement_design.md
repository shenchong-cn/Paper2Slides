# 透明背景效果改进设计文档

## 文档信息
- **创建日期**: 2025-12-16
- **版本**: v2.2 (评审修订：明确配置设计、补充 poster 模式、完善实施细节)
- **作者**: Claude Code
- **相关Issue**: 透明背景PNG效果不佳 - 白色背景不透明

## 0. 目标与范围

### 0.1 目标（面向用户体验）

当用户启用 `--transparent-bg` 生成 PNG 并插入 PPT 时：
- PNG 的“背景区域”应为真正透明（alpha=0），不出现白色矩形/面板
- 只显示内容元素（文字/图表/图形/图标），像“浮在 PPT 模板上”
- 不要求对所有 PPT 背景都 100% 可读，但应通过描边/阴影/配色让大多数背景可读

### 0.2 非目标（本次不做）

- 不引入深度学习分割模型（如 U-Net）做前景分割
- 不追求对“白色文字 + 白色面板”这类本身不可读的内容进行“无中生有式”修复

## 1. 问题分析

### 1.1 问题现象

用户使用`--transparent-bg`参数生成slides后，将PNG插入PPT时发现：
- PNG显示为白色矩形，无法透出PPT背景色
- 无法实现"内容浮在PPT背景上"的效果
- 透明效果很差，不符合预期

### 1.2 技术分析

对生成的`slide_01.png`进行详细分析：

```
模式: RGBA ✓
尺寸: 1376x768

整体透明度分布:
- 完全不透明(alpha=255): 756,218像素 (71.6%)
- 完全透明(alpha=0): 284,946像素 (27.0%)
- 半透明(1-254): 15,604像素 (1.5%)

不透明区域的颜色分析:
- 白色背景(RGB>240): 569,413像素 (75.3%)  ⚠️ 问题所在
- 浅色背景(RGB>220): 639,178像素 (84.5%)
- 有色内容(RGB<=220): 117,040像素 (15.5%)  ← 实际内容

中心区域(50%x50%)分析:
- 完全不透明: 100.0%
- 中心不透明区域的白色背景: 75.0%  ⚠️ 核心问题
```

### 1.3 根本原因

**问题本质**: 模型生成了一个**白色背景的内容卡片**，而不是真正的透明背景。

具体表现：
1. **内容卡片设计**: 模型按照提示词生成了一个白色/浅色的内容卡片
2. **卡片背景不透明**: 卡片内部的白色背景alpha=255（完全不透明）
3. **只有边缘透明**: 只有卡片外围的边缘区域是透明的（27%）
4. **PPT插入效果差**: 插入PPT时显示为白色矩形，无法透出PPT背景

**为什么会这样**:

当前提示词（`image_generator.py:365-377`）要求：
```python
"For readability on arbitrary slide templates, place all content on a centered rounded-rectangle "
"content card (e.g., white at ~90–95% opacity) and keep margins outside the card fully transparent."
```

这个提示词的问题：
- ❌ 要求生成"白色内容卡片"
- ❌ 卡片内部不透明
- ❌ 只有卡片外围透明
- ✓ 只有边缘是透明的（这不是用户想要的）

**用户真正需要的**:
- ✓ 整个背景都透明（alpha=0）
- ✓ 只保留文字、图表等实际内容
- ✓ 内容直接浮在PPT背景上
- ✓ 没有白色卡片背景

**实现侧的放大因素（现有代码行为）**：
- 透明背景流程当前同时“提示模型使用 #FF00FF 纯色背景做色键”并“提示生成白色 content card”，目标互相矛盾：色键需要整幅背景统一、面板需要大面积不透明浅色区域。
- 后处理 `_to_transparent_png` 在“识别不到色键/棋盘格”时，会回退到 `_extract_content_card_overlay`：把“卡片外部”设为透明、保留整张卡片（这会在 PPT 中表现为白色矩形），与用户期望相反。

### 1.4 对比分析

| 项目 | 当前效果 | 用户期望 |
|------|---------|---------|
| 背景透明度 | 只有边缘透明(27%) | 整个背景透明(>80%) |
| 内容卡片 | 有白色卡片背景 | 无卡片，内容直接浮动 |
| 白色背景 | 不透明(alpha=255) | 透明(alpha=0) |
| PPT插入效果 | 白色矩形 | 内容浮在PPT背景上 |
| 适配性 | 只适合浅色PPT | 适配任何颜色PPT |

## 2. 解决方案设计

### 2.1 方案概述

采用**两阶段策略**：
1. **阶段1**: 修改提示词，要求模型生成无背景的内容
2. **阶段2**: 后处理在必要时去除“浅色面板/内容卡片”，只保留实际内容（兜底）

> 设计原则：尽量让模型“直接生成可用透明背景”，后处理只做保守兜底；避免把后处理变成“无监督抠图器”导致误删内容。

### 2.2 阶段1: 提示词优化

#### 当前提示词的问题

```python
# 当前提示词 (image_generator.py:365-377)
"For readability on arbitrary slide templates, place all content on a centered rounded-rectangle "
"content card (e.g., white at ~90–95% opacity) and keep margins outside the card fully transparent."
```

这个提示词**明确要求**生成白色卡片，导致问题。

#### 新提示词设计（兼容色键兜底，不再要求 content card）

```python
TRANSPARENT_BG_PROMPT_V2 = """
CRITICAL REQUIREMENT - True Transparent Background:

1. OUTPUT FORMAT:
   - PNG with alpha channel (RGBA)
   - Prefer a true transparent background (alpha=0) for ALL background areas
   - If true alpha cannot be reliably produced, use a single flat chroma-key background color #FF00FF (magenta)
   - NEVER use #FF00FF anywhere in content (text/shapes/charts/icons)
   - NO large solid background panels/cards (no white card, no rounded rectangle panel, no page-like backdrop)

2. CONTENT RENDERING:
   - Render ONLY the actual content: text, charts, diagrams, icons
   - All content should be directly rendered on a transparent canvas (or on #FF00FF only if needed for chroma-key)
   - Avoid large filled rectangles behind content
   - Text must remain legible on various PPT templates by using outlines/strokes and/or shadows
   - Charts and diagrams should use vibrant colors

3. READABILITY STRATEGY:
   Since the background is transparent and will be placed on various PPT templates:
   - Use text with outlines/strokes for visibility (e.g., white text with dark outline, or dark text with light outline)
   - Use subtle shadows (not large glows) to separate content from background
   - Prefer medium-to-bold fonts and high-contrast colors
   - Charts and diagrams should have clear borders and fills

4. WHAT TO AVOID:
   - ❌ NO white/light content card behind everything
   - ❌ NO rounded rectangle “panel” or page background
   - ❌ NO background gradients/textures
   - ❌ NO checkerboard pattern
   - ❌ NO semi-transparent full-canvas overlays
   - ❌ NO pure white text without an outline/shadow (white-on-white cannot be recovered by post-processing)

5. VERIFICATION:
   - The final image should show ONLY content (text/charts/diagrams)
   - Everything else should be transparent (alpha=0) OR exactly #FF00FF for chroma-key (only if needed)
   - When placed on a colored background, only the content should be visible (no white rectangle)

EXAMPLE: Imagine rendering text and charts directly on a transparent canvas in Photoshop with no background layer.
"""
```

**关键改进**:
- ❌ 移除“内容卡片”要求（避免生成白色矩形）
- ✓ 明确“真透明优先，#FF00FF 色键兜底可选”（与现有后处理兼容）
- ✓ 提供文字可读性的替代方案（描边、阴影、粗体）
- ✓ 明确列出要避免的内容
- ✓ 提供具体的验证标准

### 2.3 阶段2: 后处理算法

即使优化了提示词，模型仍可能输出“浅色内容卡片/面板”。后处理应优先利用现有的两类强信号：
1) **真 alpha**（模型真的输出透明背景）或 **#FF00FF 色键**（模型按提示给出统一背景）
2) 仅在 1) 失败、且检测到“大面积浅色面板”时，才启用“面板去除”兜底

#### 核心算法: 面板去除（保守兜底，不做通用抠图）

```python
def _remove_light_panel_background(self, image_data: bytes, mime_type: str) -> tuple[bytes, str]:
    """
    在检测到“浅色内容卡片/面板”时，尝试移除面板背景，保留内容。

    重要约束：
    - 仅作为兜底：当真 alpha / 色键 / 棋盘格检测都失败时启用
    - 不保证保留“白色内容贴在白色面板上”这种本身不可读的情况
    - 优先避免误删：宁可残留少量面板，也不应大量丢失内容
    """
    from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageDraw, ImageStat
    import io

    img = Image.open(io.BytesIO(image_data)).convert("RGBA")
    w, h = img.size

    # 1) 先检测是否存在“面板/卡片”区域（复用现有 card bbox 思路：下采样 + 亮度阈值）
    # 2) 若不存在明显面板，直接返回（避免误删）
    # 3) 若存在面板：
    #    - 估计面板的“背景颜色”（取面板 bbox 边缘/角落的多数色）
    #    - 在面板内构建“内容掩码”：与背景色差异显著的像素视为内容
    #    - 对内容掩码做轻微膨胀（包含抗锯齿边缘），再可选轻微模糊平滑边缘
    #    - 将面板内“非内容”设为透明；面板外保持透明/原样
    #
    # 注：不依赖 numpy；如需加速可选引入 numpy，但不作为强制依赖。

    # 示例：用灰度阈值找亮区域 bbox（与现有 _extract_content_card_overlay 同方向）
    gray_small = ImageOps.grayscale(img.convert("RGB").resize((256, 256)))
    bin_small = gray_small.point(lambda p: 255 if p >= 220 else 0)
    bbox = bin_small.getbbox()
    if not bbox:
        return image_data, mime_type

    # 4) 将 bbox 映射回原图坐标
    x0, y0, x1, y1 = bbox
    sx, sy = w / 256.0, h / 256.0
    X0, Y0, X1, Y1 = int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)
    if X1 - X0 < w * 0.2 or Y1 - Y0 < h * 0.2:
        return image_data, mime_type  # 面板太小，不处理

    panel = img.crop((X0, Y0, X1, Y1))
    panel_rgb = panel.convert("RGB")

    # 5) 估计面板背景色：取面板边框区域的均值色（更不容易被正文污染）
    bw, bh = panel.size
    border = Image.new("L", (bw, bh), 0)
    draw = ImageDraw.Draw(border)
    t = max(2, min(bw, bh) // 40)  # border thickness
    draw.rectangle([0, 0, bw - 1, bh - 1], outline=255, width=t)
    stat = ImageStat.Stat(panel_rgb, mask=border)
    bg = tuple(int(v) for v in stat.mean)

    # 6) 以“与背景色的差异”判定内容，而不是简单亮度阈值
    diff = ImageChops.difference(panel_rgb, Image.new("RGB", (bw, bh), bg))
    diff_gray = ImageOps.grayscale(diff)
    content_diff_threshold = 25  # 可配置
    content = diff_gray.point(lambda p: 255 if p >= content_diff_threshold else 0)

    # 7) 边缘处理：轻微膨胀包含抗锯齿边缘，再可选轻微平滑
    edge_expand = 2  # 可配置
    content = content.filter(ImageFilter.MaxFilter(edge_expand * 2 + 1))
    edge_blur = 0.8  # 可配置（建议很小）
    if edge_blur > 0:
        content = content.filter(ImageFilter.GaussianBlur(radius=edge_blur))

    # 8) 将面板内非内容设为透明（用 content 掩码作为 alpha）
    out = img.copy()
    out_alpha = out.getchannel("A")
    out_alpha.paste(content, (X0, Y0))
    out.putalpha(out_alpha)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue(), "image/png"
```

#### 算法特点

1. **优先强信号**: 先走“真 alpha / #FF00FF 色键 / 棋盘格”，最后才做面板去除兜底
2. **以“背景色差异”判定内容**: 以“与估计面板背景色的差值”识别前景，避免简单亮度阈值误删浅色内容
3. **保守策略**: 无法判断的像素更倾向保留（避免误删）
4. **可控边缘**: 掩码膨胀 + 轻微平滑，尽量降低锯齿，同时避免产生明显白边光晕

#### 关于"白边/光晕（Color Fringing）"

对 alpha 做模糊会产生半透明像素；若这些像素的 RGB 仍是面板白色，会在有色 PPT 背景上出现白边。

**初版实施（v1.0）**：
- ✅ 先做内容掩码的膨胀（包含抗锯齿边缘），尽量减少对"面板背景 RGB"做半透明化
- ✅ 使用较小的边缘平滑半径（edge_blur=0.8），避免产生大量半透明像素

**后续优化（v2.0+，可选增强）**：
- 对最终 alpha=0 的像素将 RGB 置零（或置为邻域内容色）以减少残留背景色的影响
- 实现"去污染"算法：对半透明像素的 RGB 进行调整，减少白边效果
- 这些优化需要更复杂的算法，初版暂不实施

### 2.4 配置选项

#### 配置类设计（扩展现有 GenerationConfig）

直接扩展现有的 `GenerationConfig` 类，避免引入新的配置类：

```python
@dataclass
class GenerationConfig:
    """用户生成配置"""
    output_type: OutputType = OutputType.POSTER
    poster_density: PosterDensity = PosterDensity.MEDIUM
    slides_length: SlidesLength = SlidesLength.MEDIUM
    style: StyleType = StyleType.ACADEMIC
    custom_style: Optional[str] = None

    # 透明背景配置（现有 + 新增）
    transparent_bg: bool = False  # 是否启用透明背景（现有）
    cleanup_light_panel: bool = True  # 是否启用"浅色面板去除"兜底（新增）
    panel_detect_luma: int = 220  # 面板检测亮度阈值（新增）
    content_diff_threshold: int = 25  # 与面板背景色差阈值（新增）
    edge_expand: int = 2  # 内容掩码膨胀像素（新增）
    edge_blur: float = 0.8  # 边缘平滑半径（新增，建议很小）

    # 调试与回滚（新增）
    debug_save_intermediate: bool = False  # 保存中间结果用于调试（新增）
    fallback_to_old_behavior: bool = False  # 出错时回退到旧行为（新增）
```

**设计理由**：
- ✅ 保持现有架构，避免引入新类增加复杂度
- ✅ 所有透明背景相关配置集中管理
- ✅ 向后兼容：新字段有默认值，不影响现有代码

#### 命令行参数

```bash
# 基础透明背景
--transparent-bg              # 启用透明背景

# 高级控制（可选）
--keep-light-panel            # 保留浅色面板（关闭兜底清理）
--panel-luma 220              # 面板检测亮度阈值
--content-diff 25             # 内容/面板背景色差阈值
--edge-expand 2               # 内容掩码膨胀像素
--edge-blur 0.8               # 边缘平滑半径

# 调试与回滚（可选）
--debug-transparency          # 保存中间结果（面板检测、掩码等）
--fallback-on-error           # 出错时回退到旧行为
```

### 2.5 质量评估与后续动作

#### 质量评估算法

```python
@dataclass
class TransparencyQuality:
    """透明度质量评估"""
    score: float  # 0-100分
    has_large_light_panel: bool  # 是否存在大面积浅色面板残留
    light_panel_ratio: float  # 不透明区域中"浅色低饱和"比例（诊断项）
    edge_quality: str  # "smooth" | "jagged"（诊断项）
    warnings: List[str]

def assess_transparency_quality_v2(img: Image) -> TransparencyQuality:
    """评估透明度质量（不依赖 numpy）"""
    # 下采样做诊断，避免全分辨率遍历过慢
    img = img.convert("RGBA").resize((256, 256))
    alpha = list(img.getchannel("A").getdata())
    rgb = list(img.convert("RGB").getdata())

    # 检查是否还有"大面积浅色面板/背景"
    opaque_idx = [i for i, a in enumerate(alpha) if a > 200]
    if opaque_idx:
        light_neutral = 0
        for i in opaque_idx:
            r, g, b = rgb[i]
            if (r + g + b > 660) and (max(r, g, b) - min(r, g, b) < 30):
                light_neutral += 1
        light_panel_ratio = light_neutral / len(opaque_idx)
        has_large_light_panel = light_panel_ratio > 0.30
    else:
        light_panel_ratio = 0.0
        has_large_light_panel = False

    # 评分
    score = 0
    score += 60 if not has_large_light_panel else 25

    # 边缘质量
    semi_transparent = sum(1 for a in alpha if 10 < a < 245)
    semi_ratio = semi_transparent / len(alpha)
    if semi_ratio > 0.02:
        score += 20
        edge_quality = "smooth"
    else:
        score += 5
        edge_quality = "jagged"

    warnings = []
    if has_large_light_panel:
        warnings.append("Large light panel/background remains (may look like a white rectangle in PPT)")

    return TransparencyQuality(
        score=score,
        has_large_light_panel=has_large_light_panel,
        light_panel_ratio=light_panel_ratio,
        edge_quality=edge_quality,
        warnings=warnings
    )
```

#### 质量评估的后续动作

**初版实施（v1.0）**：
- ✅ 记录到日志：`logger.info(f"Transparency quality: {quality.score}/100, warnings: {quality.warnings}")`
- ✅ 保存到输出元数据：在生成的 `state.json` 中添加 `transparency_quality` 字段
- ❌ 不自动重试（避免增加复杂度和成本）

**后续优化（v2.0+）**：
- 低分时（score < 60）可选自动重试一次（需要用户配置 `--retry-on-low-quality`）
- 提供 Web UI 实时反馈质量评分
- 支持用户手动触发重新生成

> 注：透明像素比例（transparent_ratio）不作为硬指标，因为全幅图表/大图片天然不透明像素多；更关键的是"是否存在大面积浅色面板残留"。

## 3. 技术实现

### 3.1 代码修改位置

#### 文件1: `paper2slides/generator/image_generator.py`

**修改点1**: 替换透明背景提示词（内联方式，不提取常量）

在 `_build_slide_prompt` 方法（365-377行）和 `_build_poster_prompt` 方法中：
```python
# 当前代码（365-377行）
if transparent_bg:
    parts.append(
        "IMPORTANT: Output a PNG with a TRUE transparent background (alpha channel). "
        "Do NOT draw any checkerboard/grid pattern to represent transparency. "
        "Background pixels must have alpha=0; only the content (text/figures/shapes) should be opaque. "
        "For readability on arbitrary slide templates, place all content on a centered rounded-rectangle "
        "content card (e.g., white at ~90–95% opacity) and keep margins outside the card fully transparent."
    )
    parts.append(
        "IMPLEMENTATION HINT: Set the entire canvas background to a pure chroma-key color #FF00FF (magenta) "
        "and NEVER use #FF00FF anywhere else in the design. Do not use gradients/textures on the background. "
        "This background will be programmatically converted to transparency."
    )

# 修改为（使用新提示词，移除 content card 要求）
if transparent_bg:
    parts.append(
        "CRITICAL REQUIREMENT - True Transparent Background:\n"
        "1. OUTPUT FORMAT: PNG with alpha channel (RGBA). Prefer true transparent background (alpha=0) "
        "for ALL background areas. If true alpha cannot be reliably produced, use a single flat chroma-key "
        "background color #FF00FF (magenta). NEVER use #FF00FF in content. NO large solid background panels/cards.\n"
        "2. CONTENT RENDERING: Render ONLY actual content (text/charts/diagrams/icons) directly on transparent "
        "canvas (or on #FF00FF if needed). Avoid large filled rectangles behind content. Text must remain legible "
        "on various PPT templates by using outlines/strokes and/or shadows. Charts should use vibrant colors.\n"
        "3. READABILITY: Use text with outlines/strokes (e.g., white text with dark outline, or dark text with "
        "light outline). Use subtle shadows. Prefer medium-to-bold fonts and high-contrast colors.\n"
        "4. AVOID: NO white/light content card, NO rounded rectangle panel, NO background gradients/textures, "
        "NO checkerboard pattern, NO pure white text without outline/shadow."
    )
```

**设计理由**：
- ✅ 提示词直接内联在方法中，与现有代码风格一致
- ✅ 避免引入新的常量文件或模块
- ✅ Slides 和 Poster 模式使用相同的透明背景提示词
- ✅ 简化后的提示词更紧凑，减少 token 消耗

**修改点2**: 添加"浅色面板去除"方法（新增，约 80 行）
```python
def _remove_light_panel_background(self, image_data: bytes, mime_type: str) -> tuple[bytes, str]:
    """在检测到浅色面板时移除面板背景，保留内容（保守兜底）"""
    # 实现见 2.3 节算法描述
```

**修改点3**: 修改 `_to_transparent_png` 方法（476-589行）
```python
def _to_transparent_png(self, image_data: bytes, mime_type: str) -> tuple[bytes, str]:
    """
    改进的透明度处理

    流程:
    1. 若已有真实 alpha：可选检测是否仍存在"大面积浅色面板"，必要时做兜底清理
    2. 若无 alpha：先尝试 #FF00FF 色键（强信号）
    3. 再尝试去除"假透明棋盘格"
    4. 最后：仅在检测到浅色面板且 cleanup_light_panel=True 时，启用面板去除兜底
    """
    # 在现有逻辑的最后（534-543行）添加面板去除兜底
    if self.config.cleanup_light_panel:
        return self._remove_light_panel_background(image_data, mime_type)
```

**修改点4**: 在生成流程中添加质量评估（192-194, 234-236, 276-278行）
```python
if transparent_bg:
    image_data, mime_type = self._to_transparent_png(image_data, mime_type)
    # 添加质量评估和日志
    quality = assess_transparency_quality_v2(Image.open(io.BytesIO(image_data)))
    logger.info(f"Transparency quality: score={quality.score}/100, panel_ratio={quality.light_panel_ratio:.2%}, edge={quality.edge_quality}")
    if quality.warnings:
        logger.warning(f"Transparency warnings: {', '.join(quality.warnings)}")
    # 保存到元数据（可选）
    if hasattr(self, 'current_metadata'):
        self.current_metadata['transparency_quality'] = {
            'score': quality.score,
            'has_large_light_panel': quality.has_large_light_panel,
            'warnings': quality.warnings
        }
```

#### 文件2: `paper2slides/generator/config.py`

**修改**: 扩展 `GenerationConfig` 类（47-84行）
```python
@dataclass
class GenerationConfig:
    """用户生成配置"""
    output_type: OutputType = OutputType.POSTER
    poster_density: PosterDensity = PosterDensity.MEDIUM
    slides_length: SlidesLength = SlidesLength.MEDIUM
    style: StyleType = StyleType.ACADEMIC
    custom_style: Optional[str] = None

    # 透明背景配置（现有 + 新增）
    transparent_bg: bool = False
    cleanup_light_panel: bool = True  # 新增
    panel_detect_luma: int = 220  # 新增
    content_diff_threshold: int = 25  # 新增
    edge_expand: int = 2  # 新增
    edge_blur: float = 0.8  # 新增
    debug_save_intermediate: bool = False  # 新增
    fallback_to_old_behavior: bool = False  # 新增
```

#### 文件3: `paper2slides/main.py`

**新增**: 命令行参数解析（约在 argparse 部分）
```python
# 透明背景基础参数（现有）
parser.add_argument('--transparent-bg', action='store_true', help='Generate with transparent background')

# 透明背景高级参数（新增）
parser.add_argument('--keep-light-panel', action='store_true',
                    help='Keep light panel (disable cleanup fallback)')
parser.add_argument('--panel-luma', type=int, default=220,
                    help='Panel detection luminance threshold (default: 220)')
parser.add_argument('--content-diff', type=int, default=25,
                    help='Content/background color difference threshold (default: 25)')
parser.add_argument('--edge-expand', type=int, default=2,
                    help='Content mask dilation pixels (default: 2)')
parser.add_argument('--edge-blur', type=float, default=0.8,
                    help='Edge smoothing radius (default: 0.8)')

# 调试与回滚参数（新增）
parser.add_argument('--debug-transparency', action='store_true',
                    help='Save intermediate results for debugging')
parser.add_argument('--fallback-on-error', action='store_true',
                    help='Fallback to old behavior on error')

# 在构建 GenerationConfig 时传递参数
config = GenerationConfig(
    transparent_bg=args.transparent_bg,
    cleanup_light_panel=not args.keep_light_panel,  # 反转逻辑
    panel_detect_luma=args.panel_luma,
    content_diff_threshold=args.content_diff,
    edge_expand=args.edge_expand,
    edge_blur=args.edge_blur,
    debug_save_intermediate=args.debug_transparency,
    fallback_to_old_behavior=args.fallback_on_error,
    # ... 其他参数
)
```

#### 文件4: `api/server.py`（Web API）

**修改**: 在请求处理中支持新参数
```python
# 在 /generate 端点中解析请求参数
config = GenerationConfig(
    transparent_bg=request.transparent_bg,
    cleanup_light_panel=request.cleanup_light_panel,
    panel_detect_luma=request.panel_luma or 220,
    content_diff_threshold=request.content_diff or 25,
    # ... 其他参数
)
```

### 3.2 实施步骤

#### 步骤1: 配置类扩展 (优先级P0, 0.5天)
1. 扩展 `GenerationConfig` 类，添加透明背景相关字段
2. 在 `main.py` 中添加命令行参数解析
3. 在 `api/server.py` 中添加 Web API 参数支持
4. 单元测试：验证配置传递正确

#### 步骤2: 提示词优化 (优先级P0, 0.5天)
1. 修改 `_build_slide_prompt` 和 `_build_poster_prompt` 中的透明背景提示词
2. 移除"content card"要求，强调真透明 + 描边/阴影
3. 测试模型响应（生成 3-5 个样本）
4. 根据结果微调提示词

#### 步骤3: 核心算法实现 (优先级P0, 1.5天)
1. 实现 `_remove_light_panel_background` 方法（约 80 行）
2. 实现 `assess_transparency_quality_v2` 函数（约 50 行）
3. 修改 `_to_transparent_png` 方法，集成面板去除兜底
4. 调优阈值：面板检测、内容/背景色差、边缘处理
5. 单元测试：色键抠图、面板去除、质量评估

#### 步骤4: 集成和测试 (优先级P0, 1天)
1. 在生成流程中添加质量评估和日志
2. 实现调试模式（保存中间结果）
3. 实现回滚机制（出错时回退到旧行为）
4. 端到端测试：生成完整 slides/poster 并验证透明度
5. 视觉测试：在多种 PPT 模板上验证效果

#### 步骤5: 文档和优化 (优先级P1, 0.5天)
1. 更新 README 和使用文档
2. 添加透明背景使用示例
3. 性能优化（如需要）
4. 代码审查和清理

**总计时间估算**: 4-5 天（含风险缓冲）

### 3.3 测试策略

#### 单元测试
```python
def test_key_out_magenta_background():
    """测试 #FF00FF 色键抠图：背景透明、内容保留"""

def test_remove_light_panel_background_basic():
    """测试浅色面板去除：面板透明、黑字/彩色图表保留"""

def test_remove_light_panel_preserve_outlined_text():
    """测试白字+深色描边：即使面板去除也能保留（依赖提示词约束）"""

def test_quality_assessment_warns_on_large_panel():
    """测试质量评估：残留大面积浅色面板时给出告警"""
```

#### 集成测试
```python
def test_end_to_end_transparency():
    """端到端测试"""
    # 生成slides
    # 检查透明度质量
    # 验证无大面积浅色面板残留（避免白色矩形）
```

#### 视觉测试
1. 在不同颜色的PPT模板上测试（深蓝、深红、黑色、浅灰等）
2. 检查文字可读性
3. 验证边缘平滑度
4. 确认无白色矩形

### 3.4 性能考虑与 numpy 依赖策略

#### 性能分析

- 后处理耗时与图像尺寸线性相关
- 典型图像（1376x768）处理时间：
  - 下采样检测：~10ms
  - 面板去除（如触发）：~50-100ms
  - 质量评估：~20ms
- 总体性能影响：≤20%（符合目标）

#### numpy 依赖策略

**初版实施（v1.0）**：
- ✅ **不依赖 numpy**：所有算法使用 Pillow 原生操作
- ✅ 代码简洁，依赖少，易于维护
- ✅ 性能对大多数用户可接受

**后续优化（v2.0+，可选）**：
- 如用户反馈性能问题，可提供 numpy 加速版本
- 实现策略：
  ```python
  try:
      import numpy as np
      HAS_NUMPY = True
  except ImportError:
      HAS_NUMPY = False

  def _remove_light_panel_background(self, ...):
      if HAS_NUMPY:
          return self._remove_light_panel_background_numpy(...)
      else:
          return self._remove_light_panel_background_pillow(...)
  ```
- 降级路径：numpy 不可用时自动回退到 Pillow 实现

**优化措施**:
- 下采样检测（256x256）用于面板判断/定位，避免全分辨率重计算
- 仅在检测到问题时启用兜底清理，减少不必要开销
- 使用 Pillow 的高效操作（`point()`, `filter()`, `paste()`）

## 4. 预期效果

### 4.1 改进前后对比

| 指标 | 改进前 | 改进后（目标） | 说明 |
|------|--------|--------|----------|
| 透明像素比例 | 27% | 不设硬目标 | 取决于版面内容（全幅图表会更不透明） |
| 不透明浅色面板占比（诊断） | 75% | <15%（告警阈值 30%） | 避免 PPT 中出现白色矩形/面板感 |
| PPT插入效果 | 白色矩形 | 内容浮动 | 质的飞跃 |
| 透明度质量评分 | 40/100 | ≥80/100 | 面向大多数页面的目标 |
| 适配PPT模板 | 仅浅色 | 任意颜色 | 通用性提升 |

### 4.2 用户体验改进

**改进前**:
```
用户: 插入PPT后看到白色矩形，无法使用
评分: ⭐⭐ (2/5)
```

**改进后**:
```
用户: 内容完美浮在PPT背景上，效果很好
评分: ⭐⭐⭐⭐⭐ (5/5)
```

## 5. 风险评估

### 5.1 技术风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 误删浅色前景（白字/浅灰线条） | 高 | 中 | 提示词强制描边/阴影；后处理以“背景色差异”判定内容且保守；提供 `--keep-light-panel` 退回旧行为 |
| 文字可读性下降 | 中 | 中 | 通过提示词要求描边/阴影；必要时提供可选的文字增强（后续项） |
| 白边/光晕 | 中 | 中 | 避免对背景 RGB 做半透明化；必要时做 RGB 去污染/置零（可选增强） |
| 性能下降 | 低 | 低 | 下采样检测 + 仅在触发兜底时处理；如需更高性能可选引入 numpy |

### 5.2 兼容性风险

- **向后兼容**: 保持现有API，新功能通过参数控制
- **默认行为**: `--transparent-bg` 默认启用"浅色面板去除"兜底（以避免白色矩形）
- **退出机制**: 提供 `--keep-light-panel` 参数保留旧行为（仅色键/棋盘格，不做面板去除）
- **回滚机制**: 提供 `--fallback-on-error` 参数，出错时自动回退到旧行为
- **调试支持**: 提供 `--debug-transparency` 参数，保存中间结果用于问题诊断

## 6. 实施计划

### 6.1 开发阶段（总计 4-5 天）

**阶段1: 配置类扩展** (0.5天)
- [x] 分析问题根因
- [ ] 扩展 `GenerationConfig` 类
- [ ] 添加命令行参数解析
- [ ] 添加 Web API 参数支持

**阶段2: 提示词优化** (0.5天)
- [ ] 修改 slides 和 poster 提示词
- [ ] 测试模型响应
- [ ] 微调提示词

**阶段3: 核心算法实现** (1.5天)
- [ ] 实现 `_remove_light_panel_background` 方法
- [ ] 实现 `assess_transparency_quality_v2` 函数
- [ ] 修改 `_to_transparent_png` 方法
- [ ] 调优阈值和边缘处理
- [ ] 单元测试

**阶段4: 集成和测试** (1天)
- [ ] 集成质量评估和日志
- [ ] 实现调试模式和回滚机制
- [ ] 端到端测试
- [ ] 视觉测试（多种 PPT 模板）

**阶段5: 文档和优化** (0.5天)
- [ ] 更新 README 和文档
- [ ] 性能优化（如需要）
- [ ] 代码审查

**风险缓冲**: +1天（应对调试、返工、意外问题）

### 6.2 验收标准

1. **功能完整性**
   - ✓ PPT 插入后不出现“白色矩形/面板”
   - ✓ 真 alpha / #FF00FF 色键 / 棋盘格 三条路径至少一条稳定生效
   - ✓ 面板兜底启用时，内容不发生大面积丢失（保守优先）

2. **质量标准**
   - ✓ 透明度质量评分 ≥ 80/100（90%的cases）
   - ✓ 在深色/浅色 PPT 模板上大多数页面可读（依赖描边/阴影提示词约束）
   - ✓ `has_large_light_panel` 为 False（或仅轻微告警）

3. **性能标准**
   - ✓ 处理时间增加 ≤ 20%
   - ✓ 内存使用增加 ≤ 30%

4. **用户体验**
   - ✓ PPT插入效果符合预期
   - ✓ 提供清晰的质量反馈
   - ✓ 配置选项易于理解

## 7. 后续优化

### 7.1 短期优化
- 文字描边增强（提高深色背景上的可读性）
- 自适应阈值（根据图像内容自动调整）
- 批量处理优化

### 7.2 长期优化
- 使用深度学习模型进行内容分割（U-Net等）
- 支持矢量格式输出（SVG）
- 提供在线预览工具

## 8. 附录

### 8.1 算法流程图

```
输入: 生成的图像 (RGBA)
  ↓
是否已有真实透明(alpha<255)?
  ├─ 是：可选检测“大面积浅色面板”
  │     ├─ 无：直接输出 PNG
  │     └─ 有：启用“浅色面板去除”兜底 → 输出 PNG
  └─ 否：
        ↓
    尝试 #FF00FF 色键抠图（强信号）
        ├─ 成功：输出 PNG
        └─ 失败：
              ↓
          尝试检测/去除“假透明棋盘格”
              ├─ 成功：输出 PNG
              └─ 失败：
                    ↓
                检测浅色面板/内容卡片 bbox
                    ├─ 未检测到：原样输出 PNG（避免误删）
                    └─ 检测到：面板内按“与背景色差异”提取内容 → 输出 PNG
  ↓
质量评估
  ├─ 是否残留大面积浅色面板（has_large_light_panel）
  ├─ 浅色面板占比（light_panel_ratio，诊断项）
  └─ 边缘质量
  ↓
输出: 真正透明的PNG
```

### 8.2 阈值选择指南

| 阈值类型 | 推荐值 | 说明 |
|---------|--------|------|
| 色键容差（chroma key tol） | 18 | 与现有 `_key_out_chroma(..., tol=18)` 对齐 |
| 面板检测亮度阈值（panel_detect_luma） | 220 | 下采样灰度阈值，用于找浅色面板 bbox |
| 内容/背景色差阈值（content_diff_threshold） | 25 | 以“与估计面板背景色差异”判断内容 |
| 掩码膨胀（edge_expand） | 2 | 包含抗锯齿边缘，减少锯齿 |
| 边缘平滑（edge_blur） | 0.8 | 建议很小，过大易产生白边/光晕 |

### 8.3 常见问题

**Q: 会不会误删浅色文字？**
A: 有可能，尤其是“白色/浅灰文字贴在白色面板上”这种本身不可读的情况。为降低风险：
1) 提示词要求文字必须有描边/阴影（避免白-on-白）
2) 后处理以“与面板背景色差异”判定内容，并采取保守策略
3) 提供 `--keep-light-panel` 关闭兜底清理以回退旧行为

**Q: 深色背景上的文字可读性如何保证？**
A: 可以启用文字增强选项，添加描边或阴影。或者在提示词中要求使用浅色文字。

**Q: 性能影响大吗？**
A: 取决于图像尺寸与是否触发兜底清理。默认仅在检测到“大面积浅色面板”时额外处理；如需更高性能可选引入 numpy。

**Q: 可以保留旧的行为吗？**
A: 可以，使用 `--keep-light-panel` 参数关闭浅色面板去除兜底。

---

**文档结束**
