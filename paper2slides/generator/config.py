"""
Generator configuration and input types.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from enum import Enum

from paper2slides.summary import OriginalElements, PaperContent, GeneralContent


class OutputFormat(str, Enum):
    """Image output format for generation."""
    PNG = "png"
    SVG = "svg"
    BOTH = "both"


class OutputType(str, Enum):
    """Output type for generation."""
    POSTER = "poster"
    SLIDES = "slides"


class PosterDensity(str, Enum):
    """Content density level for poster."""
    SPARSE = "sparse"   
    MEDIUM = "medium"   
    DENSE = "dense"     


class SlidesLength(str, Enum):
    """Page count level for slides."""
    SHORT = "short"      # 5-8 pages
    MEDIUM = "medium"    # 8-12 pages
    LONG = "long"        # 12-15 pages


class StyleType(str, Enum):
    """Predefined style types."""
    ACADEMIC = "academic"
    DORAEMON = "doraemon"
    CUSTOM = "custom"


# Page count ranges for each slides length
SLIDES_PAGE_RANGES: Dict[str, tuple[int, int]] = {
    "short": (5, 8),
    "medium": (8, 12),
    "long": (12, 15),
}


@dataclass
class GenerationConfig:
    """
    User configuration for generation.
    
    Attributes:
        output_type: Type of output (poster or slides)
        poster_density: Content density for poster (sparse/medium/dense)
        slides_length: Page count level for slides (short/medium/long)
        style: Style type (academic/doraemon/custom)
        custom_style: User's custom style description (used when style=custom)
    """
    output_type: OutputType = OutputType.POSTER
    
    # Poster specific
    poster_density: PosterDensity = PosterDensity.MEDIUM
    
    # Slides specific
    slides_length: SlidesLength = SlidesLength.MEDIUM
    
    # Style
    style: StyleType = StyleType.ACADEMIC
    custom_style: Optional[str] = None

    # Generation options
    transparent_bg: bool = False

    # Output format options (default keeps existing PNG behavior)
    output_format: OutputFormat = OutputFormat.PNG
    svg_export_png: bool = True
    svg_viewbox_width: int = 1920
    svg_viewbox_height: int = 1080

    # Transparent background advanced options
    cleanup_light_panel: bool = True  # 是否启用"浅色面板去除"兜底
    panel_detect_luma: int = 220  # 面板检测亮度阈值
    content_diff_threshold: int = 25  # 与面板背景色差阈值
    edge_expand: int = 2  # 内容掩码膨胀像素
    edge_blur: float = 0.8  # 边缘平滑半径
    debug_save_intermediate: bool = False  # 保存中间结果用于调试
    fallback_to_old_behavior: bool = False  # 出错时回退到旧行为

    def get_page_range(self) -> tuple[int, int]:
        """Get page count range for slides."""
        return SLIDES_PAGE_RANGES.get(self.slides_length.value, (8, 12))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_type": self.output_type.value,
            "poster_density": self.poster_density.value,
            "slides_length": self.slides_length.value,
            "style": self.style.value,
            "custom_style": self.custom_style,
            "transparent_bg": self.transparent_bg,
            "output_format": self.output_format.value,
            "svg_export_png": self.svg_export_png,
            "svg_viewbox_width": self.svg_viewbox_width,
            "svg_viewbox_height": self.svg_viewbox_height,
            # Transparent background advanced options (for reproducibility/debugging)
            "cleanup_light_panel": self.cleanup_light_panel,
            "panel_detect_luma": self.panel_detect_luma,
            "content_diff_threshold": self.content_diff_threshold,
            "edge_expand": self.edge_expand,
            "edge_blur": self.edge_blur,
            "debug_save_intermediate": self.debug_save_intermediate,
            "fallback_to_old_behavior": self.fallback_to_old_behavior,
        }


@dataclass
class GenerationInput:
    """
    Complete input for generation.
    
    Attributes:
        config: User generation config
        content: PaperContent or GeneralContent from summary module
        origin: Original tables and figures from source_extractor
    """
    config: GenerationConfig
    content: Union[PaperContent, GeneralContent]
    origin: OriginalElements
    
    def is_paper(self) -> bool:
        """Check if content is from a paper document."""
        return isinstance(self.content, PaperContent)
    
    def get_summary_text(self) -> str:
        """Get the full summary text."""
        if isinstance(self.content, PaperContent):
            return self.content.to_summary()
        else:
            return self.content.content
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "is_paper": self.is_paper(),
            "summary": self.get_summary_text(),
            "tables": self.origin.get_table_info(),
            "figures": self.origin.get_figure_info(),
        }
