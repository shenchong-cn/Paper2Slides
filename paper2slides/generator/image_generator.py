"""
Image Generator

Generate poster/slides images from ContentPlan.
"""
import os
import json
import base64
import time
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import GenerationInput, OutputFormat
from ..utils.api_client import create_custom_client
from .content_planner import ContentPlan, Section
from ..utils.svg_utils import (
    SvgSanitizeOptions,
    SvgValidationError,
    validate_and_clean_svg,
    svg_to_png_bytes,
)
from ..prompts.image_generation import (
    STYLE_PROCESS_PROMPT,
    FORMAT_POSTER,
    FORMAT_SLIDE,
    POSTER_STYLE_HINTS,
    SLIDE_STYLE_HINTS,
    SLIDE_LAYOUTS_ACADEMIC,
    SLIDE_LAYOUTS_DORAEMON,
    SLIDE_LAYOUTS_DEFAULT,
    SLIDE_COMMON_STYLE_RULES,
    POSTER_COMMON_STYLE_RULES,
    VISUALIZATION_HINTS,
    CONSISTENCY_HINT,
    SLIDE_FIGURE_HINT,
    POSTER_FIGURE_HINT,
)
from ..prompts.svg_generation import build_svg_generation_prompt


@dataclass
class GeneratedImage:
    """Generated image result."""
    section_id: str
    image_data: bytes
    mime_type: str


@dataclass
class ProcessedStyle:
    """Processed custom style from LLM."""
    style_name: str       # e.g., "Cyberpunk sci-fi style with high-tech aesthetic"
    color_tone: str       # e.g., "dark background with neon accents"
    special_elements: str # e.g., "Characters appear as guides" or ""
    decorations: str      # e.g., "subtle grid pattern" or ""
    valid: bool
    error: Optional[str] = None


def process_custom_style(client: OpenAI, user_style: str, model: str = None) -> ProcessedStyle:
    """Process user's custom style request with LLM."""
    model = model or os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

    # Create a dedicated RAG client for style processing
    rag_api_key = os.getenv("RAG_LLM_API_KEY")
    rag_base_url = os.getenv("RAG_LLM_BASE_URL")

    if not rag_api_key:
        return ProcessedStyle(style_name="", color_tone="", special_elements="", decorations="", valid=False, error="RAG_LLM_API_KEY not configured")

    try:
        # Use custom client for RAG processing
        rag_client = create_custom_client(rag_api_key, rag_base_url)

        response = rag_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": STYLE_PROCESS_PROMPT.format(user_style=user_style)}],
        )

        # Parse JSON from response content
        content = response.choices[0].message.content.strip()

        # Try to extract JSON from the content
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
        else:
            # If no JSON found, try parsing the entire content
            result = json.loads(content)

        return ProcessedStyle(
            style_name=result.get("style_name", ""),
            color_tone=result.get("color_tone", ""),
            special_elements=result.get("special_elements", ""),
            decorations=result.get("decorations", ""),
            valid=result.get("valid", False),
            error=result.get("error"),
        )
    except json.JSONDecodeError as e:
        return ProcessedStyle(style_name="", color_tone="", special_elements="", decorations="", valid=False, error=f"JSON parsing failed: {str(e)}")
    except Exception as e:
        return ProcessedStyle(style_name="", color_tone="", special_elements="", decorations="", valid=False, error=str(e))


class ImageGenerator:
    """Generate poster/slides images from ContentPlan."""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        response_mime_type: Optional[str] = None,
        google_api_base_url: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("IMAGE_GEN_PROVIDER", "openrouter")).lower()
        self.api_key = api_key or os.getenv("IMAGE_GEN_API_KEY", "")
        self.base_url = base_url or os.getenv("IMAGE_GEN_BASE_URL", "https://openrouter.ai/api/v1")
        self.google_api_base_url = (google_api_base_url or os.getenv("GOOGLE_GENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")).rstrip("/")
        self.response_mime_type = response_mime_type or os.getenv("IMAGE_GEN_RESPONSE_MIME_TYPE", "text/plain")
        self.model = model or os.getenv("IMAGE_GEN_MODEL")
        
        if not self.model:
            if self.provider == "google":
                # Official Gemini API image-capable default
                self.model = "models/gemini-1.5-flash"
            else:
                self.model = "google/gemini-3-pro-image-preview"
        
        if self.provider == "openrouter":
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.provider == "google":
            self.client = None
        else:
            raise ValueError(f"Unsupported image generation provider: {self.provider}")
    
    def generate(
        self,
        plan: ContentPlan,
        gen_input: GenerationInput,
        max_workers: int = 1,
        save_callback = None,
    ) -> List[GeneratedImage]:
        """
        Generate images from ContentPlan.

        Args:
            plan: ContentPlan from ContentPlanner
            gen_input: GenerationInput with config and origin
            max_workers: Maximum parallel workers for slides (3rd+ slides run in parallel)
            save_callback: Optional callback function(generated_image, index, total) called after each image

        Returns:
            List of GeneratedImage (1 for poster, N for slides)
        """
        # Save config for use in transparency processing
        self.config = gen_input.config

        figure_images = self._load_figure_images(plan, gen_input.origin.base_path)
        style_name = gen_input.config.style.value
        custom_style = gen_input.config.custom_style
        output_format = getattr(gen_input.config, "output_format", OutputFormat.PNG)
        if isinstance(output_format, str):
            output_format = OutputFormat(output_format)

        transparent_bg = bool(getattr(gen_input.config, "transparent_bg", False))
        if transparent_bg and output_format != OutputFormat.PNG:
            logging.getLogger(__name__).warning(
                "transparent_bg is only supported in PNG mode; ignoring for SVG output"
            )
            transparent_bg = False
        
        # Process custom style with LLM if needed
        processed_style = None
        if style_name == "custom" and custom_style:
            processed_style = process_custom_style(self.client, custom_style)
            if not processed_style.valid:
                raise ValueError(f"Invalid custom style: {processed_style.error}")
        
        all_sections_md = self._format_sections_markdown(plan)
        all_images = self._filter_images(plan.sections, figure_images)
        
        if plan.output_type == "poster":
            result = self._generate_poster(
                style_name,
                processed_style,
                all_sections_md,
                all_images,
                output_format,
                transparent_bg,
            )
            if save_callback and result:
                save_callback(result[0], 0, 1)
            return result
        else:
            return self._generate_slides(
                plan,
                style_name,
                processed_style,
                all_sections_md,
                figure_images,
                max_workers,
                save_callback,
                output_format,
                transparent_bg,
            )
    
    def _generate_poster(
        self,
        style_name,
        processed_style: Optional[ProcessedStyle],
        sections_md,
        images,
        output_format: OutputFormat,
        transparent_bg: bool = False,
    ) -> List[GeneratedImage]:
        """Generate 1 poster image."""
        prompt = self._build_poster_prompt(
            format_prefix=FORMAT_POSTER,
            style_name=style_name,
            processed_style=processed_style,
            sections_md=sections_md,
            transparent_bg=transparent_bg if output_format == OutputFormat.PNG else False,
        )

        if output_format in {OutputFormat.SVG, OutputFormat.BOTH}:
            svg_data, mime_type = self._generate_svg_bytes(prompt, images, strict=(output_format == OutputFormat.BOTH))
            return [GeneratedImage(section_id="poster", image_data=svg_data, mime_type=mime_type)]

        image_data, mime_type = self._call_model(prompt, images)
        if transparent_bg:
            image_data, mime_type = self._to_transparent_png(image_data, mime_type)
        return [GeneratedImage(section_id="poster", image_data=image_data, mime_type=mime_type)]
    
    def _generate_slides(
        self,
        plan,
        style_name,
        processed_style: Optional[ProcessedStyle],
        all_sections_md,
        figure_images,
        max_workers: int,
        save_callback=None,
        output_format: OutputFormat = OutputFormat.PNG,
        transparent_bg: bool = False,
    ) -> List[GeneratedImage]:
        """Generate N slide images (slides 1-2 sequential, 3+ parallel)."""
        results = []
        total = len(plan.sections)
        
        # Select layout rules based on style
        if style_name == "custom":
            layouts = SLIDE_LAYOUTS_DEFAULT
        elif style_name == "doraemon":
            layouts = SLIDE_LAYOUTS_DORAEMON
        else:
            layouts = SLIDE_LAYOUTS_ACADEMIC
        
        style_ref_image = None  # Store 2nd slide as reference for all subsequent slides
        svg_style_ref_size = (1024, 576)  # bitmap style reference for SVG mode
        
        # Generate first 2 slides sequentially (slide 1: no ref, slide 2: becomes ref)
        for i in range(min(2, total)):
            section = plan.sections[i]
            section_md = self._format_single_section_markdown(section, plan)
            layout_rule = layouts.get(section.section_type, layouts["content"])
            
            prompt = self._build_slide_prompt(
                style_name=style_name,
                processed_style=processed_style,
                sections_md=section_md,
                layout_rule=layout_rule,
                slide_info=f"Slide {i+1} of {total}",
                context_md=all_sections_md,
                transparent_bg=transparent_bg if output_format == OutputFormat.PNG else False,
            )
            
            section_images = self._filter_images([section], figure_images)
            reference_images = []
            if style_ref_image:
                reference_images.append(style_ref_image)
            reference_images.extend(section_images)

            if output_format in {OutputFormat.SVG, OutputFormat.BOTH}:
                image_data, mime_type = self._generate_svg_bytes(
                    prompt,
                    reference_images,
                    strict=(output_format == OutputFormat.BOTH),
                )
            else:
                image_data, mime_type = self._call_model(prompt, reference_images)
                if transparent_bg:
                    image_data, mime_type = self._to_transparent_png(image_data, mime_type)
            
            # Save 2nd slide (i=1) as style reference
            if i == 1:
                if mime_type == "image/svg+xml":
                    try:
                        png_bytes = svg_to_png_bytes(
                            image_data,
                            width=svg_style_ref_size[0],
                            height=svg_style_ref_size[1],
                            background_color=None,
                        )
                        style_ref_image = {
                            "figure_id": "Reference Slide",
                            "caption": "STRICTLY MAINTAIN: same background color, same accent color, same font style, same chart/icon style. Keep visual consistency.",
                            "base64": base64.b64encode(png_bytes).decode("utf-8"),
                            "mime_type": "image/png",
                        }
                    except Exception as e:
                        logging.getLogger(__name__).warning(
                            f"Failed to rasterize SVG style reference; continuing without bitmap reference: {e}"
                        )
                        style_ref_image = None
                else:
                    style_ref_image = {
                        "figure_id": "Reference Slide",
                        "caption": "STRICTLY MAINTAIN: same background color, same accent color, same font style, same chart/icon style. Keep visual consistency.",
                        "base64": base64.b64encode(image_data).decode("utf-8"),
                        "mime_type": mime_type,
                    }
            
            generated_img = GeneratedImage(section_id=section.id, image_data=image_data, mime_type=mime_type)
            results.append(generated_img)
            
            # Save immediately if callback provided
            if save_callback:
                save_callback(generated_img, i, total)
        
        # Generate remaining slides in parallel (from 3rd onwards)
        if total > 2:
            results_dict = {}
            
            def generate_single(i, section):
                section_md = self._format_single_section_markdown(section, plan)
                layout_rule = layouts.get(section.section_type, layouts["content"])
                
                prompt = self._build_slide_prompt(
                    style_name=style_name,
                    processed_style=processed_style,
                    sections_md=section_md,
                    layout_rule=layout_rule,
                    slide_info=f"Slide {i+1} of {total}",
                    context_md=all_sections_md,
                    transparent_bg=transparent_bg,
                )
                
                section_images = self._filter_images([section], figure_images)
                reference_images = [style_ref_image] if style_ref_image else []
                reference_images.extend(section_images)

                if output_format in {OutputFormat.SVG, OutputFormat.BOTH}:
                    image_data, mime_type = self._generate_svg_bytes(
                        prompt,
                        reference_images,
                        strict=(output_format == OutputFormat.BOTH),
                    )
                else:
                    image_data, mime_type = self._call_model(prompt, reference_images)
                    if transparent_bg:
                        image_data, mime_type = self._to_transparent_png(image_data, mime_type)
                return i, GeneratedImage(section_id=section.id, image_data=image_data, mime_type=mime_type)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(generate_single, i, plan.sections[i]): i
                    for i in range(2, total)
                }
                
                for future in as_completed(futures):
                    idx, generated_img = future.result()
                    results_dict[idx] = generated_img
                    
                    # Save immediately if callback provided
                    if save_callback:
                        save_callback(generated_img, idx, total)
            
            # Append in order
            for i in range(2, total):
                results.append(results_dict[i])
        
        return results
    
    def _format_custom_style_for_poster(self, ps: ProcessedStyle) -> str:
        """Format ProcessedStyle into style hints string for poster."""
        parts = [
            ps.style_name + ".",
            "English text only.",
            "Use ROUNDED sans-serif fonts for ALL text.",
            "Characters should react to or interact with the content, with appropriate poses/actions and sizes - not just decoration."
            f"LIMITED COLOR PALETTE (3-4 colors max): {ps.color_tone}.",
            POSTER_COMMON_STYLE_RULES,
        ]
        if ps.special_elements:
            parts.append(ps.special_elements + ".")
        return " ".join(parts)
    
    def _format_custom_style_for_slide(self, ps: ProcessedStyle) -> str:
        """Format ProcessedStyle into style hints string for slide."""
        parts = [
            ps.style_name + ".",
            "English text only.",
            "Use ROUNDED sans-serif fonts for ALL text.",
            "Characters should react to or interact with the content, with appropriate poses/actions and sizes - not just decoration.",
            f"LIMITED COLOR PALETTE (3-4 colors max): {ps.color_tone}.",
            SLIDE_COMMON_STYLE_RULES,
        ]
        if ps.special_elements:
            parts.append(ps.special_elements + ".")
        return " ".join(parts)
    
    def _build_poster_prompt(self, format_prefix, style_name, processed_style: Optional[ProcessedStyle], sections_md, transparent_bg: bool = False) -> str:
        """Build prompt for poster."""
        parts = [format_prefix]

        # Add transparent background instruction if requested
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

        if style_name == "custom" and processed_style:
            parts.append(f"Style: {self._format_custom_style_for_poster(processed_style)}")
            if processed_style.decorations:
                parts.append(f"Decorations: {processed_style.decorations}")
        else:
            parts.append(POSTER_STYLE_HINTS.get(style_name, POSTER_STYLE_HINTS["academic"]))

        parts.append(VISUALIZATION_HINTS)
        parts.append(POSTER_FIGURE_HINT)
        parts.append(f"---\nContent:\n{sections_md}")
        
        return "\n\n".join(parts)
    
    def _build_slide_prompt(self, style_name, processed_style: Optional[ProcessedStyle], sections_md, layout_rule, slide_info, context_md, transparent_bg: bool = False) -> str:
        """Build prompt for slide with layout rules and consistency."""
        parts = [FORMAT_SLIDE]

        # Add transparent background instruction if requested
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

        if style_name == "custom" and processed_style:
            parts.append(f"Style: {self._format_custom_style_for_slide(processed_style)}")
        else:
            parts.append(SLIDE_STYLE_HINTS.get(style_name, SLIDE_STYLE_HINTS["academic"]))
        
        # Add layout rule, then decorations if custom style
        parts.append(layout_rule)
        if style_name == "custom" and processed_style and processed_style.decorations:
            parts.append(f"Decorations: {processed_style.decorations}")
        
        parts.append(VISUALIZATION_HINTS)
        parts.append(CONSISTENCY_HINT)
        parts.append(SLIDE_FIGURE_HINT)
        
        parts.append(slide_info)
        parts.append(f"---\nFull presentation context:\n{context_md}")
        parts.append(f"---\nThis slide content:\n{sections_md}")
        
        return "\n\n".join(parts)
    
    def _format_sections_markdown(self, plan: ContentPlan) -> str:
        """Format all sections as markdown."""
        parts = []
        for section in plan.sections:
            parts.append(self._format_single_section_markdown(section, plan))
        return "\n\n---\n\n".join(parts)
    
    def _format_single_section_markdown(self, section: Section, plan: ContentPlan) -> str:
        """Format a single section as markdown."""
        lines = [f"## {section.title}", "", section.content]
        
        for ref in section.tables:
            table = plan.tables_index.get(ref.table_id)
            if table:
                focus_str = f" (focus: {ref.focus})" if ref.focus else ""
                lines.append("")
                lines.append(f"**{ref.table_id}**{focus_str}:")
                lines.append(ref.extract if ref.extract else table.html_content)
        
        for ref in section.figures:
            fig = plan.figures_index.get(ref.figure_id)
            if fig:
                focus_str = f" (focus: {ref.focus})" if ref.focus else ""
                caption = f": {fig.caption}" if fig.caption else ""
                lines.append("")
                lines.append(f"**{ref.figure_id}**{focus_str}{caption}")
                lines.append("[Image attached]")
        
        return "\n".join(lines)
    
    def _load_figure_images(self, plan: ContentPlan, base_path: str) -> List[dict]:
        """Load figure images as base64."""
        images = []
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"
        }
        
        for fig_id, fig in plan.figures_index.items():
            if base_path:
                img_path = Path(base_path) / fig.image_path
            else:
                img_path = Path(fig.image_path)
            
            if not img_path.exists():
                continue
            
            mime_type = mime_map.get(img_path.suffix.lower(), "image/jpeg")
            
            try:
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                images.append({
                    "figure_id": fig_id,
                    "caption": fig.caption,
                    "base64": img_data,
                    "mime_type": mime_type,
                })
            except Exception:
                continue
        
        return images
    
    def _filter_images(self, sections: List[Section], figure_images: List[dict]) -> List[dict]:
        """Filter images used in given sections."""
        used_ids = set()
        for section in sections:
            for ref in section.figures:
                used_ids.add(ref.figure_id)
        return [img for img in figure_images if img.get("figure_id") in used_ids]
    
    def _call_model(self, prompt: str, reference_images: List[dict]) -> tuple:
        """Call image generation provider based on configuration."""
        if self.provider == "google":
            return self._call_model_google(prompt, reference_images)
        return self._call_model_openrouter(prompt, reference_images)

    def _call_model_for_text(self, prompt: str, reference_images: List[dict], max_tokens: int) -> str:
        """Call provider for text output (used for SVG generation)."""
        provider = (self.provider or "").lower()
        if provider == "google":
            return self._call_text_google(prompt, reference_images, max_tokens=max_tokens)
        if provider == "openrouter":
            return self._call_text_openrouter(prompt, reference_images, max_tokens=max_tokens)
        raise ValueError(f"Unsupported provider for text generation: {provider}")

    def _call_text_openrouter(self, prompt: str, reference_images: List[dict], max_tokens: int) -> str:
        content = [{"type": "text", "text": prompt}]
        use_images = os.getenv("SVG_GEN_USE_REFERENCE_IMAGES", "true").lower() not in {"0", "false", "no"}
        if use_images:
            for img in reference_images:
                if img.get("base64") and img.get("mime_type"):
                    fig_id = img.get("figure_id", "Figure")
                    caption = img.get("caption", "")
                    label = f"[{fig_id}]: {caption}" if caption else f"[{fig_id}]"
                    content.append({"type": "text", "text": label})
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img['mime_type']};base64,{img['base64']}"},
                        }
                    )

        model = os.getenv("SVG_GEN_MODEL") or self.model
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
        msg = response.choices[0].message.content
        if isinstance(msg, str):
            return msg
        if isinstance(msg, list):
            parts = []
            for part in msg:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            return "\n".join(parts)
        return str(msg or "")

    def _call_text_google(self, prompt: str, reference_images: List[dict], max_tokens: int) -> str:
        model = os.getenv("SVG_GEN_MODEL") or self.model
        model_name = model if str(model).startswith("models/") else f"models/{model}"
        url = f"{self.google_api_base_url}/{model_name}:generateContent"

        parts = [{"text": prompt}]
        use_images = os.getenv("SVG_GEN_USE_REFERENCE_IMAGES", "true").lower() not in {"0", "false", "no"}
        if use_images:
            for img in reference_images:
                if img.get("base64") and img.get("mime_type"):
                    fig_id = img.get("figure_id", "Figure")
                    caption = img.get("caption", "")
                    label = f"[{fig_id}]: {caption}" if caption else f"[{fig_id}]"
                    parts.append({"text": label})
                    parts.append({"inlineData": {"mimeType": img["mime_type"], "data": img["base64"]}})

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "text/plain", "maxOutputTokens": max_tokens},
        }
        response = requests.post(url, params={"key": self.api_key}, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Google API response has no candidates (text)")
        out_parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in out_parts)

    def _generate_svg_bytes(self, prompt: str, reference_images: List[dict], strict: bool = False) -> tuple[bytes, str]:
        """
        Generate SVG bytes via text model, validate & sanitize, and return as UTF-8 bytes.

        If strict=False, may fall back to PNG output when SVG generation fails.
        """
        logger = logging.getLogger(__name__)
        view_w = int(getattr(self.config, "svg_viewbox_width", 1920))
        view_h = int(getattr(self.config, "svg_viewbox_height", 1080))
        max_tokens = int(os.getenv("SVG_GEN_MAX_TOKENS", "8000"))

        full_prompt = f"{prompt}\n\n{build_svg_generation_prompt(view_w, view_h)}"
        try:
            svg_text = self._call_model_for_text(full_prompt, reference_images, max_tokens=max_tokens)
            cleaned = validate_and_clean_svg(
                svg_text,
                options=SvgSanitizeOptions(viewbox_width=view_w, viewbox_height=view_h, allow_images=True),
            )
            return cleaned.encode("utf-8"), "image/svg+xml"
        except (SvgValidationError, ValueError) as e:
            if strict:
                raise
            # For Google provider, don't fallback to PNG as it's not supported
            if self.provider == "google":
                logger.error(f"SVG generation failed with Google API: {e}")
                raise
            logger.warning(f"SVG validation failed; falling back to PNG: {e}")
            return self._call_model(prompt, reference_images)
        except Exception as e:
            if strict:
                raise
            # For Google provider, don't fallback to PNG as it's not supported
            if self.provider == "google":
                logger.error(f"SVG generation failed with Google API: {e}")
                raise
            logger.warning(f"SVG generation failed; falling back to PNG: {e}")
            return self._call_model(prompt, reference_images)

    def _to_transparent_png(self, image_data: bytes, mime_type: str) -> tuple[bytes, str]:
        """
        Ensure the output is a PNG that actually uses transparency.

        Some image models "fake" transparency by drawing a checkerboard background into RGB,
        or return formats without alpha (e.g., JPEG). When `transparent_bg` is requested we
        convert to RGBA PNG and attempt to key out a checkerboard-like background.
        """
        try:
            from PIL import Image  # type: ignore
        except Exception:
            logging.getLogger(__name__).warning(
                "Pillow not available; cannot post-process transparency. Returning original image bytes."
            )
            return image_data, mime_type

        logger = logging.getLogger(__name__)
        try:
            img = Image.open(io.BytesIO(image_data))
            img.load()
        except Exception as e:
            logger.warning(f"Failed to decode generated image for transparency post-process: {e}")
            return image_data, mime_type

        img_rgba = img.convert("RGBA")
        alpha = img_rgba.getchannel("A")
        a_min, a_max = alpha.getextrema()

        # If the model returns alpha but uses global semi-transparency, it looks washed out in PPT.
        # Harden alpha unless there's a meaningful fully-transparent background already.
        if a_min < 255:
            sample = alpha.resize((256, 256)).getdata()
            n = 256 * 256
            zero = sum(1 for x in sample if x == 0) / n
            mid = sum(1 for x in sample if 0 < x < 255) / n
            if mid > 0.10 and zero < 0.02:
                hardened = alpha.point(lambda x: 0 if x == 0 else 255)
                img_rgba.putalpha(hardened)
                alpha = img_rgba.getchannel("A")
                a_min, a_max = alpha.getextrema()

        # Already has real transparency; normalize to PNG.
        if a_min < 255:
            buf = io.BytesIO()
            img_rgba.save(buf, format="PNG")
            result_data = buf.getvalue()
            logger.info("Image already has real transparency (alpha channel detected)")
            self._log_transparency_quality(result_data)
            return result_data, "image/png"

        # No transparency at all; try chroma-key first (if the model followed instructions).
        rgb = img_rgba.convert("RGB")
        keyed = self._key_out_chroma(rgb, key_rgb=(255, 0, 255), tol=18)
        if keyed is not None:
            buf = io.BytesIO()
            keyed.save(buf, format="PNG")
            result_data = buf.getvalue()
            logger.info("Applied chroma-key transparency (magenta #FF00FF detected)")
            self._log_transparency_quality(result_data)
            return result_data, "image/png"

        # Attempt to remove a fake checkerboard background (common "fake transparency").
        bg_colors = self._detect_checkerboard_background_colors(rgb)
        if not bg_colors:
            # Fallback: extract a large light "content card" and make outside transparent.
            card_overlay = self._extract_content_card_overlay(img_rgba)
            if card_overlay is not None:
                buf = io.BytesIO()
                card_overlay.save(buf, format="PNG")
                result_data = buf.getvalue()
                logger.info("Applied content card extraction")
                self._log_transparency_quality(result_data)
                return result_data, "image/png"

            # Final fallback: if cleanup_light_panel is enabled, try to remove light panel background
            if hasattr(self, 'config') and self.config.cleanup_light_panel:
                logger.info("Attempting light panel background removal as final fallback")
                result_data, result_mime = self._remove_light_panel_background(image_data, mime_type)
                self._log_transparency_quality(result_data)
                return result_data, result_mime

            buf = io.BytesIO()
            img_rgba.save(buf, format="PNG")
            result_data = buf.getvalue()
            logger.warning("No transparency processing applied (no alpha/chroma-key/checkerboard detected)")
            return result_data, "image/png"

        # Require 2 distinct colors; otherwise it may just be a real solid background or card panel.
        if len(bg_colors) < 2:
            buf = io.BytesIO()
            img_rgba.save(buf, format="PNG")
            result_data = buf.getvalue()
            logger.warning("Checkerboard detection failed (insufficient distinct colors)")
            return result_data, "image/png"

        bg1, bg2 = bg_colors[0], bg_colors[1]
        w, h = rgb.size
        rgb_bytes = rgb.tobytes()

        # Conservative keying: only remove near-background, light-neutral pixels.
        # This avoids washing out colored content when the model didn't actually return transparency.
        t0 = 8    # fully transparent threshold
        t1 = 28   # fully opaque threshold

        out = bytearray(w * h * 4)
        oi = 0
        for i in range(0, len(rgb_bytes), 3):
            r = rgb_bytes[i]
            g = rgb_bytes[i + 1]
            b = rgb_bytes[i + 2]

            d1 = abs(r - bg1[0]) + abs(g - bg1[1]) + abs(b - bg1[2])
            d2 = abs(r - bg2[0]) + abs(g - bg2[1]) + abs(b - bg2[2])
            d = d1 if d1 < d2 else d2

            # Only treat it as background if it's light/neutral (i.e., checkerboard-like).
            is_neutral = (max(r, g, b) - min(r, g, b) <= 18) and (r + g + b >= 640)
            if is_neutral and d <= t0:
                a = 0
            elif is_neutral and d < t1:
                a = int((d - t0) * 255 / (t1 - t0))
            else:
                a = 255

            out[oi] = r
            out[oi + 1] = g
            out[oi + 2] = b
            out[oi + 3] = a
            oi += 4

        keyed = Image.frombytes("RGBA", (w, h), bytes(out))
        buf = io.BytesIO()
        keyed.save(buf, format="PNG")
        result_data = buf.getvalue()
        logger.info("Applied checkerboard background removal")
        self._log_transparency_quality(result_data)
        return result_data, "image/png"

    def _log_transparency_quality(self, image_data: bytes):
        """Log transparency quality assessment if enabled."""
        if not hasattr(self, 'config') or not self.config.transparent_bg:
            return

        logger = logging.getLogger(__name__)
        quality = assess_transparency_quality_v2(image_data)
        logger.info(f"Transparency quality: score={quality.score:.1f}, "
                   f"light_panel={quality.has_large_light_panel}, "
                   f"edge={quality.edge_quality}")
        if quality.warnings:
            for warning in quality.warnings:
                logger.warning(f"Transparency: {warning}")

    def _remove_light_panel_background(self, image_data: bytes, mime_type: str) -> tuple[bytes, str]:
        """
        在检测到"浅色内容卡片/面板"时，尝试移除面板背景，保留内容。

        重要约束：
        - 仅作为兜底：当真 alpha / 色键 / 棋盘格检测都失败时启用
        - 不保证保留"白色内容贴在白色面板上"这种本身不可读的情况
        - 优先避免误删：宁可残留少量面板，也不应大量丢失内容
        """
        try:
            from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageDraw, ImageStat
        except Exception:
            logging.getLogger(__name__).warning(
                "Pillow not available; cannot process light panel removal."
            )
            return image_data, mime_type

        logger = logging.getLogger(__name__)
        try:
            img = Image.open(io.BytesIO(image_data)).convert("RGBA")
            w, h = img.size

            # 1) 先检测是否存在"面板/卡片"区域（下采样 + 亮度阈值）
            gray_small = ImageOps.grayscale(img.convert("RGB").resize((256, 256)))
            panel_luma = self.config.panel_detect_luma if hasattr(self, 'config') else 220
            bin_small = gray_small.point(lambda p: 255 if p >= panel_luma else 0)
            bbox = bin_small.getbbox()

            if not bbox:
                # 未检测到面板，直接返回
                return image_data, mime_type

            # 2) 将 bbox 映射回原图坐标
            x0, y0, x1, y1 = bbox
            sx, sy = w / 256.0, h / 256.0
            X0, Y0, X1, Y1 = int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)

            # 面板太小，不处理
            if X1 - X0 < w * 0.2 or Y1 - Y0 < h * 0.2:
                return image_data, mime_type

            panel = img.crop((X0, Y0, X1, Y1))
            panel_rgb = panel.convert("RGB")

            # 3) 估计面板背景色：取面板边框区域的均值色
            bw, bh = panel.size
            border = Image.new("L", (bw, bh), 0)
            draw = ImageDraw.Draw(border)
            t = max(2, min(bw, bh) // 40)  # border thickness
            draw.rectangle([0, 0, bw - 1, bh - 1], outline=255, width=t)
            stat = ImageStat.Stat(panel_rgb, mask=border)
            bg = tuple(int(v) for v in stat.mean)

            # 4) 以"与背景色的差异"判定内容
            diff = ImageChops.difference(panel_rgb, Image.new("RGB", (bw, bh), bg))
            diff_gray = ImageOps.grayscale(diff)
            content_diff_threshold = self.config.content_diff_threshold if hasattr(self, 'config') else 25
            content = diff_gray.point(lambda p: 255 if p >= content_diff_threshold else 0)

            # 5) 边缘处理：轻微膨胀包含抗锯齿边缘
            edge_expand = self.config.edge_expand if hasattr(self, 'config') else 2
            content = content.filter(ImageFilter.MaxFilter(edge_expand * 2 + 1))

            # 可选轻微平滑
            edge_blur = self.config.edge_blur if hasattr(self, 'config') else 0.8
            if edge_blur > 0:
                content = content.filter(ImageFilter.GaussianBlur(radius=edge_blur))

            # 6) 将面板内非内容设为透明（用 content 掩码作为 alpha）
            out = img.copy()
            out_alpha = out.getchannel("A")
            out_alpha.paste(content, (X0, Y0))
            out.putalpha(out_alpha)

            buf = io.BytesIO()
            out.save(buf, format="PNG")
            logger.info("Light panel background removed successfully")
            return buf.getvalue(), "image/png"

        except Exception as e:
            logger.warning(f"Failed to remove light panel background: {e}")
            # 如果启用了回滚机制，返回原图
            if hasattr(self, 'config') and self.config.fallback_to_old_behavior:
                return image_data, mime_type
            # 否则尝试返回处理后的结果（可能部分成功）
            return image_data, mime_type

    @staticmethod
    def _key_out_chroma(rgb_img, key_rgb: tuple[int, int, int], tol: int = 18):
        """
        If the key color appears on the image border, remove it by setting alpha=0.

        Returns an RGBA Image if chroma-keying was applied; otherwise None.
        """
        try:
            from PIL import Image, ImageChops, ImageOps, ImageFilter
        except Exception:
            return None

        w, h = rgb_img.size
        if w < 10 or h < 10:
            return None

        px = rgb_img.load()
        step = max(1, min(w, h) // 200)
        margin = max(1, min(w, h) // 150)

        def close(c):
            return (
                abs(c[0] - key_rgb[0]) <= tol
                and abs(c[1] - key_rgb[1]) <= tol
                and abs(c[2] - key_rgb[2]) <= tol
            )

        border = []
        for x in range(margin, w - margin, step):
            border.append(px[x, margin])
            border.append(px[x, h - 1 - margin])
        for y in range(margin, h - margin, step):
            border.append(px[margin, y])
            border.append(px[w - 1 - margin, y])

        if not border:
            return None

        ratio = sum(1 for c in border if close(c)) / len(border)
        if ratio < 0.06:
            return None

        solid = Image.new("RGB", (w, h), key_rgb)
        diff = ImageChops.difference(rgb_img, solid)
        diff_g = ImageOps.grayscale(diff)
        key_mask = diff_g.point(lambda p: 255 if p <= tol else 0)
        # Slight blur helps reduce jagged keyed edges from compression artifacts.
        key_mask = key_mask.filter(ImageFilter.GaussianBlur(radius=0.8))
        alpha = ImageOps.invert(key_mask)
        out = rgb_img.convert("RGBA")
        out.putalpha(alpha)
        return out

    @staticmethod
    def _extract_content_card_overlay(img_rgba):
        """
        Detect a large light-colored "content card" and make everything else transparent.

        This is a robust fallback when the model ignores true alpha and/or draws a checkerboard
        only inside a framed slide.
        """
        try:
            from PIL import Image, ImageOps, ImageDraw, ImageFilter
        except Exception:
            return None

        w, h = img_rgba.size
        if w < 200 or h < 200:
            return None

        # Downsample for a stable bbox; use luminance to find bright regions.
        target = (256, 256)
        gray_small = ImageOps.grayscale(img_rgba.convert("RGB").resize(target))
        # Threshold for "card-like" light background.
        bin_small = gray_small.point(lambda p: 255 if p >= 220 else 0)
        bbox = bin_small.getbbox()
        if not bbox:
            return None

        x0, y0, x1, y1 = bbox
        area_ratio = ((x1 - x0) * (y1 - y0)) / (target[0] * target[1])
        if area_ratio < 0.10 or area_ratio > 0.92:
            return None

        sx = w / target[0]
        sy = h / target[1]
        X0 = int(x0 * sx)
        Y0 = int(y0 * sy)
        X1 = int(x1 * sx)
        Y1 = int(y1 * sy)

        mx = int(w * 0.02)
        my = int(h * 0.02)
        X0 = max(0, X0 - mx)
        Y0 = max(0, Y0 - my)
        X1 = min(w - 1, X1 + mx)
        Y1 = min(h - 1, Y1 + my)

        alpha = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(alpha)
        radius = max(12, int(min(w, h) * 0.03))
        draw.rounded_rectangle([X0, Y0, X1, Y1], radius=radius, fill=255)
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=1.0))

        out = img_rgba.copy()
        out.putalpha(alpha)
        return out

    @staticmethod
    def _detect_checkerboard_background_colors(rgb_img) -> List[tuple[int, int, int]]:
        """
        Heuristically detect 1-2 dominant light-gray background colors from non-central regions.

        Returns a list of RGB tuples, ordered by frequency.
        """
        from collections import Counter

        w, h = rgb_img.size
        px = rgb_img.load()

        def is_light_neutral(c: tuple[int, int, int]) -> bool:
            r, g, b = c
            # Avoid near-white "content cards" which are often real design elements.
            if r + g + b >= 745:
                return False
            return (max(c) - min(c) <= 18) and (r + g + b >= 640)

        # Sample edges + outer ring (skip center) to catch "fake transparency" checkerboards
        # that may not touch the image boundary (e.g., inside a framed slide screenshot).
        margin = max(2, min(w, h) // 100)
        step = max(1, min(w, h) // 120)

        samples = []
        for x in range(margin, w - margin, step):
            samples.append(px[x, margin])
            samples.append(px[x, h - 1 - margin])
        for y in range(margin, h - margin, step):
            samples.append(px[margin, y])
            samples.append(px[w - 1 - margin, y])

        # Add a coarse grid across the whole image. We already filter to light-neutral colors,
        # and exclude near-white, so this tends to pick up "fake transparency" checkerboards
        # even when they appear inside a framed screenshot.
        grid_step = max(1, min(w, h) // 80)
        for y in range(margin, h - margin, grid_step):
            for x in range(margin, w - margin, grid_step):
                samples.append(px[x, y])

        if not samples:
            return []

        # Quantize a bit so near-equal colors cluster.
        def q(c: tuple[int, int, int]) -> tuple[int, int, int]:
            return (c[0] & 0xF8, c[1] & 0xF8, c[2] & 0xF8)

        counts = Counter(q(c) for c in samples if is_light_neutral(c))
        if not counts:
            return []

        candidates = [c for c, _ in counts.most_common(10)]
        chosen: List[tuple[int, int, int]] = []
        for c in candidates:
            if not chosen:
                chosen.append(c)
                continue
            # Checkerboards often use two very close light grays; be less strict here.
            if all((abs(c[0] - p[0]) + abs(c[1] - p[1]) + abs(c[2] - p[2])) > 4 for p in chosen):
                chosen.append(c)
            if len(chosen) >= 2:
                break

        if len(chosen) < 2:
            return []

        # Ensure the second color is not just noise.
        c1, c2 = chosen[0], chosen[1]
        n1, n2 = counts.get(c1, 0), counts.get(c2, 0)
        if n2 == 0 or (n2 / max(1, n1)) < 0.25:
            return []

        return chosen
    
    def _call_model_openrouter(self, prompt: str, reference_images: List[dict]) -> tuple:
        """Call the image generation model with retry logic."""
        logger = logging.getLogger(__name__)
        content = [{"type": "text", "text": prompt}]
        
        # Add each image with figure_id and caption label
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
        
        # Retry logic for API calls
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling image generation API (attempt {attempt + 1}/{max_retries})...")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    extra_body={"modalities": ["image", "text"]}
                )
                
                # Check if response is valid
                if response is None:
                    error_msg = "API returned None response - possible rate limit or API error"
                    logger.warning(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise RuntimeError(error_msg)
                
                if not hasattr(response, 'choices') or not response.choices:
                    error_msg = f"API response has no choices: {response}"
                    logger.warning(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise RuntimeError(error_msg)
                
                message = response.choices[0].message
                if hasattr(message, 'images') and message.images:
                    image_url = message.images[0]['image_url']['url']
                    if image_url.startswith('data:'):
                        header, base64_data = image_url.split(',', 1)
                        mime_type = header.split(':')[1].split(';')[0]
                        logger.info("Image generation successful")
                        return base64.b64decode(base64_data), mime_type
                
                error_msg = "Image generation failed - no images in response"
                logger.warning(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(error_msg)
                
            except Exception as e:
                logger.error(f"Error in API call (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
        
        raise RuntimeError("Image generation failed after all retry attempts")
    
    def _call_model_google(self, prompt: str, reference_images: List[dict]) -> tuple:
        """Call the official Google Gemini API for image generation."""
        logger = logging.getLogger(__name__)
        max_retries = 3
        retry_delay = 2  # seconds
        
        model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
        url = f"{self.google_api_base_url}/{model_name}:generateContent"
        
        wants_image = self.response_mime_type.lower().startswith("image/")
        model_key = model_name.split("/", 1)[-1]
        image_capable_prefixes = (
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash",
            "gemini-3-pro-image-preview",
        )
        if wants_image and not model_key.startswith(image_capable_prefixes):
            raise ValueError(
                f"Model '{model_name}' does not support image responses with the Google Gemini API. "
                "Use an image-capable model such as 'models/gemini-1.5-flash' (or -8b/pro/2.0-flash) "
                "or change IMAGE_GEN_RESPONSE_MIME_TYPE to a text type."
            )
        
        # Compose prompt parts with optional inline reference images
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
            "generationConfig": {"responseMimeType": self.response_mime_type},
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Calling Google Gemini image API (attempt {attempt + 1}/{max_retries})...")
                response = requests.post(
                    url,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=120,
                )
                
                if response.status_code >= 400:
                    logger.warning(f"Google API error {response.status_code}: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    response.raise_for_status()
                
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    error_msg = "Google API response has no candidates"
                    logger.warning(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise RuntimeError(error_msg)
                
                parts = candidates[0].get("content", {}).get("parts", [])
                for part in parts:
                    inline = part.get("inlineData")
                    if inline and inline.get("data"):
                        mime_type = inline.get("mimeType") or self.response_mime_type
                        logger.info("Image generation successful (Google Gemini)")
                        return base64.b64decode(inline["data"]), mime_type
                    
                    text_data = part.get("text")
                    if text_data:
                        try:
                            decoded = base64.b64decode(text_data, validate=True)
                            logger.info("Image generation successful (Google Gemini, text base64 payload)")
                            return decoded, self.response_mime_type
                        except Exception:
                            continue
                
                error_msg = "Image generation failed - no image payload in response"
                logger.warning(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise RuntimeError(error_msg)
            
            except Exception as e:
                logger.error(f"Error in Google API call (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
        
        raise RuntimeError("Image generation failed after all retry attempts")


@dataclass
class TransparencyQuality:
    """透明度质量评估结果"""
    score: float  # 0-100分
    has_large_light_panel: bool  # 是否存在大面积浅色面板残留
    light_panel_ratio: float  # 不透明区域中"浅色低饱和"比例（诊断项）
    edge_quality: str  # "smooth" | "jagged"（诊断项）
    warnings: List[str]


def assess_transparency_quality_v2(img_data: bytes) -> TransparencyQuality:
    """
    评估透明度质量（不依赖 numpy）

    Args:
        img_data: PNG图像数据（bytes）

    Returns:
        TransparencyQuality: 质量评估结果
    """
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(img_data)).convert("RGBA")

        # 下采样做诊断，避免全分辨率遍历过慢
        img = img.resize((256, 256))
        alpha_data = list(img.getchannel("A").getdata())
        rgb_data = list(img.convert("RGB").getdata())

        # 检查是否还有"大面积浅色面板/背景"
        opaque_idx = [i for i, a in enumerate(alpha_data) if a > 200]

        if opaque_idx:
            light_neutral = 0
            for i in opaque_idx:
                r, g, b = rgb_data[i]
                # 浅色：RGB总和>660，低饱和：最大最小差<30
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
        semi_transparent = sum(1 for a in alpha_data if 10 < a < 245)
        semi_ratio = semi_transparent / len(alpha_data)
        if semi_ratio > 0.02:
            score += 20
            edge_quality = "smooth"
        else:
            score += 5
            edge_quality = "jagged"

        # 透明度覆盖（额外加分）
        transparent = sum(1 for a in alpha_data if a < 10)
        trans_ratio = transparent / len(alpha_data)
        if trans_ratio > 0.20:
            score += 20
        elif trans_ratio > 0.10:
            score += 10

        warnings = []
        if has_large_light_panel:
            warnings.append("Large light panel/background remains (may look like a white rectangle in PPT)")
        if edge_quality == "jagged":
            warnings.append("Edges may appear jagged (consider adjusting edge_blur parameter)")

        return TransparencyQuality(
            score=min(100, score),  # 限制最高100分
            has_large_light_panel=has_large_light_panel,
            light_panel_ratio=light_panel_ratio,
            edge_quality=edge_quality,
            warnings=warnings
        )

    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to assess transparency quality: {e}")
        return TransparencyQuality(
            score=0,
            has_large_light_panel=False,
            light_panel_ratio=0.0,
            edge_quality="unknown",
            warnings=[f"Assessment failed: {str(e)}"]
        )


def save_images_as_pdf(images: List[GeneratedImage], output_path: str):
    """
    Save generated images as a single PDF file.

    Args:
        images: List of GeneratedImage from ImageGenerator.generate()
        output_path: Output PDF file path
    """
    from PIL import Image
    import io

    pdf_images = []

    for img in images:
        # Load image from bytes
        pil_img = Image.open(io.BytesIO(img.image_data))

        # Convert RGBA to RGB (PDF doesn't support alpha)
        if pil_img.mode == 'RGBA':
            background = Image.new("RGB", pil_img.size, (255, 255, 255))
            background.paste(pil_img, mask=pil_img.getchannel("A"))
            pil_img = background
        elif pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        pdf_images.append(pil_img)

    if pdf_images:
        # Save first image and append the rest
        pdf_images[0].save(
            output_path,
            save_all=True,
            append_images=pdf_images[1:] if len(pdf_images) > 1 else [],
            resolution=100.0,
        )
        print(f"PDF saved: {output_path}")
