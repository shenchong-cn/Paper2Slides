"""
SVG utilities for generation.

Includes:
- Extraction from noisy LLM output
- Safety-focused sanitization for PPT-friendly subset
- Optional SVG->PNG rasterization (for PDF export and style reference images)
"""

from __future__ import annotations

import base64
import re
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import xml.etree.ElementTree as ET


_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"


class SvgValidationError(ValueError):
    pass


def extract_svg(text: str) -> str:
    """
    Extract the first SVG document from a model output.

    Supports:
    - Markdown fenced blocks (```xml|svg)
    - Noisy prefixes/suffixes around a <svg>...</svg> region
    """
    if not text:
        raise SvgValidationError("Empty SVG output")

    fenced = re.search(r"```(?:xml|svg)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)

    start = text.find("<svg")
    end = text.rfind("</svg>")
    if start == -1 or end == -1 or end <= start:
        raise SvgValidationError("No <svg>...</svg> block found")

    return text[start : end + len("</svg>")].strip()


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1].lower()


def _is_event_attr(name: str) -> bool:
    return name.lower().startswith("on")


def _parse_number(value: str) -> Optional[float]:
    """
    Parse a numeric SVG length.
    - Accepts plain numbers and px.
    - Returns None for percentages or unknown units.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("%"):
        return None
    s = re.sub(r"px$", "", s, flags=re.IGNORECASE)
    try:
        return float(s)
    except Exception:
        return None


def _viewbox_dims(view_box: str) -> Tuple[float, float]:
    parts = (view_box or "").replace(",", " ").split()
    if len(parts) != 4:
        raise SvgValidationError(f"Invalid viewBox: {view_box!r}")
    try:
        return float(parts[2]), float(parts[3])
    except Exception as e:
        raise SvgValidationError(f"Invalid viewBox numbers: {view_box!r}") from e


def _get_attr_any_ns(el: ET.Element, names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in el.attrib:
            return el.attrib.get(name)
    return None


def _validate_data_image_href(href: str, max_data_bytes: int) -> str:
    if not href:
        raise SvgValidationError("Empty image href")
    if not href.startswith("data:"):
        raise SvgValidationError("External image href is not allowed")

    m = re.match(r"^data:([^;]+);base64,(.*)$", href, re.DOTALL | re.IGNORECASE)
    if not m:
        raise SvgValidationError("Only base64 data: URIs are allowed for images")

    mime = (m.group(1) or "").lower().strip()
    if mime not in {"image/png", "image/jpeg", "image/jpg"}:
        raise SvgValidationError(f"Disallowed embedded image mimeType: {mime}")

    b64 = m.group(2).strip()
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise SvgValidationError("Invalid base64 in embedded image") from e
    if len(raw) > max_data_bytes:
        raise SvgValidationError(f"Embedded image too large: {len(raw)} bytes")

    return href


@dataclass(frozen=True)
class SvgSanitizeOptions:
    viewbox_width: int = 1920
    viewbox_height: int = 1080
    allow_images: bool = True
    max_embedded_image_bytes: int = 2 * 1024 * 1024  # 2MB


def validate_and_clean_svg(svg_text: str, options: SvgSanitizeOptions) -> str:
    """
    Validate, sanitize, and normalize an SVG for PPT-friendly rendering.

    Policy:
    - Reject DTD/ENTITY outright (avoid entity expansion / DoS)
    - Keep a small tag/attribute subset
    - Remove script/foreignObject/style and any event handlers
    - Remove full-canvas background rects to keep transparency
    """
    svg_code = extract_svg(svg_text)

    lowered = svg_code.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise SvgValidationError("DOCTYPE/ENTITY is not allowed in SVG")

    try:
        root = ET.fromstring(svg_code)
    except ET.ParseError as e:
        raise SvgValidationError(f"Invalid SVG XML: {e}") from e

    if _local_name(root.tag) != "svg":
        raise SvgValidationError("Root element is not <svg>")

    # Ensure namespaces exist for consistent serialization.
    if "xmlns" not in root.attrib:
        root.attrib["xmlns"] = _SVG_NS
    ET.register_namespace("", _SVG_NS)
    ET.register_namespace("xlink", _XLINK_NS)

    # Normalize viewBox + width/height for PPT stability.
    if "viewBox" not in root.attrib:
        root.attrib["viewBox"] = f"0 0 {options.viewbox_width} {options.viewbox_height}"
    vb_w, vb_h = _viewbox_dims(root.attrib.get("viewBox", ""))
    root.attrib["width"] = str(int(round(vb_w)))
    root.attrib["height"] = str(int(round(vb_h)))

    allowed_tags = {
        "svg",
        "g",
        "rect",
        "circle",
        "ellipse",
        "line",
        "polyline",
        "polygon",
        "path",
        "text",
        "tspan",
        "defs",
        "lineargradient",
        "radialgradient",
        "stop",
        "image",
    }
    # Attributes are intentionally strict; expand only as needed.
    common_attrs = {
        "id",
        "class",
        "opacity",
        "transform",
        "fill",
        "fill-opacity",
        "stroke",
        "stroke-width",
        "stroke-opacity",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-dasharray",
        "stroke-dashoffset",
    }
    svg_attrs = {"xmlns", "width", "height", "viewbox", "preserveaspectratio"}
    shape_attrs = {
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "rx",
        "ry",
        "cx",
        "cy",
        "r",
        "d",
        "points",
    }
    text_attrs = {
        "x",
        "y",
        "dx",
        "dy",
        "text-anchor",
        "font-family",
        "font-size",
        "font-weight",
        "dominant-baseline",
    }
    gradient_attrs = {"x1", "y1", "x2", "y2", "cx", "cy", "r", "fx", "fy", "gradientunits"}
    stop_attrs = {"offset", "stop-color", "stop-opacity"}
    image_attrs = {"x", "y", "width", "height", "preserveaspectratio", "href"}

    def allowed_attrs_for(tag: str) -> set[str]:
        if tag == "svg":
            return svg_attrs | common_attrs
        if tag in {"g"}:
            return common_attrs
        if tag in {"rect", "circle", "ellipse", "line", "polyline", "polygon", "path"}:
            return common_attrs | shape_attrs
        if tag in {"text", "tspan"}:
            return common_attrs | text_attrs
        if tag in {"lineargradient", "radialgradient"}:
            return common_attrs | gradient_attrs
        if tag == "stop":
            return common_attrs | stop_attrs
        if tag == "image":
            return common_attrs | image_attrs
        return common_attrs

    def should_remove_full_canvas_rect(el: ET.Element) -> bool:
        if _local_name(el.tag) != "rect":
            return False
        fill = (el.attrib.get("fill") or "").strip().lower()
        if not fill or fill == "none":
            return False

        x = el.attrib.get("x", "0")
        y = el.attrib.get("y", "0")
        w = el.attrib.get("width")
        h = el.attrib.get("height")
        if w is None or h is None:
            return False

        def is_zero(v: str) -> bool:
            n = _parse_number(v)
            return n is not None and abs(n) < 1e-6

        def is_full_dim(v: str, dim: float) -> bool:
            s = str(v).strip()
            if s == "100%":
                return True
            n = _parse_number(s)
            return n is not None and abs(n - dim) < 1e-3

        return is_zero(x) and is_zero(y) and is_full_dim(w, vb_w) and is_full_dim(h, vb_h)

    # Build parent map for removals.
    parent_map = {child: parent for parent in root.iter() for child in parent}

    # Remove invalid tags and unsafe attributes.
    for el in list(root.iter()):
        tag = _local_name(el.tag)

        if tag not in allowed_tags:
            parent = parent_map.get(el)
            if parent is not None:
                parent.remove(el)
            continue

        # Drop full-canvas background rectangles (including nested).
        if should_remove_full_canvas_rect(el):
            parent = parent_map.get(el)
            if parent is not None:
                parent.remove(el)
            continue

        # Attribute sanitization.
        allowed = allowed_attrs_for(tag)
        to_delete = []
        to_set = {}
        for raw_key, raw_val in el.attrib.items():
            key_local = raw_key.split("}", 1)[-1].lower()

            if _is_event_attr(key_local):
                to_delete.append(raw_key)
                continue
            if key_local == "style":
                to_delete.append(raw_key)
                continue
            if key_local not in allowed:
                # allow namespaced href via normalization below
                if tag == "image" and key_local in {"href"}:
                    pass
                else:
                    to_delete.append(raw_key)
                    continue

            if tag == "image" and key_local in {"href"}:
                if not options.allow_images:
                    raise SvgValidationError("<image> is not allowed by policy")
                cleaned_href = _validate_data_image_href(str(raw_val), options.max_embedded_image_bytes)
                to_set[raw_key] = cleaned_href

        for k in to_delete:
            el.attrib.pop(k, None)
        for k, v in to_set.items():
            el.attrib[k] = v

        # Normalize xlink:href to href if present (and sanitized).
        if tag == "image":
            href = _get_attr_any_ns(el, ["href", f"{{{_XLINK_NS}}}href"])
            if href:
                el.attrib["href"] = _validate_data_image_href(str(href), options.max_embedded_image_bytes)
                el.attrib.pop(f"{{{_XLINK_NS}}}href", None)

    xml = ET.tostring(root, encoding="unicode")
    if not xml.lstrip().startswith("<?xml"):
        xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml
    return xml


def svg_to_png_bytes(svg_bytes: bytes, width: int, height: int, background_color: Optional[str] = None) -> bytes:
    """
    Rasterize SVG bytes into PNG bytes.

    Tries cairosvg first (preferred), then svglib+reportlab as a fallback.
    """
    try:
        import cairosvg  # type: ignore

        return cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=width,
            output_height=height,
            background_color=background_color,
        )
    except ImportError:
        pass

    try:
        from svglib.svglib import svg2rlg  # type: ignore
        from reportlab.graphics import renderPM  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=True) as f:
            f.write(svg_bytes)
            f.flush()
            drawing = svg2rlg(f.name)
        png = renderPM.drawToString(drawing, fmt="PNG", bg=0x00000000)
        return png
    except ImportError as e:
        raise ImportError(
            "SVG->PNG rasterization requires 'cairosvg' or 'svglib'. "
            "Install one of them to enable SVG export to PNG/PDF and bitmap style references."
        ) from e

