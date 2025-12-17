"""
Prompts for SVG generation.

The goal is a PPT-friendly SVG subset with transparent background.
"""

from __future__ import annotations


def build_svg_generation_prompt(viewbox_width: int, viewbox_height: int) -> str:
    return f"""
CRITICAL REQUIREMENT - Generate SVG code ONLY (no explanations).

1) OUTPUT FORMAT:
- Output a single complete SVG document (XML).
- Must include: width="{viewbox_width}" height="{viewbox_height}" viewBox="0 0 {viewbox_width} {viewbox_height}"
- Transparent background by default: DO NOT draw a full-canvas background rectangle.
- Prefer inline presentation attributes (fill/stroke/font-size/...) over <style>.

2) PPT-SAFE SUBSET:
- Allowed tags: svg, g, rect, circle, ellipse, line, polyline, polygon, path, text, tspan, defs, linearGradient, radialGradient, stop, image
- Forbidden: script, foreignObject, style, filters (<filter>/<fe*>), animations, external CSS, external URLs
- No JavaScript, no event handlers (no onload/onclick/...)

3) TEXT & READABILITY:
- Use <text>/<tspan> for all text (no <foreignObject>).
- Use web-safe fonts: font-family="Arial, sans-serif"
- Ensure text is readable on varied PPT backgrounds:
  - Prefer double-layer outline text:
    1) outline text: fill="none" stroke="black" stroke-width="3"
    2) main text: fill="white" stroke="none"

4) IMAGES:
- Avoid <image> unless absolutely necessary.
- If used, only allow data: URIs (base64) and avoid large payloads.

5) STRUCTURE:
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{viewbox_width}" height="{viewbox_height}" viewBox="0 0 {viewbox_width} {viewbox_height}">
  <!-- content -->
</svg>
""".strip()

