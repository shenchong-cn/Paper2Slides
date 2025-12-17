import base64
import unittest
import xml.etree.ElementTree as ET

from paper2slides.utils.svg_utils import (
    SvgSanitizeOptions,
    SvgValidationError,
    extract_svg,
    validate_and_clean_svg,
)


class TestSvgSanitize(unittest.TestCase):
    def test_extract_svg_from_fenced_block(self):
        raw = "prefix\n```xml\n<svg viewBox=\"0 0 10 10\"></svg>\n```\nsuffix"
        svg = extract_svg(raw)
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))

    def test_removes_full_canvas_background_rect_and_unsafe_bits(self):
        raw = """```xml
<svg viewBox="0 0 1920 1080">
  <rect x="0" y="0" width="1920" height="1080" fill="white"/>
  <g>
    <rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>
  </g>
  <text x="10" y="20" onclick="alert(1)" fill="black">Hello</text>
  <script>alert(1)</script>
</svg>
```"""
        cleaned = validate_and_clean_svg(raw, options=SvgSanitizeOptions(viewbox_width=1920, viewbox_height=1080))
        root = ET.fromstring(cleaned.split("\n", 1)[-1])  # drop xml declaration for parsing

        # Ensure root looks like SVG and has width/height
        self.assertEqual(root.tag.split("}", 1)[-1].lower(), "svg")
        self.assertEqual(root.attrib.get("width"), "1920")
        self.assertEqual(root.attrib.get("height"), "1080")

        # Ensure unsafe bits are removed
        for el in root.iter():
            tag = el.tag.split("}", 1)[-1].lower()
            self.assertNotEqual(tag, "script")
            self.assertFalse(any(k.lower().startswith("on") for k in el.attrib.keys()))

        # Ensure full-canvas rects are removed (only rects in input are full-canvas)
        self.assertFalse(any(el.tag.split('}', 1)[-1].lower() == "rect" for el in root.iter()))

    def test_rejects_doctype(self):
        raw = "<svg viewBox=\"0 0 10 10\"><!DOCTYPE svg></svg>"
        with self.assertRaises(SvgValidationError):
            validate_and_clean_svg(raw, options=SvgSanitizeOptions(viewbox_width=10, viewbox_height=10))

    def test_rejects_external_image_href(self):
        raw = """<svg viewBox="0 0 10 10">
  <image x="0" y="0" width="10" height="10" href="https://example.com/x.png"/>
</svg>"""
        with self.assertRaises(SvgValidationError):
            validate_and_clean_svg(raw, options=SvgSanitizeOptions(viewbox_width=10, viewbox_height=10))

    def test_allows_small_data_image_href(self):
        payload = base64.b64encode(b"not-a-real-png").decode("ascii")
        raw = f"""<svg viewBox="0 0 10 10">
  <image x="0" y="0" width="10" height="10" href="data:image/png;base64,{payload}"/>
  <text x="1" y="2">ok</text>
</svg>"""
        cleaned = validate_and_clean_svg(
            raw,
            options=SvgSanitizeOptions(viewbox_width=10, viewbox_height=10, allow_images=True, max_embedded_image_bytes=1024),
        )
        self.assertIn("data:image/png;base64", cleaned)


if __name__ == "__main__":
    unittest.main()
