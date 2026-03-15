"""VLM prompt for image type classification."""

PROMPT_NAME = "vlm_classify"
PROMPT_VERSION = "v1.1.0"

PROMPT = """Classify the educational content type of this image.

OUTPUT FORMAT (JSON only, no markdown fences):
{"image_type": "code_screenshot" | "diagram" | "text_capture" | "equation" | "other", "confidence": 0.95}

DEFINITIONS — assign to the FIRST matching type:
- "code_screenshot": source code, terminal/shell output, IDE screenshot, config file snippet
- "diagram": technical diagram directly explaining a concept — flowchart, architecture diagram, neural network structure, data pipeline, sequence diagram, ER diagram, graph/chart with axes, system topology
- "text_capture": educational text content — lecture slide, handwritten notes, textbook page section with instructional content, whiteboard explanation
- "equation": standalone mathematical formulas, LaTeX expressions, derivations
- "other": NOT educational content — book/report cover, title page, publisher logo, author photo, decorative illustration, animal/nature photo, marketing image, blank/watermark page, table of contents page

CRITICAL: If the image is a book cover, front matter, publisher branding, or any decorative/non-instructional image, output "other" even if it contains text.
- Output ONLY valid JSON"""
