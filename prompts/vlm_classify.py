"""VLM prompt for image type classification."""

PROMPT_NAME = "vlm_classify"
PROMPT_VERSION = "v1.0.0"

PROMPT = """Classify the type of content in this image.

OUTPUT FORMAT (JSON only, no markdown fences):
{"image_type": "code_screenshot" | "diagram" | "text_capture" | "equation" | "other", "confidence": 0.95}

RULES:
- "code_screenshot": programming code, terminal output, IDE screenshot
- "diagram": flowchart, architecture diagram, sequence diagram, mindmap, ER diagram
- "text_capture": lecture slide, handwritten notes, textbook page, article
- "equation": mathematical formulas or expressions
- "other": none of the above
- Output ONLY valid JSON"""
