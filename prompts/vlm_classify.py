"""VLM prompt for image type classification."""

PROMPT_NAME = "vlm_classify"
PROMPT_VERSION = "v1.2.0"

PROMPT = """Classify the educational content type of this image.

OUTPUT FORMAT (JSON only, no markdown fences):
{"image_type": "code_screenshot" | "diagram" | "text_capture" | "equation" | "other", "confidence": 0.95}

DEFINITIONS — assign to the FIRST matching type:
- "code_screenshot": source code, terminal/shell output, IDE screenshot, config file snippet
- "diagram": technical diagram explaining a concept — flowchart, architecture diagram, neural network structure, data pipeline, sequence diagram, ER diagram, graph/chart with labeled axes, system topology
- "text_capture": educational text WITH instructional content — lecture slide explaining a topic, handwritten notes, textbook page section, whiteboard explanation. The text must teach something technical, not just display a title.
- "equation": standalone mathematical formulas, LaTeX expressions, derivations
- "other": anything that does NOT convey educational/instructional content:
    - publisher/company logos (e.g. O'Reilly, Hanbit, Packt)
    - book title images (large title text, author name, series name — even if it says a technical term like "Deep Learning")
    - book/report covers or back covers
    - chapter divider pages with only a chapter number and title
    - copyright pages, dedication pages, table of contents
    - author portraits or photos
    - decorative illustrations, animals, nature photos
    - blank or near-blank pages

ASK YOURSELF: "Would a student learn something technical from this image alone?" If NO → "other".
Output ONLY valid JSON."""
