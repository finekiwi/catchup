"""VLM prompt for image type classification."""

PROMPT_NAME = "vlm_classify"
PROMPT_VERSION = "v1.4.0"

PROMPT = """Classify the educational content type of this image.

OUTPUT FORMAT (JSON only, no markdown fences):
{"image_type": "code_screenshot" | "diagram" | "chart" | "text_capture" | "equation" | "other", "confidence": 0.95}

DEFINITIONS — assign to the FIRST matching type:
- "code_screenshot": source code, terminal/shell output, IDE screenshot, config file snippet
- "diagram": technical diagram explaining a concept — flowchart, architecture diagram, neural network structure, data pipeline, sequence diagram, ER diagram, system topology. Focus is structure, entities, or directional flow rather than quantitative values.
- "chart": quantitative visualization — bar chart, line chart, scatter plot, histogram, pie chart, heatmap, or graph with labeled axes, legends, numeric scales, or plotted series. Focus is measured values, comparisons, or trends.
- "text_capture": educational text WITH instructional content — lecture slide explaining a topic, handwritten notes, textbook page section, whiteboard explanation. The text must teach something technical, not just display a title.
- "equation": standalone mathematical formulas, LaTeX expressions, derivations
- "other": anything that does NOT convey educational/instructional content:
    - publisher/company logos (e.g. O'Reilly, Hanbit, Packt)
    - book title images (large title text, author name, series name — even if it says a technical term like "Deep Learning")
    - book/report covers or back covers
    - chapter divider pages with only a chapter number and title
    - mascot characters, cartoon characters, decorative illustrations
    - chapter divider art, title-page artwork, paper-craft style decorations
    - decorative animals or ornamental characters, even if they look like a diagram
    - copyright pages, dedication pages, table of contents
    - author portraits or photos
    - decorative illustrations, animals, nature photos
    - blank or near-blank pages

ASK YOURSELF: "Would a student learn something technical from this image alone?" If NO → "other".
Output ONLY valid JSON."""
