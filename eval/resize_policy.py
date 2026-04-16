"""Helpers for the CU-16 image resize policy experiment."""

from __future__ import annotations

import csv
import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from models.document import ImageType
from parsers.image_parser import (
    _PROMPT_GETTER_BY_IMAGE_TYPE,
    classify_image,
    parse_vlm_output,
)
from utils.image_preprocessor import ResizeConfig, estimate_token_count, preprocess_image
from utils.models import compute_cost
from vlm.client import VLMResult, call_vlm

LOGGER = logging.getLogger(__name__)

DATASET_BUCKETS: tuple[ImageType, ...] = (
    ImageType.CODE_SCREENSHOT,
    ImageType.DIAGRAM,
    ImageType.TEXT_CAPTURE,
    ImageType.CHART,
    ImageType.OTHER,
)
AUTO_PDF_BUCKETS = {ImageType.DIAGRAM, ImageType.CHART, ImageType.OTHER}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
UNIFORM_POLICY: dict[ImageType, ResizeConfig] = {
    ImageType.CODE_SCREENSHOT: ResizeConfig(1600, "PNG", None),
    ImageType.DIAGRAM: ResizeConfig(1600, "PNG", None),
    ImageType.TEXT_CAPTURE: ResizeConfig(1600, "JPEG", 85),
    ImageType.CHART: ResizeConfig(1600, "PNG", None),
    ImageType.OTHER: ResizeConfig(1600, "JPEG", 70),
}
PROVIDER_COST_MODELS = {
    "openai_tile": "gpt-4o-mini",
    "openai_patch": "gpt-4.1-nano",
    "anthropic": "claude-haiku-4-5-20251001",
    "google": "gemini-3-flash-preview",
}


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    image_path: str
    image_type: str
    source_kind: str
    source_doc: str
    filename: str


@dataclass(frozen=True)
class ConditionArtifact:
    condition: str
    image_path: str
    preprocess: dict[str, Any]


def bucket_dir_name(image_type: ImageType) -> str:
    """Return the dataset folder name for a ground-truth bucket."""
    return image_type.value


def image_type_from_bucket(name: str) -> ImageType:
    """Parse a dataset folder name into an ImageType."""
    normalized = name.strip().lower()
    for image_type in DATASET_BUCKETS:
        if image_type.value == normalized:
            return image_type
    raise ValueError(f"Unsupported bucket name: {name}")


def discover_manifest_rows(dataset_dir: Path) -> list[ManifestRow]:
    """Scan a bucketed dataset directory and return manifest rows."""
    rows: list[ManifestRow] = []
    for image_type in DATASET_BUCKETS:
        bucket_dir = dataset_dir / bucket_dir_name(image_type)
        if not bucket_dir.exists():
            continue
        for image_path in sorted(bucket_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
                continue
            source_kind, source_doc = infer_image_source(image_path.name)
            rows.append(
                ManifestRow(
                    image_id=image_path.stem,
                    image_path=str(image_path),
                    image_type=image_type.value,
                    source_kind=source_kind,
                    source_doc=source_doc,
                    filename=image_path.name,
                )
            )
    return rows


def write_manifest_csv(rows: Iterable[ManifestRow], output_path: Path) -> Path:
    """Write manifest rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "image_path", "image_type", "source_kind", "source_doc", "filename"],
        )
        writer.writeheader()
        for row in row_list:
            writer.writerow(asdict(row))
    return output_path


def infer_image_source(filename: str) -> tuple[str, str]:
    """Infer source kind and source document from the dataset filename."""
    if filename.startswith("pdf_auto__"):
        parts = filename.split("__")
        if len(parts) >= 3:
            return "pdf_auto", parts[1]
    return "manual_capture", ""


def select_pdf_bucket(
    image_path: str | Path,
    *,
    label_hint: str | None,
    classify_model: str,
) -> ImageType | None:
    """Decide which dataset bucket should receive a PDF-derived figure."""
    normalized_label = (label_hint or "").strip().lower()
    if normalized_label == "chart":
        return ImageType.CHART

    try:
        detected = classify_image(str(image_path), classify_model)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "PDF figure auto-classification failed for %s: %s; falling back to DIAGRAM",
            image_path,
            exc,
        )
        return ImageType.DIAGRAM

    if detected in AUTO_PDF_BUCKETS:
        return detected
    return None


def collect_pdf_figures_to_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    classify_model: str,
) -> list[Path]:
    """Extract PDF figures into typed dataset buckets under output_dir."""
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import PictureItem

    copied_paths: list[Path] = []
    for pdf_path in sorted(source_dir.glob("*.pdf")):
        LOGGER.info("Preparing resize golden figures from %s", pdf_path.name)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = 1.0
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(pdf_path), raises_on_error=False)
        if ConversionStatus is not None and result.status == ConversionStatus.FAILURE:
            LOGGER.warning("Docling conversion failed for %s; skipping", pdf_path.name)
            continue
        docling_doc = result.document
        picture_items = [
            item
            for item, _level in docling_doc.iterate_items()
            if isinstance(item, PictureItem)
        ]

        for order, picture_item in enumerate(picture_items):
            pil_image = picture_item.get_image(docling_doc)
            if pil_image is None:
                continue
            label_value = getattr(getattr(picture_item, "label", None), "value", "")
            page = _extract_page(picture_item)
            temp_image = output_dir / "_tmp" / f"{pdf_path.stem}__p{page or 'na'}__o{order}.png"
            temp_image.parent.mkdir(parents=True, exist_ok=True)
            pil_image.save(temp_image, format="PNG")
            bucket = select_pdf_bucket(
                temp_image,
                label_hint=str(label_value),
                classify_model=classify_model,
            )
            if bucket is None:
                temp_image.unlink(missing_ok=True)
                continue
            destination = (
                output_dir
                / bucket_dir_name(bucket)
                / f"pdf_auto__{pdf_path.stem}__p{page or 'na'}__o{order}.png"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_image), str(destination))
            copied_paths.append(destination)
    return copied_paths


def build_condition_artifacts(
    image_path: str,
    image_type: ImageType,
) -> list[ConditionArtifact]:
    """Build the 3 experiment conditions for one image."""
    with Image.open(image_path) as image:
        original_size = image.size
    original_tokens = estimate_provider_metrics(image_path, output_tokens=0)
    with_before = {
        "original_size": original_size,
        "processed_size": original_size,
        "format": Path(image_path).suffix.replace(".", "").upper() or "PNG",
        "resized": False,
        "original_tokens_openai": original_tokens["openai_tile"]["input_tokens"],
        "processed_tokens_openai": original_tokens["openai_tile"]["input_tokens"],
        "token_reduction_pct": 0.0,
    }

    uniform_path, uniform_meta = preprocess_image(
        image_path,
        image_type,
        override_config=UNIFORM_POLICY[image_type],
    )
    adaptive_path, adaptive_meta = preprocess_image(image_path, image_type)

    return [
        ConditionArtifact("before", image_path, with_before),
        ConditionArtifact("after_uniform", uniform_path, uniform_meta),
        ConditionArtifact("after_adaptive", adaptive_path, adaptive_meta),
    ]


def run_single_analysis(
    image_path: str,
    image_type: ImageType,
    *,
    model: str,
    language: str,
    stage: str,
) -> VLMResult:
    """Run one analysis call with the prompt selected by ground-truth image type."""
    prompt = _PROMPT_GETTER_BY_IMAGE_TYPE[image_type](language)
    return call_vlm(model, image_path, prompt, stage=stage)


def evaluate_quality(raw_response: str, image_type: ImageType) -> dict[str, Any]:
    """Compute structural quality metrics from a raw VLM response."""
    stripped = raw_response.strip()
    metrics: dict[str, Any] = {
        "json_parse_success": False,
        "schema_validation_success": False,
        "required_fields_ok": False,
        "empty_output": not bool(stripped),
        "has_truncation": False,
        "errors_present": False,
        "content_length": len(stripped),
    }
    try:
        payload = parse_vlm_output(raw_response, image_type)
    except Exception:
        return metrics

    metrics["json_parse_success"] = True
    metrics["schema_validation_success"] = True
    metrics["has_truncation"] = bool(payload.has_truncation)
    metrics["errors_present"] = bool(payload.errors)

    if image_type == ImageType.CODE_SCREENSHOT:
        metrics["required_fields_ok"] = bool(payload.language and (payload.code or payload.code_markdown))
    elif image_type in {ImageType.DIAGRAM, ImageType.CHART}:
        metrics["required_fields_ok"] = bool(
            payload.description.strip() and (payload.components or payload.relationships)
        )
    elif image_type == ImageType.TEXT_CAPTURE:
        metrics["required_fields_ok"] = bool(payload.content.strip()) and len(payload.content.strip()) >= 20
    else:
        content = getattr(payload, "content", "") or getattr(payload, "description", "")
        metrics["required_fields_ok"] = bool(str(content).strip())

    return metrics


def estimate_provider_metrics(image_path: str, *, output_tokens: int) -> dict[str, dict[str, float | int]]:
    """Estimate provider token and cost metrics for an image path."""
    metrics: dict[str, dict[str, float | int]] = {}
    for provider, model_name in PROVIDER_COST_MODELS.items():
        input_tokens = estimate_token_count(image_path, provider=provider)
        metrics[provider] = {
            "input_tokens": input_tokens,
            "cost_usd": round(compute_cost(model_name, input_tokens, output_tokens), 8),
        }
    return metrics


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Aggregate experiment rows by condition and by type x condition."""
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_type_condition[(row["image_type"], row["condition"])].append(row)

    return {
        "by_condition": [
            _aggregate_bucket({"condition": condition}, bucket_rows)
            for condition, bucket_rows in sorted(by_condition.items())
        ],
        "by_type_condition": [
            _aggregate_bucket(
                {"image_type": image_type, "condition": condition},
                bucket_rows,
            )
            for (image_type, condition), bucket_rows in sorted(by_type_condition.items())
        ],
    }


def _aggregate_bucket(prefix: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        **prefix,
        "count": count,
        "avg_input_tokens": round(_mean(rows, "input_tokens"), 2),
        "avg_output_tokens": round(_mean(rows, "output_tokens"), 2),
        "avg_cost_usd": round(_mean(rows, "cost_usd"), 8),
        "avg_latency_ms": round(_mean(rows, "latency_ms"), 2),
        "json_parse_success_rate": round(_rate(rows, "json_parse_success"), 4),
        "required_fields_ok_rate": round(_rate(rows, "required_fields_ok"), 4),
        "empty_output_rate": round(_rate(rows, "empty_output"), 4),
        "truncation_rate": round(_rate(rows, "has_truncation"), 4),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row.get(key, 0) or 0) for row in rows) / len(rows)


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key)) / len(rows)


def load_manifest_csv(manifest_path: Path) -> list[ManifestRow]:
    """Load the dataset manifest."""
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [ManifestRow(**row) for row in reader]


def run_resize_policy_eval(
    manifest_path: Path,
    *,
    model: str,
    repeats: int,
    language: str,
    output_dir: Path,
    seed: int = 16,
) -> dict[str, Any]:
    """Run the resize policy comparison experiment and save report files."""
    manifest = load_manifest_csv(manifest_path)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for item in manifest:
        image_type = ImageType(item.image_type)
        artifacts = build_condition_artifacts(item.image_path, image_type)

        for repeat_index in range(repeats):
            run_order = list(artifacts)
            rng.shuffle(run_order)
            for artifact in run_order:
                result = run_single_analysis(
                    artifact.image_path,
                    image_type,
                    model=model,
                    language=language,
                    stage=f"resize_eval_{artifact.condition}",
                )
                quality = evaluate_quality(result.content, image_type)
                provider_metrics = estimate_provider_metrics(
                    artifact.image_path,
                    output_tokens=result.output_tokens,
                )
                rows.append(
                    {
                        "image_id": item.image_id,
                        "image_type": item.image_type,
                        "source_kind": item.source_kind,
                        "source_doc": item.source_doc,
                        "condition": artifact.condition,
                        "repeat": repeat_index + 1,
                        "processed_image_path": artifact.image_path,
                        "success": result.success,
                        "error": result.error or "",
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "cost_usd": result.cost_usd,
                        "latency_ms": result.latency_ms,
                        **artifact.preprocess,
                        **quality,
                        "est_input_tokens_openai_tile": provider_metrics["openai_tile"]["input_tokens"],
                        "est_cost_usd_openai_tile": provider_metrics["openai_tile"]["cost_usd"],
                        "est_input_tokens_openai_patch": provider_metrics["openai_patch"]["input_tokens"],
                        "est_cost_usd_openai_patch": provider_metrics["openai_patch"]["cost_usd"],
                        "est_input_tokens_anthropic": provider_metrics["anthropic"]["input_tokens"],
                        "est_cost_usd_anthropic": provider_metrics["anthropic"]["cost_usd"],
                        "est_input_tokens_google": provider_metrics["google"]["input_tokens"],
                        "est_cost_usd_google": provider_metrics["google"]["cost_usd"],
                        "raw_response": result.content,
                    }
                )

    aggregates = aggregate_rows(rows)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"resize_policy_eval_{timestamp}.json"
    csv_path = output_dir / f"resize_policy_eval_{timestamp}.csv"
    json_path.write_text(
        json.dumps(
            {
                "model": model,
                "manifest_path": str(manifest_path),
                "rows": rows,
                "aggregates": aggregates,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_eval_csv(rows, csv_path)
    return {"json_path": str(json_path), "csv_path": str(csv_path), "aggregates": aggregates}


def _write_eval_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = [key for key in rows[0].keys() if key != "raw_response"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            filtered = {key: value for key, value in row.items() if key in fieldnames}
            writer.writerow(filtered)


def _extract_page(item: object) -> int | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    first = prov[0] if isinstance(prov, list) else prov
    page_no = getattr(first, "page_no", None)
    try:
        return int(page_no) if page_no is not None else None
    except (TypeError, ValueError):
        return None
