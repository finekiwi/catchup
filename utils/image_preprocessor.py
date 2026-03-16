"""Adaptive image preprocessing for VLM inputs."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from models.document import ImageType

LOGGER = logging.getLogger(__name__)

_OPENAI_TILE_BASE_TOKENS = 85
_OPENAI_TILE_PER_TILE_TOKENS = 170
_OPENAI_PATCH_SIZE = 32
_OPENAI_PATCH_BUDGET = 1536
_OPENAI_PATCH_MINI_MULTIPLIER = 1.62
_OPENAI_PATCH_NANO_MULTIPLIER = 2.46
_ANTHROPIC_MAX_LONG_EDGE = 1568
_ANTHROPIC_MAX_TOKENS = 1600
_ANTHROPIC_EFFECTIVE_MAX_AREA = 1_000_000
_GOOGLE_TOKEN_BUCKETS = {
    "low": 280,
    "medium": 560,
    "high": 1120,
    "ultra_high": 2240,
}


@dataclass(frozen=True)
class ResizeConfig:
    max_long_edge: int
    output_format: str
    jpeg_quality: int | None

RESIZE_POLICY: dict[ImageType, ResizeConfig] = {
    ImageType.CODE_SCREENSHOT: ResizeConfig(
        max_long_edge=1600, output_format="PNG", jpeg_quality=None
    ),
    ImageType.DIAGRAM: ResizeConfig(
        max_long_edge=1600, output_format="PNG", jpeg_quality=None
    ),
    ImageType.TEXT_CAPTURE: ResizeConfig(
        max_long_edge=1600, output_format="JPEG", jpeg_quality=85
    ),
    ImageType.CHART: ResizeConfig(
        max_long_edge=1024, output_format="PNG", jpeg_quality=None
    ),
    ImageType.EQUATION: ResizeConfig(
        max_long_edge=1600, output_format="PNG", jpeg_quality=None
    ),
    ImageType.OTHER: ResizeConfig(
        max_long_edge=512, output_format="JPEG", jpeg_quality=70
    ),
}


def preprocess_image(
    image_path: str,
    image_type: ImageType,
    override_config: ResizeConfig | None = None,
) -> tuple[str, dict]:
    """Resize and convert image format based on the image type policy."""
    config = override_config or _get_resize_config(image_type)
    source_path = Path(image_path)

    try:
        with Image.open(source_path) as image:
            image.verify()
        with Image.open(source_path) as image:
            image.load()
            original = image.copy()
            original_size = original.size
            original_format = _normalize_format(image.format or source_path.suffix)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        LOGGER.warning("Image preprocessing failed for %s: %s", image_path, exc)
        raise ValueError(f"Invalid or corrupt image: {image_path}") from exc

    processed = original
    processed_size = original_size
    resized = False

    target_size = _scaled_size(original_size, config.max_long_edge)
    if target_size != original_size:
        processed = processed.resize(target_size, Image.LANCZOS)
        processed_size = processed.size
        resized = True

    target_format = _normalize_format(config.output_format)
    format_changed = original_format != target_format

    if target_format == "JPEG":
        processed = _flatten_for_jpeg(processed)

    if not resized and not format_changed:
        processed_path = source_path
        processed_format = original_format
    else:
        processed_path = source_path.with_name(
            f"{source_path.stem}_preprocessed{_extension_for_format(target_format)}"
        )
        save_kwargs = {"format": target_format}
        if target_format == "JPEG":
            save_kwargs["quality"] = config.jpeg_quality if config.jpeg_quality else 85
            save_kwargs["optimize"] = True
            save_kwargs["subsampling"] = 0
        processed.save(processed_path, **save_kwargs)
        processed_format = target_format

    original_tokens_openai = _estimate_token_count_from_size(
        *original_size, provider="openai"
    )
    processed_tokens_openai = _estimate_token_count_from_size(
        *processed_size, provider="openai"
    )
    token_reduction_pct = 0.0
    if original_tokens_openai > 0:
        token_reduction_pct = round(
            ((original_tokens_openai - processed_tokens_openai) / original_tokens_openai)
            * 100,
            2,
        )

    metadata = {
        "original_size": original_size,
        "processed_size": processed_size,
        "original_tokens_openai": original_tokens_openai,
        "processed_tokens_openai": processed_tokens_openai,
        "token_reduction_pct": token_reduction_pct,
        "format": processed_format,
        "resized": resized,
    }

    LOGGER.info(
        (
            "Image preprocessed: path=%s image_type=%s original_size=%sx%s "
            "processed_size=%sx%s original_format=%s processed_format=%s "
            "resized=%s original_tokens_openai=%s processed_tokens_openai=%s "
            "token_reduction_pct=%.2f"
        ),
        image_path,
        str(image_type),
        original_size[0],
        original_size[1],
        processed_size[0],
        processed_size[1],
        original_format,
        processed_format,
        resized,
        original_tokens_openai,
        processed_tokens_openai,
        token_reduction_pct,
    )

    return str(processed_path), metadata


def estimate_token_count(image_path: str, provider: str = "openai") -> int:
    """Estimate VLM image input tokens from local image dimensions."""
    with Image.open(image_path) as image:
        width, height = image.size
    return _estimate_token_count_from_size(width, height, provider=provider)


def _estimate_token_count_from_size(width: int, height: int, provider: str) -> int:
    normalized = provider.strip().lower()

    if normalized in {"openai", "openai_tile"}:
        return _estimate_openai_tile_tokens(width, height)

    if normalized in {"openai_patch", "openai_patch_mini", "gpt-5-mini", "gpt-4.1-mini"}:
        return _estimate_openai_patch_tokens(
            width, height, multiplier=_OPENAI_PATCH_MINI_MULTIPLIER
        )

    if normalized in {"openai_patch_nano", "gpt-5-nano", "gpt-4.1-nano"}:
        return _estimate_openai_patch_tokens(
            width, height, multiplier=_OPENAI_PATCH_NANO_MULTIPLIER
        )

    if normalized == "anthropic":
        return _estimate_anthropic_tokens(width, height)

    if normalized in {"google", "google_high"}:
        return _GOOGLE_TOKEN_BUCKETS["high"]
    if normalized == "google_low":
        return _GOOGLE_TOKEN_BUCKETS["low"]
    if normalized == "google_medium":
        return _GOOGLE_TOKEN_BUCKETS["medium"]
    if normalized == "google_ultra_high":
        return _GOOGLE_TOKEN_BUCKETS["ultra_high"]

    raise ValueError(f"Unsupported provider: {provider}")


def _estimate_openai_tile_tokens(width: int, height: int) -> int:
    scaled_width, scaled_height = _fit_within(width, height, max_edge=2048)
    short_edge = min(scaled_width, scaled_height)
    if short_edge > 768:
        scale = 768 / short_edge
        scaled_width = max(1, math.floor(scaled_width * scale))
        scaled_height = max(1, math.floor(scaled_height * scale))

    tiles_w = math.ceil(scaled_width / 512)
    tiles_h = math.ceil(scaled_height / 512)
    return _OPENAI_TILE_BASE_TOKENS + (
        _OPENAI_TILE_PER_TILE_TOKENS * tiles_w * tiles_h
    )


def _estimate_openai_patch_tokens(width: int, height: int, *, multiplier: float) -> int:
    patch_w = math.ceil(width / _OPENAI_PATCH_SIZE)
    patch_h = math.ceil(height / _OPENAI_PATCH_SIZE)
    image_tokens = patch_w * patch_h

    if image_tokens > _OPENAI_PATCH_BUDGET:
        resize_factor = math.sqrt(
            (_OPENAI_PATCH_SIZE**2 * _OPENAI_PATCH_BUDGET) / (width * height)
        )
        width_patch_ratio = (width * resize_factor) / _OPENAI_PATCH_SIZE
        height_patch_ratio = (height * resize_factor) / _OPENAI_PATCH_SIZE
        resize_factor *= min(
            math.floor(width_patch_ratio) / width_patch_ratio,
            math.floor(height_patch_ratio) / height_patch_ratio,
        )

        resized_width = max(1, math.floor(width * resize_factor))
        resized_height = max(1, math.floor(height * resize_factor))
        image_tokens = math.ceil(resized_width / _OPENAI_PATCH_SIZE) * math.ceil(
            resized_height / _OPENAI_PATCH_SIZE
        )

    return math.ceil(image_tokens * multiplier)


def _estimate_anthropic_tokens(width: int, height: int) -> int:
    scaled_width, scaled_height = _fit_within(
        width, height, max_edge=_ANTHROPIC_MAX_LONG_EDGE
    )
    tokens = (scaled_width * scaled_height) / 750
    if tokens > _ANTHROPIC_MAX_TOKENS:
        scale = math.sqrt(
            _ANTHROPIC_EFFECTIVE_MAX_AREA / (scaled_width * scaled_height)
        )
        scaled_width = max(1, math.floor(scaled_width * scale))
        scaled_height = max(1, math.floor(scaled_height * scale))
        scaled_width = max(_OPENAI_PATCH_SIZE, scaled_width - (scaled_width % _OPENAI_PATCH_SIZE))
        scaled_height = max(
            _OPENAI_PATCH_SIZE, scaled_height - (scaled_height % _OPENAI_PATCH_SIZE)
        )
        tokens = (scaled_width * scaled_height) / 750
    return int(tokens)


def _fit_within(width: int, height: int, *, max_edge: int) -> tuple[int, int]:
    if max(width, height) <= max_edge:
        return width, height
    scale = max_edge / max(width, height)
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def _scaled_size(size: tuple[int, int], max_long_edge: int) -> tuple[int, int]:
    width, height = size
    if max(width, height) <= max_long_edge:
        return size
    scale = max_long_edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _flatten_for_jpeg(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def _get_resize_config(image_type: ImageType) -> ResizeConfig:
    key = _normalize_image_type(image_type)
    return RESIZE_POLICY.get(key, RESIZE_POLICY[ImageType.OTHER])


def _normalize_image_type(image_type: ImageType | str) -> ImageType:
    if isinstance(image_type, ImageType):
        return image_type

    raw = str(image_type).strip().lower()
    try:
        return ImageType(raw)
    except ValueError:
        return ImageType.OTHER


def _normalize_format(value: str) -> str:
    normalized = value.strip().lstrip(".").upper()
    if normalized in {"JPG", "JPEG"}:
        return "JPEG"
    if normalized == "PNG":
        return "PNG"
    raise ValueError(f"Unsupported output format: {value}")


def _extension_for_format(image_format: str) -> str:
    if image_format == "JPEG":
        return ".jpg"
    if image_format == "PNG":
        return ".png"
    raise ValueError(f"Unsupported image format: {image_format}")


__all__ = [
    "RESIZE_POLICY",
    "ResizeConfig",
    "estimate_token_count",
    "preprocess_image",
]
