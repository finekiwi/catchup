"""Unit tests for adaptive image preprocessing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from models.document import ImageType
from utils.image_preprocessor import (
    ResizeConfig,
    estimate_token_count,
    preprocess_image,
)


def _make_image(
    path: Path,
    *,
    size: tuple[int, int],
    mode: str = "RGB",
    image_format: str = "PNG",
    color: tuple[int, ...] = (32, 64, 96),
) -> None:
    Image.new(mode, size, color=color).save(path, format=image_format)


def _save_kwargs_spy(record: dict[str, object]):
    original_save = Image.Image.save

    def _wrapped_save(self, fp, format=None, **params):
        record["format"] = format
        record.update(params)
        return original_save(self, fp, format=format, **params)

    return patch.object(Image.Image, "save", new=_wrapped_save)


def test_code_screenshot_resize(tmp_path: Path) -> None:
    image_path = tmp_path / "code.png"
    _make_image(image_path, size=(2000, 1500), image_format="PNG")

    processed_path, metadata = preprocess_image(str(image_path), ImageType.CODE_SCREENSHOT)

    assert processed_path.endswith("_preprocessed.png")
    assert metadata["processed_size"] == (1600, 1200)
    assert metadata["format"] == "PNG"
    assert metadata["resized"] is True


def test_diagram_resize(tmp_path: Path) -> None:
    image_path = tmp_path / "diagram.png"
    _make_image(image_path, size=(3000, 2000), image_format="PNG")

    processed_path, metadata = preprocess_image(str(image_path), ImageType.DIAGRAM)

    assert processed_path.endswith("_preprocessed.png")
    assert metadata["processed_size"] == (1600, 1067)
    assert metadata["format"] == "PNG"


def test_text_capture_format(tmp_path: Path) -> None:
    image_path = tmp_path / "text.png"
    _make_image(image_path, size=(2000, 1500), image_format="PNG")
    save_kwargs: dict[str, object] = {}

    with _save_kwargs_spy(save_kwargs):
        processed_path, metadata = preprocess_image(str(image_path), ImageType.TEXT_CAPTURE)

    assert processed_path.endswith("_preprocessed.jpg")
    assert metadata["processed_size"] == (1600, 1200)
    assert metadata["format"] == "JPEG"
    assert save_kwargs["quality"] == 85


def test_chart_resize(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    _make_image(image_path, size=(2000, 1500), image_format="PNG")

    processed_path, metadata = preprocess_image(str(image_path), ImageType.CHART)

    assert processed_path.endswith("_preprocessed.png")
    assert metadata["processed_size"] == (1024, 768)
    assert metadata["format"] == "PNG"


def test_other_aggressive(tmp_path: Path) -> None:
    image_path = tmp_path / "logo.png"
    _make_image(image_path, size=(2000, 1500), image_format="PNG")
    save_kwargs: dict[str, object] = {}

    with _save_kwargs_spy(save_kwargs):
        processed_path, metadata = preprocess_image(str(image_path), ImageType.OTHER)

    assert processed_path.endswith("_preprocessed.jpg")
    assert metadata["processed_size"] == (512, 384)
    assert metadata["format"] == "JPEG"
    assert save_kwargs["quality"] == 70


def test_small_image_no_resize(tmp_path: Path) -> None:
    image_path = tmp_path / "small.jpg"
    _make_image(image_path, size=(800, 600), image_format="JPEG")

    processed_path, metadata = preprocess_image(str(image_path), ImageType.TEXT_CAPTURE)

    assert processed_path == str(image_path)
    assert metadata["original_size"] == (800, 600)
    assert metadata["processed_size"] == (800, 600)
    assert metadata["resized"] is False


def test_corrupt_image_handling(tmp_path: Path) -> None:
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Invalid or corrupt image"):
        preprocess_image(str(image_path), ImageType.OTHER)


def test_override_config(tmp_path: Path) -> None:
    image_path = tmp_path / "override.png"
    _make_image(image_path, size=(2000, 1500), image_format="PNG")

    processed_path, metadata = preprocess_image(
        str(image_path),
        ImageType.DIAGRAM,
        override_config=ResizeConfig(
            max_long_edge=1000,
            output_format="JPEG",
            jpeg_quality=60,
        ),
    )

    assert processed_path.endswith("_preprocessed.jpg")
    assert metadata["processed_size"] == (1000, 750)
    assert metadata["format"] == "JPEG"


def test_metadata_token_reduction(tmp_path: Path) -> None:
    image_path = tmp_path / "decorative.png"
    _make_image(image_path, size=(2000, 1500), image_format="PNG")

    _, metadata = preprocess_image(str(image_path), ImageType.OTHER)

    assert metadata["original_tokens_openai"] == 765
    assert metadata["processed_tokens_openai"] == 255
    assert metadata["token_reduction_pct"] == pytest.approx(66.67, abs=0.01)


def test_estimate_tokens_openai_tile(tmp_path: Path) -> None:
    image_path = tmp_path / "wide.png"
    _make_image(image_path, size=(1920, 1080), image_format="PNG")

    assert estimate_token_count(str(image_path), provider="openai_tile") == 1105


def test_estimate_tokens_openai_patch(tmp_path: Path) -> None:
    image_path = tmp_path / "patch.png"
    _make_image(image_path, size=(1920, 1080), image_format="PNG")

    assert estimate_token_count(str(image_path), provider="openai_patch") == 2443


def test_estimate_tokens_anthropic(tmp_path: Path) -> None:
    image_path = tmp_path / "anthropic.png"
    _make_image(image_path, size=(4000, 3000), image_format="PNG")

    assert estimate_token_count(str(image_path), provider="anthropic") == 1327


def test_estimate_tokens_google(tmp_path: Path) -> None:
    image_path = tmp_path / "gemini.png"
    _make_image(image_path, size=(4000, 3000), image_format="PNG")

    assert estimate_token_count(str(image_path), provider="google") == 1120


def test_aspect_ratio_preserved(tmp_path: Path) -> None:
    image_path = tmp_path / "ratio.png"
    _make_image(image_path, size=(2500, 1700), image_format="PNG")

    processed_path, metadata = preprocess_image(str(image_path), ImageType.DIAGRAM)

    with Image.open(processed_path) as processed:
        processed_ratio = processed.width / processed.height

    original_ratio = metadata["original_size"][0] / metadata["original_size"][1]
    assert processed_ratio == pytest.approx(original_ratio, rel=0.002)


def test_rgba_to_rgb_jpeg(tmp_path: Path) -> None:
    image_path = tmp_path / "rgba.png"
    _make_image(
        image_path,
        size=(400, 300),
        mode="RGBA",
        image_format="PNG",
        color=(255, 0, 0, 128),
    )

    processed_path, metadata = preprocess_image(str(image_path), ImageType.TEXT_CAPTURE)

    with Image.open(processed_path) as processed:
        pixel = processed.getpixel((0, 0))
        assert processed.mode == "RGB"

    assert metadata["format"] == "JPEG"
    assert pixel[0] >= 240
    assert 110 <= pixel[1] <= 145
    assert 110 <= pixel[2] <= 145
