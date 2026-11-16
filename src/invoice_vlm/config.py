"""YAML config loading shared by train/evaluate/infer."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("model_id", "sources", "max_pixels", "lora", "train", "hub", "eval"):
        if key not in cfg:
            raise KeyError(f"{path}: missing required key {key!r}")
    return cfg


def cap_pixels(image: Any, max_pixels: int, min_pixels: int = 0) -> Any:
    """Downscale a PIL image (keeping aspect) so width*height <= max_pixels.

    Images below ``min_pixels`` are upscaled to it; the processor would do the same, but
    doing it here keeps the token count predictable and independent of library defaults.
    """
    image = image.convert("RGB")
    w, h = image.size
    area = w * h
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
    elif min_pixels and area < min_pixels:
        scale = (min_pixels / area) ** 0.5
    else:
        return image
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(size, resample=3)  # 3 = PIL.Image.BICUBIC
