from pathlib import Path

from PIL import Image

from invoice_vlm.config import cap_pixels, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_repo_config_loads():
    cfg = load_config(ROOT / "configs" / "qwen35_2b_lora.yaml")
    assert cfg["model_id"] == "Qwen/Qwen3.5-2B"
    assert cfg["hub"]["private"] is True


def test_cap_pixels_downscales_keeping_aspect():
    out = cap_pixels(Image.new("L", (2000, 3000)), max_pixels=1_000_000)
    assert out.mode == "RGB"
    assert out.width * out.height <= 1_000_000
    assert abs(out.width / out.height - 2000 / 3000) < 0.01


def test_cap_pixels_upscales_tiny_and_leaves_normal_alone():
    assert cap_pixels(Image.new("RGB", (100, 100)), 1_000_000, 65536).size[0] >= 255
    assert cap_pixels(Image.new("RGB", (800, 600)), 1_000_000, 65536).size == (800, 600)
