"""Load base model (+ optional LoRA adapter) and run batched greedy generation."""

from __future__ import annotations

from typing import Any

from invoice_vlm.config import cap_pixels
from invoice_vlm.data import build_messages


def load_model(model_id: str, adapter: str | None = None, revision: str | None = None):
    """Return (model, processor). ``adapter`` is a local dir or a Hub repo id."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, revision=revision)
        model = model.merge_and_unload()  # faster generation; weights stay in memory only
    model.eval()
    processor.tokenizer.padding_side = "left"  # required for batched generation
    return model, processor


def generate(
    model: Any,
    processor: Any,
    images: list[Any],
    doc_types: list[str],
    max_pixels: int,
    min_pixels: int = 0,
    max_new_tokens: int = 1536,
) -> list[str]:
    """Greedy-decode one answer per image. Extraction wants determinism, not sampling."""
    import torch

    images = [cap_pixels(im, max_pixels, min_pixels) for im in images]
    prompts = [
        processor.apply_chat_template(
            build_messages(im, dt), tokenize=False, add_generation_prompt=True
        )
        for im, dt in zip(images, doc_types, strict=True)
    ]
    batch = processor(
        text=prompts, images=[[im] for im in images], padding=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[:, batch["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)
