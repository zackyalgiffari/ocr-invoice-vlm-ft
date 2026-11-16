"""Extract fields from one invoice/receipt image and print JSON.

    python -m invoice_vlm.infer --config configs/qwen35_2b_lora.yaml \
        --adapter zackyalgiffari/qwen3.5-2b-receipt-invoice-extract-lora --type receipt struk.jpg
"""

from __future__ import annotations

import argparse
import json
import sys

from invoice_vlm.config import load_config
from invoice_vlm.metrics import extract_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("image")
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", help="LoRA adapter dir or Hub repo id; omit for the base model")
    p.add_argument("--type", choices=["receipt", "invoice"], default="invoice")
    args = p.parse_args()

    cfg = load_config(args.config)
    from PIL import Image

    from invoice_vlm.modeling import generate, load_model

    model, processor = load_model(cfg["model_id"], args.adapter)
    [output] = generate(
        model, processor, [Image.open(args.image)], [args.type],
        cfg["max_pixels"], cfg.get("min_pixels", 0), cfg["eval"]["max_new_tokens"],
    )
    parsed = extract_json(output)
    if parsed is None:
        print(output)
        sys.exit("model output is not valid JSON")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
