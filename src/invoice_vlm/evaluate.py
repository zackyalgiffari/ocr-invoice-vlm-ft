"""Score a model (base or base+LoRA) on the test split.

Writes ``predictions.jsonl`` and ``metrics.json`` into ``--out``.

    python -m invoice_vlm.evaluate --config configs/qwen35_2b_lora.yaml --out /content/run          # zero-shot
    python -m invoice_vlm.evaluate --config ... --adapter /content/run/model --out /content/run     # fine-tuned
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from invoice_vlm.config import load_config
from invoice_vlm.data import load_split
from invoice_vlm.metrics import score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", help="LoRA adapter dir or Hub repo id; omit for zero-shot")
    p.add_argument("--revision", help="adapter revision on the Hub")
    p.add_argument("--model-id", help="override config model_id")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, help="cap examples per source")
    args = p.parse_args()

    cfg = load_config(args.config)
    from invoice_vlm.modeling import generate, load_model

    model, processor = load_model(args.model_id or cfg["model_id"], args.adapter, args.revision)
    ds = load_split(args.split, tuple(cfg["sources"]), limit=args.limit)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bs = cfg["eval"]["batch_size"]
    rows = []
    started = time.time()
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for start in range(0, len(ds), bs):
            chunk = ds[start : start + bs]
            outputs = generate(
                model,
                processor,
                chunk["image"],
                chunk["doc_type"],
                cfg["max_pixels"],
                cfg.get("min_pixels", 0),
                cfg["eval"]["max_new_tokens"],
            )
            for source, target, output in zip(chunk["source"], chunk["target"], outputs, strict=True):
                rows.append((source, output, json.loads(target)))
                f.write(json.dumps({"source": source, "target": target, "output": output},
                                   ensure_ascii=False) + "\n")
            print(f"[eval] {min(start + bs, len(ds))}/{len(ds)}", flush=True)

    metrics = {
        "model_id": args.model_id or cfg["model_id"],
        "adapter": args.adapter,
        "split": args.split,
        "seconds": round(time.time() - started, 1),
        "scores": score(rows),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
