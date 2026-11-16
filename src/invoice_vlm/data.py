"""Load public invoice/receipt datasets and turn them into chat-style SFT examples.

Both sources store labels in the Donut format: a JSON string ``{"gt_parse": {...}}``
in a ``ground_truth`` column next to an ``image`` column. The model is trained to emit
the inner ``gt_parse`` object as compact JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # heavy import only needed at runtime inside load_split
    from datasets import Dataset


@dataclass(frozen=True)
class Source:
    key: str
    hf_id: str
    doc_type: str  # selects the instruction, see INSTRUCTIONS


SOURCES: dict[str, Source] = {
    "cord": Source("cord", "naver-clova-ix/cord-v2", "receipt"),
    "invoices": Source("invoices", "katanaml-org/invoices-donut-data-v1", "invoice"),
}

SPLITS = ("train", "validation", "test")

INSTRUCTIONS: dict[str, str] = {
    "receipt": "Extract all fields from this receipt image. Respond with only a JSON object.",
    "invoice": "Extract all fields from this invoice image. Respond with only a JSON object.",
}


def parse_ground_truth(raw: str) -> Any:
    """Return the label object from a Donut-style ``ground_truth`` string."""
    gt = json.loads(raw)
    if "gt_parse" in gt:
        return gt["gt_parse"]
    # Some Donut datasets store several acceptable answers; the first is canonical.
    if "gt_parses" in gt and gt["gt_parses"]:
        return gt["gt_parses"][0]
    raise ValueError(f"ground_truth has neither gt_parse nor gt_parses: keys={list(gt)}")


def to_target(label: Any) -> str:
    """Canonical training target: compact JSON, non-ASCII kept as-is (fewer tokens)."""
    return json.dumps(label, ensure_ascii=False, separators=(",", ":"))


def build_messages(image: Any, doc_type: str, target: str | None = None) -> list[dict]:
    """Chat messages for one example. ``target`` is omitted for inference prompts."""
    messages: list[dict] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": INSTRUCTIONS[doc_type]},
            ],
        }
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def load_split(
    split: str,
    sources: tuple[str, ...] = tuple(SOURCES),
    limit: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Load one split from every source as columns: image, source, doc_type, target.

    ``limit`` caps examples *per source* (used for smoke tests). The train split is
    shuffled so the two document types are interleaved.
    """
    from datasets import concatenate_datasets, load_dataset

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    parts = []
    for key in sources:
        src = SOURCES[key]
        ds = load_dataset(src.hf_id, split=split)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        ds = ds.map(
            lambda row, src=src: {
                "source": src.key,
                "doc_type": src.doc_type,
                "target": to_target(parse_ground_truth(row["ground_truth"])),
            },
            remove_columns=[c for c in ds.column_names if c != "image"],
        )
        parts.append(ds)

    merged = concatenate_datasets(parts)
    return merged.shuffle(seed=seed) if split == "train" else merged
