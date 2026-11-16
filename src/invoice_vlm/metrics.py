"""Scoring for JSON extraction: validity, field-level P/R/F1, document exact match.

Fields are compared as a multiset of ``(path, value)`` leaves. List indices are dropped
from the path (``menu[].nm``), so line items are matched by content regardless of order,
and a duplicated item counts twice.

``value_f1`` repeats the comparison on values alone, ignoring keys. A zero-shot model
reads the document correctly but invents its own key names, which scores 0 on field F1;
the gap between the two numbers is "can read" vs "follows the schema".

Re-score saved predictions without a GPU:  python -m invoice_vlm.metrics predictions.jsonl
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text: str) -> Any | None:
    """Parse the model's answer; tolerate thinking blocks, code fences and chatter."""
    text = _THINK.sub("", text).strip()
    text = _FENCE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value).split()).lower()


def flatten(obj: Any, prefix: str = "") -> Counter[tuple[str, str]]:
    """Multiset of (path, normalized leaf value). Empty strings are ignored."""
    out: Counter[tuple[str, str]] = Counter()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for v in obj:
            out += flatten(v, f"{prefix}[]")
    elif obj is not None:
        value = _norm(obj)
        if value:
            out[(prefix, value)] += 1
    return out


@dataclass
class Scores:
    n: int = 0
    valid_json: int = 0
    exact_match: int = 0
    tp: int = 0
    n_pred: int = 0
    n_gold: int = 0
    value_tp: int = 0
    by_source: dict[str, Scores] = field(default_factory=dict)

    def add(self, pred: Any | None, gold: Any) -> None:
        self.n += 1
        g = flatten(gold)
        self.n_gold += sum(g.values())
        if pred is None:
            return
        self.valid_json += 1
        p = flatten(pred)
        self.n_pred += sum(p.values())
        self.tp += sum((p & g).values())
        self.value_tp += sum((_values(p) & _values(g)).values())
        self.exact_match += int(p == g)

    def summary(self) -> dict[str, float]:
        precision, recall, f1 = _prf(self.tp, self.n_pred, self.n_gold)
        _, _, value_f1 = _prf(self.value_tp, self.n_pred, self.n_gold)
        return {
            "n": self.n,
            "json_valid": self.valid_json / self.n if self.n else 0.0,
            "exact_match": self.exact_match / self.n if self.n else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "value_f1": value_f1,
        }


def _values(flat: Counter[tuple[str, str]]) -> Counter[str]:
    out: Counter[str] = Counter()
    for (_, value), count in flat.items():
        out[value] += count
    return out


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score(rows: Iterable[tuple[str, str, Any]]) -> dict[str, dict[str, float]]:
    """``rows`` = (source, raw model output, gold label). Returns overall + per-source."""
    overall = Scores()
    for source, output, gold in rows:
        pred = extract_json(output)
        overall.add(pred, gold)
        overall.by_source.setdefault(source, Scores()).add(pred, gold)
    result = {"overall": overall.summary()}
    result.update({s: sc.summary() for s, sc in sorted(overall.by_source.items())})
    return result


def rescore(predictions_path: str) -> dict[str, dict[str, float]]:
    """Score a predictions.jsonl written by evaluate.py."""
    rows = []
    with open(predictions_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append((r["source"], r["output"], json.loads(r["target"])))
    return score(rows)


if __name__ == "__main__":
    import sys

    print(json.dumps(rescore(sys.argv[1]), indent=2))
