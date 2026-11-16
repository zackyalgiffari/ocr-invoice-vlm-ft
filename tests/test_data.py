import json

import pytest

from invoice_vlm.data import INSTRUCTIONS, SOURCES, build_messages, parse_ground_truth, to_target

# Shapes copied from the first train rows of each dataset (datasets-server, 2026-11-16).
CORD_GT = json.dumps(
    {
        "gt_parse": {
            "menu": [{"nm": "Nasi Campur Bali", "cnt": "1 x", "price": "75,000"}],
            "total": {"total_price": "1,591,600"},
        },
        "meta": {"version": "2.0.0", "split": "train"},
    }
)
INVOICE_GT = json.dumps(
    {
        "gt_parse": {
            "header": {"invoice_no": "40378170", "invoice_date": "10/15/2012"},
            "items": [{"item_desc": "Corkscrew", "item_qty": "1,00"}],
            "summary": {"total_gross_worth": "$8,25"},
        }
    }
)


def test_parse_ground_truth_returns_inner_object_without_meta():
    label = parse_ground_truth(CORD_GT)
    assert set(label) == {"menu", "total"}
    assert label["menu"][0]["nm"] == "Nasi Campur Bali"


def test_parse_ground_truth_accepts_gt_parses_list():
    assert parse_ground_truth(json.dumps({"gt_parses": [{"a": 1}, {"a": 2}]})) == {"a": 1}


def test_parse_ground_truth_rejects_unknown_shape():
    with pytest.raises(ValueError):
        parse_ground_truth(json.dumps({"something": {}}))


def test_to_target_is_compact_roundtrippable_and_keeps_unicode():
    label = {"nm": "Kopi Susu Gula Aren", "price": "Rp 25.000", "note": "café"}
    target = to_target(label)
    assert ": " not in target and ", " not in target
    assert "café" in target
    assert json.loads(target) == label


def test_build_messages_training_and_inference_shapes():
    image = object()
    train = build_messages(image, "invoice", '{"a":1}')
    assert [m["role"] for m in train] == ["user", "assistant"]
    assert train[0]["content"][0] == {"type": "image", "image": image}
    assert train[0]["content"][1]["text"] == INSTRUCTIONS["invoice"]
    assert train[1]["content"] == [{"type": "text", "text": '{"a":1}'}]

    infer = build_messages(image, "receipt")
    assert len(infer) == 1 and infer[0]["content"][1]["text"] == INSTRUCTIONS["receipt"]


def test_every_source_has_an_instruction():
    assert all(src.doc_type in INSTRUCTIONS for src in SOURCES.values())
    assert parse_ground_truth(INVOICE_GT)["header"]["invoice_no"] == "40378170"
