import pytest

from invoice_vlm.metrics import extract_json, flatten, score

GOLD = {
    "menu": [{"nm": "Es Teh", "price": "5,000"}, {"nm": "Nasi Goreng", "price": "25,000"}],
    "total": {"total_price": "30,000"},
}


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '<think>\n\n</think>\n\n{"a": 1}',
        'Here is the result: {"a": 1} hope it helps',
    ],
)
def test_extract_json_tolerates_wrappers(text):
    assert extract_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "no json here", '{"a": 1', "{broken}"])
def test_extract_json_returns_none_on_garbage(text):
    assert extract_json(text) is None


def test_flatten_drops_list_indices_and_normalizes():
    flat = flatten({"menu": [{"nm": "  Es  Teh "}, {"nm": "es teh"}], "x": "", "y": None})
    assert flat == {("menu[].nm", "es teh"): 2}


def test_perfect_prediction_with_reordered_items():
    pred = {"total": {"total_price": "30,000"}, "menu": list(reversed(GOLD["menu"]))}
    s = score([("cord", str(pred).replace("'", '"'), GOLD)])["overall"]
    assert s["exact_match"] == 1.0 and s["f1"] == 1.0 and s["json_valid"] == 1.0


def test_partial_prediction_precision_recall():
    pred = '{"menu": [{"nm": "Es Teh", "price": "5,000"}], "total": {"total_price": "31,000"}}'
    s = score([("cord", pred, GOLD)])["overall"]
    # gold has 5 leaves; pred has 3, of which 2 are correct.
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 5)
    assert s["exact_match"] == 0.0


def test_value_f1_credits_right_values_under_wrong_keys():
    # What the zero-shot baseline does: correct reading, invented schema.
    pred = '{"items": [{"name": "Es Teh", "cost": "5,000"}, {"name": "Nasi Goreng", "cost": "25,000"}], "grand_total": "30,000"}'
    s = score([("cord", pred, GOLD)])["overall"]
    assert s["f1"] == 0.0
    assert s["value_f1"] == 1.0


def test_rescore_reads_evaluate_output(tmp_path):
    import json

    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"source": "cord", "target": '{"a":"1"}', "output": '{"a": "1"}'}) + "\n",
        encoding="utf-8",
    )
    from invoice_vlm.metrics import rescore

    assert rescore(str(path))["overall"]["f1"] == 1.0


def test_invalid_json_counts_as_zero_recall_and_per_source_split():
    result = score([("cord", "garbage", GOLD), ("invoices", '{"a": "1"}', {"a": "1"})])
    assert result["overall"]["json_valid"] == 0.5
    assert result["cord"]["recall"] == 0.0
    assert result["invoices"]["f1"] == 1.0


def test_lenient_forgives_label_whitespace_and_single_item_object():
    gold = {"menu": {"nm": "Es Teh", "price": "5,000"}, "total": {"total_price": "$ 1 490,55"}}
    pred = '{"menu": [{"nm": "Es Teh", "price": "5,000"}], "total": {"total_price": "$1490,55"}}'
    s = score([("cord", pred, gold)])["overall"]
    assert s["f1"] == 0.0 and s["exact_match"] == 0.0
    assert s["lenient_f1"] == 1.0 and s["lenient_exact_match"] == 1.0


def test_lenient_still_penalizes_wrong_values():
    pred = '{"menu": [{"nm": "Es Teh", "price": "5,000"}, {"nm": "Nasi Goreng", "price": "25,000"}], "total": {"total_price": "31,000"}}'
    s = score([("cord", pred, GOLD)])["overall"]
    assert s["lenient_f1"] == pytest.approx(0.8) and s["lenient_exact_match"] == 0.0
