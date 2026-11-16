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


def test_invalid_json_counts_as_zero_recall_and_per_source_split():
    result = score([("cord", "garbage", GOLD), ("invoices", '{"a": "1"}', {"a": "1"})])
    assert result["overall"]["json_valid"] == 0.5
    assert result["cord"]["recall"] == 0.0
    assert result["invoices"]["f1"] == 1.0
