import json
import os

from utils import (atomic_write_json, extract_labeled_line, extract_last_number,
                   numbers_equal, sentence_count, word_count)


def test_atomic_write_json(tmp_path):
    path = str(tmp_path / "results.json")
    data = [{"task_id": "t1", "answer": "hello"}]
    atomic_write_json(path, data)
    with open(path) as f:
        assert json.load(f) == data
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_extract_last_number():
    assert extract_last_number("The answer is 144 items.") == 144
    assert extract_last_number("Total: $1,250.50") == 1250.50
    assert extract_last_number("Answer: 195.") == 195
    assert extract_last_number("no numbers here") is None


def test_extract_labeled_line():
    text = "Some reasoning.\nExpression: 240 * 0.85 - 60\nAnswer: 144 items"
    assert extract_labeled_line(text, "Expression") == "240 * 0.85 - 60"
    assert extract_labeled_line(text, "Answer") == "144 items"
    assert extract_labeled_line("**Answer:** 42", "Answer") == "42"
    assert extract_labeled_line(text, "Missing") is None


def test_numbers_equal():
    assert numbers_equal(144, 144.0)
    assert numbers_equal(0.1 + 0.2, 0.3)
    assert not numbers_equal(144, 145)
    assert not numbers_equal(None, 1)


def test_counts():
    assert sentence_count("One. Two! Three?") == 3
    assert sentence_count("Just one sentence with 3.5 in it.") == 1
    assert word_count("a b  c") == 3
