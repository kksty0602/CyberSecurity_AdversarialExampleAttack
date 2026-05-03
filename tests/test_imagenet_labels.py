import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.imagenet_labels import load_labels, get_label_options, parse_label_option


def test_load_labels_returns_1000_items():
    labels = load_labels()
    assert len(labels) == 1000
    assert isinstance(labels[0], str)


def test_get_label_options_format():
    options = get_label_options()
    assert len(options) == 1000
    assert options[0].startswith("0: ")
    assert options[7].startswith("7: ")


def test_parse_label_option():
    assert parse_label_option("7: cock") == 7
    assert parse_label_option("123: some_label") == 123
