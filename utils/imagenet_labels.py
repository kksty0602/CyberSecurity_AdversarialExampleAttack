"""
utils/imagenet_labels.py
提供 ImageNet-1K 人类可读标签列表，供 Streamlit 下拉选择框使用。
"""

import json
import urllib.request
from typing import List

LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
    "master/imagenet-simple-labels.json"
)


def load_labels() -> List[str]:
    """从网络加载 ImageNet 标签，失败时返回数字 ID 列表作为回退。"""
    try:
        with urllib.request.urlopen(LABELS_URL, timeout=10) as response:
            labels = json.loads(response.read().decode("utf-8"))
        return labels
    except Exception:
        return [str(i) for i in range(1000)]


def get_label_options() -> List[str]:
    """返回带索引前缀的标签列表，如 ['0: tench', '1: goldfish', ...]。"""
    labels = load_labels()
    return [f"{i}: {name}" for i, name in enumerate(labels)]


def parse_label_option(option: str) -> int:
    """从 '7: cock' 格式的字符串中解析出类别索引。"""
    return int(option.split(":")[0])
