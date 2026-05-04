"""测试 utils/helpers.py"""

import pytest
from src.utils.helpers import compute_iou, _deep_merge, DEFAULT_CONFIG


class TestComputeIoU:
    def test_perfect_overlap(self):
        assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0

    def test_no_overlap(self):
        assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_partial_overlap(self):
        # 两个 10x10 框，偏移 5px → 交集 5x10
        iou = compute_iou([0, 0, 10, 10], [5, 0, 15, 10])
        # 交集=50, 并集=100+100-50=150, IoU=0.333...
        assert abs(iou - 1 / 3) < 0.01

    def test_one_contains_another(self):
        iou = compute_iou([0, 0, 20, 20], [5, 5, 15, 15])
        # 交集=100, 并集=400, IoU=0.25
        assert abs(iou - 0.25) < 0.01

    def test_zero_area(self):
        assert compute_iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_nested_override(self):
        base = {"parent": {"x": 1, "y": 2}}
        override = {"parent": {"y": 200}}
        result = _deep_merge(base, override)
        assert result == {"parent": {"x": 1, "y": 200}}

    def test_new_key_added(self):
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_original_not_mutated(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}  # 原对象未被修改

    def test_override_with_empty_dict(self):
        result = _deep_merge(DEFAULT_CONFIG, {})
        assert result == DEFAULT_CONFIG
