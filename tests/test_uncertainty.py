"""测试 sampling/uncertainty.py"""

import pytest
from src.sampling.uncertainty import UncertaintySampler, UncertaintyFlag
from src.inference.engine import InferenceResult, Detection


def _make_det(cls, conf, bbox, frame_id=0):
    return Detection(class_name=cls, confidence=conf, bbox=bbox, frame_id=frame_id)


def _make_result(frame_id, detections):
    return InferenceResult(frame_id=frame_id, detections=detections)


def _full_config():
    """返回完整的 sampling 配置字典（模拟 deep-merge 后的结果）"""
    return {
        "sampling": {
            "confidence_threshold": 0.4,
            "bbox_jitter_iou_threshold": 0.5,
            "jitter_window_size": 5,
            "jitter_ratio_threshold": 0.6,
            "fusion_strategy": "any",
            "history_max_age": 500,
        }
    }


class TestLowConfidence:
    def test_below_threshold(self):
        sampler = UncertaintySampler(_full_config())
        result = _make_result(1, [
            _make_det("driver", 0.35, [10, 10, 50, 50]),
        ])
        flags = sampler.evaluate(result)
        assert len(flags) >= 1
        conf_flags = [f for f in flags if f.reason == "low_confidence"]
        assert len(conf_flags) == 1
        assert conf_flags[0].details["class"] == "driver"

    def test_above_threshold(self):
        sampler = UncertaintySampler(_full_config())
        result = _make_result(1, [
            _make_det("driver", 0.85, [10, 10, 50, 50]),
        ])
        flags = sampler.evaluate(result)
        conf_flags = [f for f in flags if f.reason == "low_confidence"]
        assert len(conf_flags) == 0

    def test_mixed_confidence(self):
        sampler = UncertaintySampler(_full_config())
        result = _make_result(1, [
            _make_det("driver", 0.90, [10, 10, 50, 50]),
            _make_det("pet", 0.15, [60, 60, 100, 100]),
        ])
        flags = sampler.evaluate(result)
        conf_flags = [f for f in flags if f.reason == "low_confidence"]
        assert len(conf_flags) == 1
        assert conf_flags[0].details["class"] == "pet"


class TestBBoxJitter:
    def test_stable_bboxes_no_jitter(self):
        sampler = UncertaintySampler(_full_config())
        # 连续 4 帧同样位置
        for fid in range(4):
            result = _make_result(fid, [
                _make_det("driver", 0.9, [10, 10, 50, 50], fid),
            ])
            flags = sampler.evaluate(result)

        # 最后一帧不应该有 jitter
        result = _make_result(4, [
            _make_det("driver", 0.9, [10, 10, 50, 50], 4),
        ])
        flags = sampler.evaluate(result)
        jitter_flags = [f for f in flags if f.reason == "bbox_jitter"]
        assert len(jitter_flags) == 0

    def test_jitter_detection(self):
        sampler = UncertaintySampler(_full_config())
        # 大幅抖动：每帧位置跳跃
        positions = [
            [10, 10, 50, 50],
            [200, 200, 240, 240],  # 距离很大 → IoU ≈ 0
            [10, 10, 50, 50],       # 距离很大 → IoU ≈ 0
            [200, 200, 240, 240],  # 距离很大 → IoU ≈ 0
        ]
        for fid, bbox in enumerate(positions):
            result = _make_result(fid, [
                _make_det("pet", 0.9, bbox, fid),
            ])
            sampler.evaluate(result)  # 填充历史

        # 第 5 帧触发检查
        result = _make_result(4, [
            _make_det("pet", 0.9, [10, 10, 50, 50], 4),
        ])
        flags = sampler.evaluate(result)
        jitter_flags = [f for f in flags if f.reason == "bbox_jitter"]
        # 窗口内有 4 对相邻帧，全部 IoU ≈ 0 → jitter_ratio = 1.0 >= 0.6
        assert len(jitter_flags) >= 1
        assert jitter_flags[0].details["class"] == "pet"


class TestFusionStrategy:
    def test_fusion_any(self):
        config = _full_config()
        config["sampling"]["fusion_strategy"] = "any"
        sampler = UncertaintySampler(config)

        # 低置信度 + 稳定框 → any 会报 low_confidence
        for fid in range(4):
            result = _make_result(fid, [
                _make_det("driver", 0.9, [10, 10, 50, 50], fid),
                _make_det("pet", 0.15, [60, 60, 100, 100], fid),  # 低置信度
            ])
            sampler.evaluate(result)

        result = _make_result(4, [
            _make_det("driver", 0.9, [10, 10, 50, 50], 4),
            _make_det("pet", 0.15, [60, 60, 100, 100], 4),
        ])
        flags = sampler.evaluate(result)
        reasons = {f.reason for f in flags}
        assert "low_confidence" in reasons

    def test_fusion_all(self):
        config = _full_config()
        config["sampling"]["fusion_strategy"] = "all"
        sampler = UncertaintySampler(config)

        # 只有低置信度、没有 jitter → all 不报
        result = _make_result(1, [
            _make_det("pet", 0.15, [60, 60, 100, 100]),
        ])
        flags = sampler.evaluate(result)
        assert len(flags) == 0


class TestHistoryPruning:
    def test_stale_entry_removed(self):
        sampler = UncertaintySampler(_full_config())
        sampler.history_max_age = 3

        # 第 0 帧出现 pet
        sampler.evaluate(_make_result(0, [
            _make_det("pet", 0.9, [10, 10, 50, 50], 0),
        ]))
        assert "pet" in sampler._history

        # 第 10 帧触发清理（>= max_age=3 后），pet 已过期
        sampler.evaluate(_make_result(10, [
            _make_det("driver", 0.9, [20, 20, 60, 60], 10),
        ]))
        assert "pet" not in sampler._history
