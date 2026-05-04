"""测试 annotation/response_parser.py"""

import pytest
from src.annotation.response_parser import VLMResponseParser
from src.annotation.vlm_backends import VLMDetection


class TestVLMResponseParser:
    def setup_method(self):
        self.parser = VLMResponseParser()

    # ---- 直接 JSON 数组 ----
    def test_direct_json_array(self):
        text = '''[{"class": "pet", "bbox": [10, 20, 60, 80], "confidence": 0.85}]'''
        result = self.parser.parse(text)
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "pet"
        assert result.detections[0].bbox == [10, 20, 60, 80]
        assert result.detections[0].confidence == 0.85

    # ---- Markdown 代码块 ----
    def test_markdown_json_block(self):
        text = '''```json
[{"class": "dog", "bbox": [5, 10, 55, 70], "confidence": 0.72}]
```'''
        result = self.parser.parse(text)
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "dog"

    def test_markdown_plain_block(self):
        text = '''```\n[{"class": "bag", "bbox": [30, 40, 80, 90], "confidence": 0.66}]\n```'''
        result = self.parser.parse(text)
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "bag"

    # ---- 字典格式 ----
    def test_wrapped_in_dict_detections_key(self):
        text = '''{"detections": [{"class": "phone", "bbox": [100, 200, 150, 250], "confidence": 0.9}]}'''
        result = self.parser.parse(text)
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "phone"

    def test_wrapped_in_dict_results_key(self):
        text = '''{"results": [{"class": "hand", "bbox": [0, 0, 100, 100], "confidence": 0.55}]}'''
        result = self.parser.parse(text)
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "hand"

    # ---- 独立字段格式 ----
    def test_individual_fields_xmin_xmax(self):
        text = '''[{"class": "driver", "xmin": 10, "ymin": 20, "xmax": 50, "ymax": 80, "confidence": 0.88}]'''
        result = self.parser.parse(text)
        assert len(result.detections) == 1
        assert result.detections[0].bbox == [10, 20, 50, 80]

    def test_individual_fields_x1_x2(self):
        text = '''[{"class": "pet", "x1": 5, "y1": 10, "x2": 55, "y2": 60, "score": 0.7}]'''
        result = self.parser.parse(text)
        assert len(result.detections) == 1
        assert result.detections[0].bbox == [5, 10, 55, 60]
        assert result.detections[0].confidence == 0.7

    # ---- 正则回退 ----
    def test_regex_class_bbox_pattern(self):
        text = 'Found: class: pet, bbox: [10, 20, 30, 40] in the image'
        result = self.parser.parse(text)
        assert len(result.detections) >= 1
        det = result.detections[0]
        assert det.class_name == "pet"
        assert det.bbox == [10, 20, 30, 40]

    def test_regex_box_tag_format(self):
        text = '<box>(10, 20),(100, 200)</box>'
        result = self.parser.parse(text)
        assert len(result.detections) >= 1
        assert result.detections[0].bbox == [10, 20, 100, 200]

    def test_regex_ref_box_format(self):
        text = '<ref>pet</ref><box>(15, 25),(85, 95)</box>'
        result = self.parser.parse(text)
        assert len(result.detections) >= 1
        assert result.detections[0].class_name == "pet"
        assert result.detections[0].bbox == [15, 25, 85, 95]

    # ---- 边界情况 ----
    def test_empty_string(self):
        result = self.parser.parse("")
        assert result.detections == []
        assert result.raw_output == ""

    def test_truncated_json(self):
        text = '[{"class": "pet", "bbox": [10, 20, 60'
        result = self.parser.parse(text)
        # 应优雅降级，返回空
        assert isinstance(result.detections, list)

    def test_multiple_detections(self):
        text = '''[
            {"class": "driver", "bbox": [10, 10, 50, 50], "confidence": 0.9},
            {"class": "passenger", "bbox": [100, 10, 140, 50], "confidence": 0.85}
        ]'''
        result = self.parser.parse(text)
        assert len(result.detections) == 2
        assert result.detections[0].class_name == "driver"
        assert result.detections[1].class_name == "passenger"
