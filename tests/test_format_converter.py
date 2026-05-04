"""测试 annotation/format_converter.py"""

import os
import json
import tempfile
import pytest
from src.annotation.vlm_backends import VLMDetection


def _make_det(cls, bbox, conf=0.9):
    return VLMDetection(class_name=cls, bbox=bbox, confidence=conf, raw_text="")


class TestYOLOFormatConverter:
    def test_bbox_to_yolo(self):
        from src.annotation.format_converter import YOLOFormatConverter

        conv = YOLOFormatConverter(["driver", "pet"], tempfile.mkdtemp())
        # bbox [10, 20, 60, 80] in 100x200 image
        xc, yc, w, h = conv._bbox_to_yolo([10, 20, 60, 80], 100, 200)
        assert abs(xc - 0.35) < 0.01  # (10+60)/2 / 100
        assert abs(yc - 0.25) < 0.01  # (20+80)/2 / 200
        assert abs(w - 0.50) < 0.01   # (60-10) / 100
        assert abs(h - 0.30) < 0.01   # (80-20) / 200

    def test_convert_writes_label_file(self):
        from src.annotation.format_converter import YOLOFormatConverter

        tmpdir = tempfile.mkdtemp()
        conv = YOLOFormatConverter(["driver", "pet", "bag"], tmpdir)

        dets = [
            _make_det("driver", [10, 20, 60, 80], 0.9),
            _make_det("pet", [30, 40, 70, 90], 0.7),
        ]
        label_path = conv.convert("frame001.jpg", dets, 100, 200)

        assert os.path.isfile(label_path)
        with open(label_path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        # 第一行: class_id=0 (driver), 检查格式
        parts = lines[0].strip().split()
        assert len(parts) == 5
        assert int(parts[0]) == 0  # driver

    def test_dynamic_class_addition(self):
        from src.annotation.format_converter import YOLOFormatConverter

        tmpdir = tempfile.mkdtemp()
        conv = YOLOFormatConverter(["driver"], tmpdir)

        # 新类别 "unknown_thing" 应被自动添加
        idx = conv.add_class("unknown_thing")
        assert idx == 1
        assert "unknown_thing" in conv.class_names

    def test_generate_data_yaml(self):
        from src.annotation.format_converter import YOLOFormatConverter

        tmpdir = tempfile.mkdtemp()
        conv = YOLOFormatConverter(["driver", "pet"], tmpdir)
        yaml_path = conv.generate_data_yaml("test_dataset")

        assert os.path.isfile(yaml_path)
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert data["nc"] == 2
        assert data["names"] == {0: "driver", 1: "pet"}


class TestCOCOFormatConverter:
    def test_add_image_and_save(self):
        from src.annotation.format_converter import COCOFormatConverter

        tmpdir = tempfile.mkdtemp()
        conv = COCOFormatConverter(["driver", "pet"], tmpdir)

        dets = [
            _make_det("driver", [10, 20, 60, 80], 0.9),
            _make_det("pet", [30, 40, 70, 90], 0.7),
        ]
        img_id = conv.add_image("frame001.jpg", dets, 640, 480)
        assert img_id == 1

        coco_path = conv.save("test_annotations.json")
        assert os.path.isfile(coco_path)

        with open(coco_path) as f:
            data = json.load(f)

        assert len(data["images"]) == 1
        assert data["images"][0]["file_name"] == "frame001.jpg"
        assert data["images"][0]["width"] == 640
        assert data["images"][0]["height"] == 480

        assert len(data["annotations"]) == 2
        assert data["annotations"][0]["category_id"] == 1  # COCO id 从 1 开始
        assert data["annotations"][0]["bbox"] == [10, 20, 50, 60]  # [x, y, w, h]

        assert len(data["categories"]) == 2
        assert data["categories"][0]["name"] == "driver"

    def test_bbox_to_coco(self):
        from src.annotation.format_converter import COCOFormatConverter

        conv = COCOFormatConverter(["driver"], tempfile.mkdtemp())
        assert conv._bbox_to_coco([10, 20, 60, 80]) == [10, 20, 50, 60]

    def test_statistics(self):
        from src.annotation.format_converter import COCOFormatConverter

        tmpdir = tempfile.mkdtemp()
        conv = COCOFormatConverter(["driver", "pet"], tmpdir)

        conv.add_image("img1.jpg", [_make_det("driver", [1, 1, 10, 10])], 100, 100)
        conv.add_image("img2.jpg", [
            _make_det("pet", [1, 1, 10, 10]),
            _make_det("pet", [20, 20, 30, 30]),
        ], 100, 100)

        stats = conv.get_statistics()
        assert stats["total_images"] == 2
        assert stats["total_annotations"] == 3
        assert stats["category_counts"]["driver"] == 1
        assert stats["category_counts"]["pet"] == 2
