"""
标注格式转换器

将 VLM 标注结果转换为标准目标检测数据集格式：
  1. YOLO 格式 — 每张图像一个 .txt，归一化中心点坐标
  2. COCO 格式 — 全局 JSON 文件
"""

import json
import os
from datetime import datetime

import yaml

from src.annotation.vlm_backends import VLMDetection, VLMResponse


class YOLOFormatConverter:
    """YOLO 格式转换器

    输出格式（每行一个目标）:
        <class_id> <x_center> <y_center> <width> <height>
    坐标均为归一化值 (0~1)。
    同时生成 classes.txt 类别映射文件。
    """

    def __init__(self, class_names: list[str], output_dir: str):
        """
        Args:
            class_names: 类别列表，索引即为 class_id
            output_dir: 输出目录（将创建 images/ 和 labels/ 子目录）
        """
        self.class_names = list(class_names)
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        self._write_classes_file()

    def _write_classes_file(self):
        """写入 classes.txt"""
        path = os.path.join(self.output_dir, "classes.txt")
        with open(path, "w", encoding="utf-8") as f:
            for name in self.class_names:
                f.write(f"{name}\n")

    def _bbox_to_yolo(self, bbox: list, img_w: int, img_h: int) -> tuple:
        """将 [x1, y1, x2, y2] 绝对像素坐标转换为 YOLO 归一化格式

        Returns:
            (x_center, y_center, width, height) 均归一化到 0~1
        """
        x1, y1, x2, y2 = bbox
        x_center = ((x1 + x2) / 2.0) / img_w
        y_center = ((y1 + y2) / 2.0) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        # 裁剪到合法范围
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        return x_center, y_center, w, h

    def add_class(self, class_name: str) -> int:
        """动态添加新类别"""
        if class_name not in self.class_to_id:
            idx = len(self.class_names)
            self.class_names.append(class_name)
            self.class_to_id[class_name] = idx
            self._write_classes_file()
            return idx
        return self.class_to_id[class_name]

    def convert(self, image_name: str, detections: list[VLMDetection],
                img_w: int, img_h: int) -> str:
        """将单张图像的检测结果转换为 YOLO label 文件

        Args:
            image_name: 图像文件名（如 frame000021.jpg）
            detections: VLM 检测结果列表
            img_w: 图像宽度
            img_h: 图像高度

        Returns:
            label 文件路径
        """
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_name)

        lines = []
        for det in detections:
            class_id = self.class_to_id.get(det.class_name)
            if class_id is None:
                class_id = self.add_class(det.class_name)
            xc, yc, w, h = self._bbox_to_yolo(det.bbox, img_w, img_h)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return label_path

    def generate_data_yaml(self, dataset_name: str = "cabin_dataset") -> str:
        """生成 YOLO 训练用的 data.yaml"""
        data = {
            "path": os.path.abspath(self.output_dir),
            "train": "images",
            "val": "images",
            "names": {i: name for i, name in enumerate(self.class_names)},
            "nc": len(self.class_names),
        }
        yaml_path = os.path.join(self.output_dir, f"{dataset_name}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return yaml_path


class COCOFormatConverter:
    """COCO 格式转换器

    输出格式：单个 JSON 文件，包含 images, annotations, categories 三个列表。
    """

    def __init__(self, class_names: list[str], output_dir: str):
        self.class_names = list(class_names)
        self.class_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}  # COCO id 从 1 开始
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._images = []
        self._annotations = []
        self._image_id_counter = 0
        self._ann_id_counter = 0

    def _bbox_to_coco(self, bbox: list) -> list:
        """将 [x1, y1, x2, y2] 转换为 COCO 格式 [x, y, width, height]"""
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]

    def add_class(self, class_name: str) -> int:
        """动态添加新类别"""
        if class_name not in self.class_to_id:
            idx = len(self.class_names) + 1
            self.class_names.append(class_name)
            self.class_to_id[class_name] = idx
            return idx
        return self.class_to_id[class_name]

    def add_image(self, image_name: str, detections: list[VLMDetection],
                  img_w: int, img_h: int) -> int:
        """添加一张图像及其标注

        Returns:
            image_id
        """
        self._image_id_counter += 1
        img_id = self._image_id_counter

        self._images.append({
            "id": img_id,
            "file_name": image_name,
            "width": img_w,
            "height": img_h,
        })

        for det in detections:
            cat_id = self.class_to_id.get(det.class_name)
            if cat_id is None:
                cat_id = self.add_class(det.class_name)

            self._ann_id_counter += 1
            coco_bbox = self._bbox_to_coco(det.bbox)
            area = coco_bbox[2] * coco_bbox[3]

            self._annotations.append({
                "id": self._ann_id_counter,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": coco_bbox,
                "area": area,
                "iscrowd": 0,
                "score": det.confidence,
            })

        return img_id

    def save(self, filename: str = "annotations.json") -> str:
        """将完整 COCO 数据集保存为 JSON"""
        categories = [
            {"id": idx + 1, "name": name, "supercategory": "cabin_object"}
            for idx, name in enumerate(self.class_names)
        ]

        coco_data = {
            "info": {
                "description": "VLM Auto-annotated Cabin Dataset",
                "version": "1.0",
                "year": 2026,
                "date_created": datetime.now().isoformat(),
            },
            "images": self._images,
            "annotations": self._annotations,
            "categories": categories,
        }

        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)

        return output_path

    def get_statistics(self) -> dict:
        """返回数据集统计"""
        cat_counts = {}
        for ann in self._annotations:
            cat_id = ann["category_id"]
            name = self.class_names[cat_id - 1] if cat_id <= len(self.class_names) else "unknown"
            cat_counts[name] = cat_counts.get(name, 0) + 1

        return {
            "total_images": len(self._images),
            "total_annotations": len(self._annotations),
            "category_counts": cat_counts,
            "num_categories": len(self.class_names),
        }
