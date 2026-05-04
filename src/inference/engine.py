"""
基础模型推理模块

支持两种后端：
  1. simulated — 直接使用模拟器生成的 ground-truth 元数据，添加随机扰动模拟真实推理
  2. yolo     — 使用 YOLOv8 进行真实推理（需安装 ultralytics）
"""

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection:
    """单个检测结果"""
    class_name: str
    confidence: float
    bbox: list            # [x1, y1, x2, y2]
    frame_id: int = -1


@dataclass
class InferenceResult:
    """单帧推理结果"""
    frame_id: int
    detections: list = field(default_factory=list)   # List[Detection]
    raw_confidences: list = field(default_factory=list)  # 所有类别的原始置信度向量


class BaseInferenceEngine:
    """推理引擎基类"""

    def __init__(self, config: dict):
        inf_cfg = config.get("inference", {})
        self.conf_threshold = inf_cfg.get("conf_threshold", 0.25)
        self.classes = inf_cfg.get("classes", [
            "driver", "passenger", "child_seat", "pet",
            "left_object", "phone_usage", "smoking", "seatbelt_off", "normal"
        ])
        self.device = inf_cfg.get("device", "cpu")

    def infer(self, frame: np.ndarray, frame_id: int = -1) -> InferenceResult:
        raise NotImplementedError


class SimulatedInferenceEngine(BaseInferenceEngine):
    """模拟推理引擎

    直接利用模拟器产出的 ground-truth 元数据，并添加随机噪声来模拟
    真实模型推理中的置信度波动和检测框偏移，使下游不确定性采样有意义。
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # 模拟推理噪声参数
        self.conf_noise_std = 0.08       # 置信度高斯噪声标准差
        self.bbox_noise_pixels = 12      # 检测框偏移像素范围

    def infer(self, frame: np.ndarray, frame_id: int = -1,
              gt_objects: Optional[list] = None) -> InferenceResult:
        """
        Args:
            frame: 输入图像 (H, W, 3)
            frame_id: 帧编号
            gt_objects: 模拟器提供的 ground-truth 对象列表
                        [{"class": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
        """
        detections = []
        raw_confs = []

        if gt_objects is None:
            gt_objects = []

        h, w = frame.shape[:2]

        for obj in gt_objects:
            # 对置信度添加噪声
            base_conf = obj.get("confidence", 0.5)
            noisy_conf = base_conf + random.gauss(0, self.conf_noise_std)
            noisy_conf = max(0.01, min(0.99, noisy_conf))

            # 生成该检测对应的完整类别置信度向量
            class_idx = self.classes.index(obj["class"]) if obj["class"] in self.classes else -1
            conf_vector = [random.uniform(0.01, 0.15) for _ in self.classes]
            if class_idx >= 0:
                conf_vector[class_idx] = noisy_conf
            # 对检测框添加像素偏移噪声
            bbox = list(obj["bbox"])
            for i in range(4):
                bbox[i] += random.randint(-self.bbox_noise_pixels, self.bbox_noise_pixels)
            # 裁剪到图像范围内
            bbox[0] = max(0, min(bbox[0], w - 1))
            bbox[1] = max(0, min(bbox[1], h - 1))
            bbox[2] = max(bbox[0] + 1, min(bbox[2], w))
            bbox[3] = max(bbox[1] + 1, min(bbox[3], h))

            # 随机 False Negative：小概率丢弃一个真实检测
            if random.random() < 0.05:
                continue

            # 生成该检测对应的完整类别置信度向量
            class_idx = self.classes.index(obj["class"]) if obj["class"] in self.classes else -1
            conf_vector = [random.uniform(0.01, 0.15) for _ in self.classes]
            if class_idx >= 0:
                conf_vector[class_idx] = noisy_conf
            # 归一化
            total = sum(conf_vector)
            conf_vector = [c / total for c in conf_vector]
            raw_confs.append(conf_vector)

            det = Detection(
                class_name=obj["class"],
                confidence=round(noisy_conf, 4),
                bbox=bbox,
                frame_id=frame_id,
            )
            detections.append(det)

        # 随机 False Positive：小概率产生一个虚假检测
        if random.random() < 0.08:
            fp_class = random.choice(self.classes)
            fp_x1 = random.randint(0, w - 40)
            fp_y1 = random.randint(0, h - 40)
            fp_x2 = fp_x1 + random.randint(20, 60)
            fp_y2 = fp_y1 + random.randint(20, 60)
            fp_conf = random.uniform(0.10, 0.40)
            detections.append(Detection(
                class_name=fp_class,
                confidence=round(fp_conf, 4),
                bbox=[fp_x1, fp_y1, min(fp_x2, w), min(fp_y2, h)],
                frame_id=frame_id,
            ))

        return InferenceResult(
            frame_id=frame_id,
            detections=detections,
            raw_confidences=raw_confs,
        )


class YOLOInferenceEngine(BaseInferenceEngine):
    """YOLOv8 推理引擎（需安装 ultralytics）"""

    def __init__(self, config: dict):
        super().__init__(config)
        inf_cfg = config.get("inference", {})
        weights = inf_cfg.get("yolo_weights", "yolov8n.pt")
        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
        except ImportError:
            raise ImportError("使用 YOLO 推理需要安装 ultralytics: pip install ultralytics")

    def infer(self, frame: np.ndarray, frame_id: int = -1, **kwargs) -> InferenceResult:
        results = self.model(frame, conf=self.conf_threshold, device=self.device, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].tolist()
                cls_name = r.names.get(cls_id, f"class_{cls_id}")
                detections.append(Detection(
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    bbox=[int(v) for v in xyxy],
                    frame_id=frame_id,
                ))

        return InferenceResult(frame_id=frame_id, detections=detections)


def create_engine(config: dict) -> BaseInferenceEngine:
    """工厂方法：根据配置创建推理引擎"""
    model_type = config.get("inference", {}).get("model_type", "simulated")
    if model_type == "yolo":
        return YOLOInferenceEngine(config)
    return SimulatedInferenceEngine(config)
