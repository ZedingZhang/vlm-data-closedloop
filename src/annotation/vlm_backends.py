"""
VLM 推理后端

支持多种 VLM 接入方式：
  1. SimulatedVLM   — 模拟 VLM 输出（无需 GPU，用于开发测试）
  2. QwenVLBackend  — 调用 Qwen-VL-Chat（本地部署或 API）
  3. GroundingDINOBackend — 调用 Grounding-DINO（本地部署或 API）
"""

import json
import random
import base64
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class VLMDetection:
    """VLM 返回的单个检测结果"""
    class_name: str
    bbox: list              # [x_min, y_min, x_max, y_max] 绝对像素坐标
    confidence: float
    raw_text: str = ""      # VLM 原始返回文本片段


@dataclass
class VLMResponse:
    """VLM 完整响应"""
    detections: list = field(default_factory=list)   # List[VLMDetection]
    raw_output: str = ""
    task_type: str = "detection"
    metadata: dict = field(default_factory=dict)


class VLMBackend(ABC):
    """VLM 后端基类"""

    @abstractmethod
    def infer(self, image: np.ndarray, prompt: dict) -> VLMResponse:
        """对单张图像执行 VLM 推理

        Args:
            image: BGR 图像 (H, W, 3)
            prompt: PromptFactory.build_prompt() 的输出

        Returns:
            VLMResponse
        """
        pass


class SimulatedVLMBackend(VLMBackend):
    """模拟 VLM 后端

    根据图像中的像素特征和 prompt 中的目标类别，生成模拟的检测结果。
    用于在无 GPU 环境下测试完整的标注管线。
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    def _analyze_image_features(self, image: np.ndarray) -> dict:
        """通过简单的图像分析提取特征，辅助模拟检测"""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

        # 检测暗区域（可能是遮挡）
        dark_mask = gray < 30
        dark_ratio = float(np.sum(dark_mask)) / (h * w)

        # 检测亮区域（可能是过曝）
        bright_mask = gray > 230
        bright_ratio = float(np.sum(bright_mask)) / (h * w)

        # 检测绿色区域（模拟器中宠物用绿色绘制）
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        green_ratio = float(np.sum(green_mask > 0)) / (h * w)

        # 检测蓝色区域（模拟器中遗留物品用蓝色绘制）
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        blue_ratio = float(np.sum(blue_mask > 0)) / (h * w)

        return {
            "height": h, "width": w,
            "mean_brightness": mean_brightness,
            "std_brightness": std_brightness,
            "dark_ratio": dark_ratio,
            "bright_ratio": bright_ratio,
            "green_ratio": green_ratio,
            "blue_ratio": blue_ratio,
        }

    def _find_colored_region(self, image: np.ndarray,
                             lower: tuple, upper: tuple) -> Optional[list]:
        """在图像中查找特定颜色区域的边界框"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                x, y, cw, ch = cv2.boundingRect(largest)
                return [x, y, x + cw, y + ch]
        return None

    def infer(self, image: np.ndarray, prompt: dict) -> VLMResponse:
        h, w = image.shape[:2]
        features = self._analyze_image_features(image)
        target_classes = prompt.get("target_classes", [])
        task_type = prompt.get("task_type", "detection")
        detections = []

        if task_type == "detection":
            # 根据图像特征生成模拟检测
            # 绿色区域 → 宠物
            if features["green_ratio"] > 0.005:
                bbox = self._find_colored_region(image, (35, 50, 50), (85, 255, 255))
                if bbox:
                    detections.append(VLMDetection(
                        class_name="pet",
                        bbox=bbox,
                        confidence=round(random.uniform(0.60, 0.92), 3),
                    ))

            # 蓝色区域 → 遗留物品
            if features["blue_ratio"] > 0.003:
                bbox = self._find_colored_region(image, (100, 50, 50), (130, 255, 255))
                if bbox:
                    detections.append(VLMDetection(
                        class_name="left_object",
                        bbox=bbox,
                        confidence=round(random.uniform(0.55, 0.88), 3),
                    ))

            # 大面积暗区域 → 遮挡
            if features["dark_ratio"] > 0.2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                dark_mask = (gray < 30).astype(np.uint8) * 255
                contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, cw, ch = cv2.boundingRect(largest)
                    if cw * ch > (w * h * 0.05):
                        detections.append(VLMDetection(
                            class_name="obstruction",
                            bbox=[x, y, x + cw, y + ch],
                            confidence=round(random.uniform(0.50, 0.85), 3),
                        ))

            # 始终尝试检测驾驶员头部（肤色区域）
            bbox = self._find_colored_region(image, (0, 20, 80), (25, 180, 200))
            if bbox:
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if 15 < bw < w // 2 and 15 < bh < h // 2:
                    detections.append(VLMDetection(
                        class_name="driver",
                        bbox=bbox,
                        confidence=round(random.uniform(0.70, 0.95), 3),
                    ))

        elif task_type == "classification":
            # 光照分类
            if features["bright_ratio"] > 0.3:
                condition = "overexposed"
                severity = min(1.0, features["bright_ratio"])
            elif features["mean_brightness"] < 40:
                condition = "underexposed"
                severity = 1.0 - features["mean_brightness"] / 100
            else:
                condition = "normal"
                severity = 0.1
            detections.append(VLMDetection(
                class_name=condition,
                bbox=[0, 0, w, h],
                confidence=round(severity, 3),
            ))

        # 构造原始输出文本
        raw_json = json.dumps([
            {"class": d.class_name, "bbox": d.bbox, "confidence": d.confidence}
            for d in detections
        ], indent=2)

        return VLMResponse(
            detections=detections,
            raw_output=raw_json,
            task_type=task_type,
            metadata=features,
        )


class QwenVLBackend(VLMBackend):
    """Qwen-VL-Chat 后端

    支持两种调用方式：
      1. 本地 transformers 推理
      2. OpenAI 兼容 API（vLLM / Ollama 等部署）
    """

    def __init__(self, config: dict = None):
        cfg = (config or {}).get("vlm", {}).get("qwen", {})
        self.mode = cfg.get("mode", "api")           # "local" | "api"
        self.api_base = cfg.get("api_base", "http://localhost:8000/v1")
        self.model_name = cfg.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.max_tokens = cfg.get("max_tokens", 1024)
        self._model = None
        self._processor = None

    def _load_local_model(self):
        """懒加载本地模型"""
        if self._model is not None:
            return
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, device_map="auto"
        )

    def _encode_image(self, image: np.ndarray) -> str:
        """将 OpenCV 图像编码为 base64"""
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def infer(self, image: np.ndarray, prompt: dict) -> VLMResponse:
        if self.mode == "local":
            return self._infer_local(image, prompt)
        return self._infer_api(image, prompt)

    def _infer_local(self, image: np.ndarray, prompt: dict) -> VLMResponse:
        """本地 transformers 推理"""
        self._load_local_model()
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{self._encode_image(image)}"},
                {"type": "text", "text": prompt["user"]},
            ]},
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(text=[text], images=image_inputs, videos=video_inputs,
                                 padding=True, return_tensors="pt").to(self._model.device)
        output_ids = self._model.generate(**inputs, max_new_tokens=self.max_tokens)
        trimmed = output_ids[0][len(inputs.input_ids[0]):]
        raw_output = self._processor.decode(trimmed, skip_special_tokens=True)
        return self._parse_response(raw_output, prompt)

    def _infer_api(self, image: np.ndarray, prompt: dict) -> VLMResponse:
        """OpenAI 兼容 API 调用"""
        import requests
        b64 = self._encode_image(image)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt["user"]},
                ]},
            ],
            "max_tokens": self.max_tokens,
        }
        resp = requests.post(f"{self.api_base}/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        raw_output = resp.json()["choices"][0]["message"]["content"]
        return self._parse_response(raw_output, prompt)

    def _parse_response(self, raw_output: str, prompt: dict) -> VLMResponse:
        """解析 VLM 文本输出为结构化结果"""
        from src.annotation.response_parser import VLMResponseParser
        parser = VLMResponseParser()
        return parser.parse(raw_output, prompt.get("task_type", "detection"))


class GroundingDINOBackend(VLMBackend):
    """Grounding-DINO 后端

    支持两种调用方式：
      1. 本地 groundingdino 推理
      2. API 调用（如 Hugging Face Inference API）
    """

    def __init__(self, config: dict = None):
        cfg = (config or {}).get("vlm", {}).get("grounding_dino", {})
        self.mode = cfg.get("mode", "api")
        self.api_url = cfg.get("api_url", "http://localhost:7860/api/predict")
        self.model_path = cfg.get("model_path", "IDEA-Research/grounding-dino-base")
        self.box_threshold = cfg.get("box_threshold", 0.25)
        self.text_threshold = cfg.get("text_threshold", 0.20)
        self._model = None

    def _build_text_prompt(self, prompt: dict) -> str:
        """将目标类别列表转为 Grounding-DINO 文本 prompt"""
        classes = prompt.get("target_classes", [])
        if classes:
            return " . ".join(classes) + " ."
        return "pet . bag . phone . hand . obstruction . person ."

    def infer(self, image: np.ndarray, prompt: dict) -> VLMResponse:
        if self.mode == "local":
            return self._infer_local(image, prompt)
        return self._infer_api(image, prompt)

    def _infer_local(self, image: np.ndarray, prompt: dict) -> VLMResponse:
        """本地 groundingdino 推理"""
        from groundingdino.util.inference import load_model, predict
        import torch
        from PIL import Image as PILImage
        import groundingdino.datasets.transforms as T

        if self._model is None:
            self._model = load_model(
                "groundingdino/config/GroundingDINO_SwinB_cfg.py",
                self.model_path
            )

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        transformed, _ = transform(pil_img, None)

        text_prompt = self._build_text_prompt(prompt)
        boxes, logits, phrases = predict(
            model=self._model,
            image=transformed,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        h, w = image.shape[:2]
        detections = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            detections.append(VLMDetection(
                class_name=phrase.strip(),
                bbox=[max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
                confidence=round(float(logit), 4),
            ))

        return VLMResponse(
            detections=detections,
            raw_output=str({"boxes": boxes.tolist(), "phrases": phrases}),
            task_type="detection",
        )

    def _infer_api(self, image: np.ndarray, prompt: dict) -> VLMResponse:
        """API 调用"""
        import requests
        _, buf = cv2.imencode(".jpg", image)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        text_prompt = self._build_text_prompt(prompt)

        payload = {
            "image": b64,
            "text": text_prompt,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
        }
        resp = requests.post(self.api_url, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        h, w = image.shape[:2]
        detections = []
        for item in result.get("detections", []):
            bbox = item.get("bbox", [0, 0, 0, 0])
            # 如果是归一化坐标则转换
            if all(0 <= v <= 1.0 for v in bbox):
                bbox = [int(bbox[0] * w), int(bbox[1] * h),
                        int(bbox[2] * w), int(bbox[3] * h)]
            detections.append(VLMDetection(
                class_name=item.get("class", "unknown"),
                bbox=bbox,
                confidence=round(item.get("confidence", 0.5), 4),
            ))

        return VLMResponse(
            detections=detections,
            raw_output=json.dumps(result),
            task_type="detection",
        )


def create_vlm_backend(config: dict) -> VLMBackend:
    """工厂方法：根据配置创建 VLM 后端"""
    vlm_cfg = config.get("vlm", {})
    backend_type = vlm_cfg.get("backend", "simulated")

    if backend_type == "qwen":
        return QwenVLBackend(config)
    elif backend_type == "grounding_dino":
        return GroundingDINOBackend(config)
    else:
        return SimulatedVLMBackend(config)
