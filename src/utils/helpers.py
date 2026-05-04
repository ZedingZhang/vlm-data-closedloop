"""
工具函数集
"""

import os
import json
import logging
from copy import deepcopy
from datetime import datetime

import yaml


# ====================================================================
# 默认配置（唯一真相来源，与 configs/pipeline_config.yaml 保持同步）
# ====================================================================

DEFAULT_CONFIG = {
    "simulator": {
        "width": 640,
        "height": 480,
        "fps": 25,
        "total_frames": 500,
        "longtail_prob": 0.15,
        "longtail_scenes": [
            "pet_in_rear", "left_object", "extreme_lighting", "camera_occluded"
        ],
    },
    "inference": {
        "model_type": "simulated",
        "yolo_weights": "yolov8n.pt",
        "conf_threshold": 0.25,
        "device": "cpu",
        "classes": [
            "driver", "passenger", "child_seat", "pet",
            "left_object", "phone_usage", "smoking", "seatbelt_off", "normal"
        ],
    },
    "sampling": {
        "confidence_threshold": 0.4,
        "bbox_jitter_iou_threshold": 0.5,
        "jitter_window_size": 5,
        "jitter_ratio_threshold": 0.6,
        "fusion_strategy": "any",
        "history_max_age": 500,
    },
    "storage": {
        "output_dir": "data/hard_examples",
        "log_dir": "data/logs",
        "save_format": "image",
        "save_metadata": True,
        "dedup_interval": 10,
    },
    "vlm": {
        "backend": "simulated",
        "qwen": {
            "mode": "api",
            "api_base": "http://localhost:8000/v1",
            "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "max_tokens": 1024,
            "max_retries": 3,
            "retry_delay": 2.0,
        },
        "grounding_dino": {
            "mode": "api",
            "api_url": "http://localhost:7860/api/predict",
            "model_path": "IDEA-Research/grounding-dino-base",
            "box_threshold": 0.25,
            "text_threshold": 0.20,
            "max_retries": 3,
            "retry_delay": 2.0,
        },
    },
    "annotation": {
        "output_format": "both",
        "output_dir": "data/annotations",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并：override 中的值覆盖 base 中的对应值，嵌套字典递归合并"""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件，与默认配置深度合并后返回"""
    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_CONFIG, user_config)


def setup_logger(name: str, log_dir: str, level=logging.INFO) -> logging.Logger:
    """创建带文件和控制台输出的 logger"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # 控制台
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # 文件
        log_file = os.path.join(log_dir, f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def save_json(data: dict, path: str):
    """保存 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_iou(box_a: list, box_b: list) -> float:
    """计算两个 bbox 的 IoU (Intersection over Union)

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        IoU 值 (0~1)
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
