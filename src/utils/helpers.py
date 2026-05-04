"""
工具函数集
"""

import os
import json
import logging
from datetime import datetime

import yaml


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
