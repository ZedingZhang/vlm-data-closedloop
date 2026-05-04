"""
难例库存储模块

负责将不确定性采样标记的帧保存到磁盘，包括：
  - 原始图像帧
  - 检测元数据（JSON）
  - 去重逻辑（同一目标短时间内不重复保存）
"""

import os
import json
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from src.sampling.uncertainty import UncertaintyFlag
from src.inference.engine import InferenceResult


class HardExampleStorage:
    """难例库管理器"""

    def __init__(self, config: dict):
        stor_cfg = config["storage"]
        self.output_dir = stor_cfg["output_dir"]
        self.log_dir = stor_cfg["log_dir"]
        self.save_format = stor_cfg["save_format"]
        self.save_metadata = stor_cfg["save_metadata"]
        self.dedup_interval = stor_cfg["dedup_interval"]

        # 去重记录：(class_name, reason) -> 上次保存的 frame_id
        self._last_saved: dict[tuple[str, str], int] = {}

        # 统计
        self.total_saved = 0
        self.total_skipped_dedup = 0

        # 初始化目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 按原因分类存储子目录
        for reason in ["low_confidence", "bbox_jitter", "combined"]:
            os.makedirs(os.path.join(self.output_dir, reason), exist_ok=True)

    def _should_save(self, flag: UncertaintyFlag) -> bool:
        """去重判断：同一类别同一原因在 dedup_interval 帧内不重复保存"""
        class_name = flag.details.get("class", "unknown")
        key = (class_name, flag.reason)
        last_frame = self._last_saved.get(key, -self.dedup_interval - 1)
        return (flag.frame_id - last_frame) >= self.dedup_interval

    def save(self, frame: np.ndarray, flag: UncertaintyFlag,
             inference_result: Optional[InferenceResult] = None) -> Optional[str]:
        """保存一个难例帧

        Args:
            frame: 原始图像帧
            flag: 不确定性标记
            inference_result: 对应的推理结果（可选，用于保存完整元数据）

        Returns:
            保存路径（如果被去重跳过则返回 None）
        """
        if not self._should_save(flag):
            self.total_skipped_dedup += 1
            return None

        # 更新去重记录
        class_name = flag.details.get("class", "unknown")
        key = (class_name, flag.reason)
        self._last_saved[key] = flag.frame_id

        # 构造文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"frame{flag.frame_id:06d}_{class_name}_{timestamp}"

        # 保存图像
        reason_dir = os.path.join(self.output_dir, flag.reason)
        img_path = os.path.join(reason_dir, f"{base_name}.jpg")
        cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 保存元数据
        if self.save_metadata:
            meta = {
                "frame_id": flag.frame_id,
                "reason": flag.reason,
                "uncertainty_score": flag.score,
                "flag_details": flag.details,
                "timestamp": timestamp,
            }
            if inference_result:
                meta["all_detections"] = [
                    {
                        "class": d.class_name,
                        "confidence": d.confidence,
                        "bbox": d.bbox,
                    }
                    for d in inference_result.detections
                ]
            meta_path = os.path.join(reason_dir, f"{base_name}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        self.total_saved += 1
        return img_path

    def get_statistics(self) -> dict:
        """返回存储统计信息"""
        # 统计各类别难例数量
        category_counts = {}
        for reason in ["low_confidence", "bbox_jitter", "combined"]:
            reason_dir = os.path.join(self.output_dir, reason)
            if os.path.isdir(reason_dir):
                img_count = len([f for f in os.listdir(reason_dir) if f.endswith(".jpg")])
                category_counts[reason] = img_count

        return {
            "total_saved": self.total_saved,
            "total_skipped_dedup": self.total_skipped_dedup,
            "category_counts": category_counts,
            "output_dir": self.output_dir,
        }
