"""
视频推流模拟器 - 模拟座舱监控视频流

生成合成的座舱监控帧，包含正常场景和长尾/极端场景：
  - 后排宠物、遗留物品、奇异光照、摄像头被异物遮挡
"""

import random
import time
from dataclasses import dataclass, field
from typing import Generator

import cv2
import numpy as np


@dataclass
class SceneMetadata:
    """单帧的场景元数据"""
    frame_id: int
    timestamp: float
    scene_type: str            # "normal" | 长尾场景名
    objects: list = field(default_factory=list)  # [{"class": ..., "bbox": [x1,y1,x2,y2]}]


class CabinVideoSimulator:
    """座舱视频推流模拟器

    通过 OpenCV 在内存中合成帧，模拟车载摄像头采集的座舱画面。
    以一定概率注入长尾场景，用于下游不确定性采样的测试。
    """

    # 每种长尾场景的颜色标识和典型检测框
    SCENE_PALETTE = {
        "normal":           ((180, 180, 180), "N"),
        "pet_in_rear":      ((60, 180, 75),   "P"),
        "left_object":      ((50, 120, 220),  "O"),
        "extreme_lighting": ((0, 255, 255),   "L"),
        "camera_occluded":  ((0, 0, 180),     "X"),
    }

    def __init__(self, config: dict):
        sim_cfg = config.get("simulator", {})
        self.width = sim_cfg.get("width", 640)
        self.height = sim_cfg.get("height", 480)
        self.fps = sim_cfg.get("fps", 25)
        self.total_frames = sim_cfg.get("total_frames", 500)
        self.longtail_prob = sim_cfg.get("longtail_prob", 0.15)
        self.longtail_scenes = sim_cfg.get("longtail_scenes", [
            "pet_in_rear", "left_object", "extreme_lighting", "camera_occluded"
        ])
        self._frame_id = 0

    # ------------------------------------------------------------------
    # 帧生成
    # ------------------------------------------------------------------

    def _draw_cabin_background(self) -> np.ndarray:
        """绘制基础座舱背景（灰色座椅区域 + 仪表盘区域）"""
        frame = np.full((self.height, self.width, 3), (40, 40, 45), dtype=np.uint8)

        # 仪表盘区域（底部深色条）
        cv2.rectangle(frame, (0, self.height - 60), (self.width, self.height),
                      (25, 25, 30), -1)

        # 左右座椅轮廓
        seat_w, seat_h = self.width // 3, self.height // 2
        # 驾驶座
        cv2.rectangle(frame, (20, 80), (20 + seat_w, 80 + seat_h),
                      (60, 55, 50), -1)
        cv2.rectangle(frame, (20, 80), (20 + seat_w, 80 + seat_h),
                      (80, 75, 70), 2)
        # 副驾
        cv2.rectangle(frame, (self.width - 20 - seat_w, 80),
                      (self.width - 20, 80 + seat_h),
                      (60, 55, 50), -1)
        cv2.rectangle(frame, (self.width - 20 - seat_w, 80),
                      (self.width - 20, 80 + seat_h),
                      (80, 75, 70), 2)

        # 后排（上方区域）
        cv2.rectangle(frame, (10, 10), (self.width - 10, 75),
                      (50, 48, 45), -1)

        return frame

    def _inject_normal_scene(self, frame: np.ndarray) -> list:
        """注入正常场景元素（驾驶员 + 可选乘客）"""
        objects = []
        # 驾驶员头部（始终存在）
        cx, cy = 20 + self.width // 6, 120
        r = 25 + random.randint(-3, 3)
        cv2.circle(frame, (cx, cy), r, (140, 120, 100), -1)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        objects.append({"class": "driver", "bbox": bbox, "confidence": round(random.uniform(0.75, 0.98), 3)})

        # 50% 概率有副驾乘客
        if random.random() < 0.5:
            cx2 = self.width - 20 - self.width // 6
            cy2 = 120 + random.randint(-5, 5)
            r2 = 22 + random.randint(-3, 3)
            cv2.circle(frame, (cx2, cy2), r2, (130, 115, 95), -1)
            bbox2 = [cx2 - r2, cy2 - r2, cx2 + r2, cy2 + r2]
            objects.append({"class": "passenger", "bbox": bbox2, "confidence": round(random.uniform(0.70, 0.95), 3)})

        return objects

    def _inject_pet_in_rear(self, frame: np.ndarray) -> list:
        """后排宠物场景"""
        objects = self._inject_normal_scene(frame)
        # 在后排区域绘制小动物形状
        px = random.randint(60, self.width - 80)
        py = random.randint(15, 60)
        pw, ph = random.randint(30, 55), random.randint(20, 40)
        color = (60, 180, 75)
        cv2.ellipse(frame, (px, py), (pw // 2, ph // 2), 0, 0, 360, color, -1)
        # 耳朵
        cv2.circle(frame, (px - pw // 4, py - ph // 2), 6, color, -1)
        cv2.circle(frame, (px + pw // 4, py - ph // 2), 6, color, -1)
        bbox = [px - pw // 2, py - ph // 2 - 6, px + pw // 2, py + ph // 2]
        # 宠物检测置信度通常较低（长尾）
        objects.append({"class": "pet", "bbox": bbox, "confidence": round(random.uniform(0.10, 0.50), 3)})
        return objects

    def _inject_left_object(self, frame: np.ndarray) -> list:
        """遗留物品场景"""
        objects = self._inject_normal_scene(frame)
        ox = random.randint(self.width // 3, 2 * self.width // 3)
        oy = random.randint(self.height - 120, self.height - 70)
        ow, oh = random.randint(25, 50), random.randint(20, 40)
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (50, 120, 220), -1)
        cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (80, 150, 250), 2)
        bbox = [ox, oy, ox + ow, oy + oh]
        objects.append({"class": "left_object", "bbox": bbox, "confidence": round(random.uniform(0.15, 0.55), 3)})
        return objects

    def _inject_extreme_lighting(self, frame: np.ndarray) -> list:
        """奇异光照场景（过曝/欠曝）"""
        objects = self._inject_normal_scene(frame)
        if random.random() < 0.5:
            # 过曝 — 大面积高亮
            overlay = np.full_like(frame, (200, 200, 180), dtype=np.uint8)
            alpha = random.uniform(0.4, 0.7)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        else:
            # 欠曝 — 整体变暗
            frame[:] = (frame * random.uniform(0.15, 0.35)).astype(np.uint8)
        # 光照干扰导致所有检测置信度下降
        for obj in objects:
            obj["confidence"] = round(obj["confidence"] * random.uniform(0.3, 0.6), 3)
        return objects

    def _inject_camera_occluded(self, frame: np.ndarray) -> list:
        """摄像头被异物遮挡"""
        objects = self._inject_normal_scene(frame)
        # 随机覆盖一个大色块模拟遮挡
        ox = random.randint(0, self.width // 3)
        oy = random.randint(0, self.height // 3)
        ow = random.randint(self.width // 3, self.width * 2 // 3)
        oh = random.randint(self.height // 3, self.height * 2 // 3)
        # 半透明黑色/棕色遮挡
        overlay = frame.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + ow, oy + oh),
                      (15 + random.randint(0, 20), 10, 5), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        # 被遮挡区域内的目标置信度极低
        for obj in objects:
            bx1, by1, bx2, by2 = obj["bbox"]
            if bx1 < ox + ow and bx2 > ox and by1 < oy + oh and by2 > oy:
                obj["confidence"] = round(random.uniform(0.05, 0.25), 3)
        return objects

    SCENE_INJECTORS = {
        "pet_in_rear":      _inject_pet_in_rear,
        "left_object":      _inject_left_object,
        "extreme_lighting": _inject_extreme_lighting,
        "camera_occluded":  _inject_camera_occluded,
    }

    def _generate_frame(self) -> tuple[np.ndarray, SceneMetadata]:
        """生成单帧画面及其元数据"""
        frame = self._draw_cabin_background()

        # 决定是否注入长尾场景
        if random.random() < self.longtail_prob and self.longtail_scenes:
            scene_type = random.choice(self.longtail_scenes)
            injector = self.SCENE_INJECTORS.get(scene_type)
            if injector:
                objects = injector(self, frame)
            else:
                objects = self._inject_normal_scene(frame)
                scene_type = "normal"
        else:
            scene_type = "normal"
            objects = self._inject_normal_scene(frame)

        # 添加轻微随机噪声，模拟真实传感器
        noise = np.random.randint(-8, 9, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 添加帧信息水印
        info_text = f"Frame:{self._frame_id:05d}  Scene:{scene_type}"
        cv2.putText(frame, info_text, (10, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        meta = SceneMetadata(
            frame_id=self._frame_id,
            timestamp=time.time(),
            scene_type=scene_type,
            objects=objects,
        )
        self._frame_id += 1
        return frame, meta

    # ------------------------------------------------------------------
    # 推流接口
    # ------------------------------------------------------------------

    def stream(self) -> Generator[tuple[np.ndarray, SceneMetadata], None, None]:
        """
        生成器：按照配置的 FPS 逐帧产出 (frame, metadata)。

        用法::

            sim = CabinVideoSimulator(config)
            for frame, meta in sim.stream():
                process(frame, meta)
        """
        interval = 1.0 / self.fps
        count = 0
        while self.total_frames == 0 or count < self.total_frames:
            t0 = time.monotonic()
            frame, meta = self._generate_frame()
            yield frame, meta
            count += 1
            # 控制帧率
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def stream_fast(self) -> Generator[tuple[np.ndarray, SceneMetadata], None, None]:
        """不限帧率的快速推流（用于离线处理）"""
        count = 0
        while self.total_frames == 0 or count < self.total_frames:
            frame, meta = self._generate_frame()
            yield frame, meta
            count += 1
