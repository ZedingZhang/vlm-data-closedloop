"""
不确定性采样算法模块

实现两种核心的不确定性度量：
  1. 低置信度采样 — 检测结果中分类置信度极低的帧
  2. 检测框抖动采样 — 同一目标在连续帧中检测框剧烈抖动（IoU 异常低）
"""

from collections import defaultdict
from dataclasses import dataclass, field

from src.inference.engine import InferenceResult, Detection
from src.utils.helpers import compute_iou


@dataclass
class UncertaintyFlag:
    """不确定性标记"""
    frame_id: int
    reason: str                     # "low_confidence" | "bbox_jitter" | "combined"
    score: float                    # 不确定性分数 (0~1, 越高越不确定)
    details: dict = field(default_factory=dict)


class UncertaintySampler:
    """不确定性采样器

    维护一个滑动窗口，对每帧推理结果做两维度的不确定性评估。
    """

    def __init__(self, config: dict):
        samp_cfg = config.get("sampling", {})
        self.conf_threshold = samp_cfg.get("confidence_threshold", 0.4)
        self.iou_threshold = samp_cfg.get("bbox_jitter_iou_threshold", 0.5)
        self.window_size = samp_cfg.get("jitter_window_size", 5)
        self.jitter_ratio = samp_cfg.get("jitter_ratio_threshold", 0.6)
        self.fusion = samp_cfg.get("fusion_strategy", "any")
        # 历史最大保留帧数：超过此范围未出现的类别将被清理
        self.history_max_age = samp_cfg.get("history_max_age", 500)

        # 滑动窗口缓存：class_name -> deque of (frame_id, bbox)
        self._history: dict[str, list[tuple[int, list]]] = defaultdict(list)
        self._last_prune_frame = 0

    # ------------------------------------------------------------------
    # 低置信度评估
    # ------------------------------------------------------------------

    def _check_low_confidence(self, result: InferenceResult) -> list[UncertaintyFlag]:
        """检查当前帧中是否存在低置信度检测"""
        flags = []
        for det in result.detections:
            if det.confidence < self.conf_threshold:
                flags.append(UncertaintyFlag(
                    frame_id=result.frame_id,
                    reason="low_confidence",
                    score=round(1.0 - det.confidence, 4),
                    details={
                        "class": det.class_name,
                        "confidence": det.confidence,
                        "bbox": det.bbox,
                        "threshold": self.conf_threshold,
                    }
                ))
        return flags

    # ------------------------------------------------------------------
    # 检测框抖动评估
    # ------------------------------------------------------------------

    def _update_history(self, result: InferenceResult):
        """将当前帧的检测结果加入滑动窗口"""
        for det in result.detections:
            history = self._history[det.class_name]
            history.append((result.frame_id, det.bbox))
            # 保留窗口大小
            if len(history) > self.window_size:
                history.pop(0)

    def _prune_history(self, current_frame_id: int):
        """清理长期未出现的类别记录，防止内存无限增长

        每 history_max_age 帧执行一次，移除最近 history_max_age 帧内
        没有新检测的类别条目。
        """
        stale = [
            cls_name
            for cls_name, entries in self._history.items()
            if not entries or current_frame_id - entries[-1][0] > self.history_max_age
        ]
        for cls_name in stale:
            del self._history[cls_name]

    def _check_bbox_jitter(self, result: InferenceResult) -> list[UncertaintyFlag]:
        """检查检测框在滑动窗口内是否发生剧烈抖动

        算法：
        对于当前帧中的每个检测目标，取其类别的历史检测框序列，
        计算相邻帧之间的 IoU。如果窗口内 IoU < 阈值的帧对比例
        超过 jitter_ratio_threshold，则判定为抖动。
        """
        flags = []
        seen_classes = set()

        for det in result.detections:
            if det.class_name in seen_classes:
                continue
            seen_classes.add(det.class_name)

            history = self._history.get(det.class_name, [])
            if len(history) < 2:
                continue

            # 计算相邻帧之间的 IoU
            iou_values = []
            low_iou_count = 0
            for i in range(1, len(history)):
                _, prev_box = history[i - 1]
                _, curr_box = history[i]
                iou = compute_iou(prev_box, curr_box)
                iou_values.append(iou)
                if iou < self.iou_threshold:
                    low_iou_count += 1

            pair_count = len(iou_values)
            if pair_count == 0:
                continue

            jitter_ratio = low_iou_count / pair_count

            if jitter_ratio >= self.jitter_ratio:
                avg_iou = sum(iou_values) / pair_count

                flags.append(UncertaintyFlag(
                    frame_id=result.frame_id,
                    reason="bbox_jitter",
                    score=round(1.0 - avg_iou, 4),
                    details={
                        "class": det.class_name,
                        "jitter_ratio": round(jitter_ratio, 4),
                        "avg_iou": round(avg_iou, 4),
                        "window_size": len(history),
                        "iou_threshold": self.iou_threshold,
                    }
                ))

        return flags

    # ------------------------------------------------------------------
    # 综合评估
    # ------------------------------------------------------------------

    def evaluate(self, result: InferenceResult) -> list[UncertaintyFlag]:
        """对单帧推理结果进行不确定性评估

        Args:
            result: 推理结果

        Returns:
            不确定性标记列表（空列表 = 当前帧不是难例）
        """
        # 先更新历史，再做抖动检测
        self._update_history(result)

        # 定期清理过期的历史记录（每 history_max_age 帧一次）
        if result.frame_id - self._last_prune_frame >= self.history_max_age:
            self._prune_history(result.frame_id)
            self._last_prune_frame = result.frame_id

        conf_flags = self._check_low_confidence(result)
        jitter_flags = self._check_bbox_jitter(result)

        if self.fusion == "all":
            # 两种条件都满足才标记
            conf_classes = {f.details["class"] for f in conf_flags}
            jitter_classes = {f.details["class"] for f in jitter_flags}
            common = conf_classes & jitter_classes
            if common:
                # 合并为 combined 标记
                combined = []
                for f in conf_flags:
                    if f.details["class"] in common:
                        f.reason = "combined"
                        # 查找对应的 jitter 信息
                        for jf in jitter_flags:
                            if jf.details["class"] == f.details["class"]:
                                f.details["jitter_ratio"] = jf.details.get("jitter_ratio")
                                f.details["avg_iou"] = jf.details.get("avg_iou")
                                f.score = round((f.score + jf.score) / 2, 4)
                                break
                        combined.append(f)
                return combined
            return []
        else:
            # "any": 任一条件满足即标记
            all_flags = conf_flags + jitter_flags
            return all_flags

    def get_statistics(self) -> dict:
        """返回当前采样器的统计信息"""
        return {
            "tracked_classes": list(self._history.keys()),
            "history_lengths": {k: len(v) for k, v in self._history.items()},
        }
