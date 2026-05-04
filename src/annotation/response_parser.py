"""
VLM 响应解析器

将 VLM 返回的自由文本/JSON 解析为结构化的检测结果。
处理各种不规范输出：markdown 代码块、多余文字、坐标格式差异等。
"""

import json
import re
from typing import Optional

from src.annotation.vlm_backends import VLMDetection, VLMResponse


class VLMResponseParser:
    """VLM 响应文本解析器"""

    def parse(self, raw_text: str, task_type: str = "detection") -> VLMResponse:
        """解析 VLM 原始输出

        尝试多种解析策略，按优先级：
          1. 直接 JSON 数组
          2. markdown 代码块中的 JSON
          3. 正则提取 bbox 模式
        """
        raw_text = raw_text.strip()
        detections = []

        # 策略 1: 提取 JSON
        json_data = self._extract_json(raw_text)
        if json_data is not None:
            if isinstance(json_data, list):
                detections = self._parse_detection_list(json_data)
            elif isinstance(json_data, dict):
                detections = self._parse_detection_dict(json_data)

        # 策略 2: 正则匹配 bbox
        if not detections:
            detections = self._regex_extract(raw_text)

        return VLMResponse(
            detections=detections,
            raw_output=raw_text,
            task_type=task_type,
        )

    def _extract_json(self, text: str) -> Optional[any]:
        """从文本中提取 JSON 内容"""
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 从 markdown 代码块中提取
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # 查找第一个 [ 或 { 到最后一个 ] 或 }
        for start_char, end_char in [('[', ']'), ('{', '}')]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue

        return None

    def _parse_detection_list(self, items: list) -> list[VLMDetection]:
        """解析检测结果列表"""
        detections = []
        for item in items:
            if not isinstance(item, dict):
                continue
            det = self._parse_single_detection(item)
            if det:
                detections.append(det)
        return detections

    def _parse_detection_dict(self, data: dict) -> list[VLMDetection]:
        """解析单个字典或嵌套结构"""
        # 可能是单个检测
        det = self._parse_single_detection(data)
        if det:
            return [det]

        # 可能是包含 detections/results 键的字典
        for key in ["detections", "results", "objects", "predictions"]:
            if key in data and isinstance(data[key], list):
                return self._parse_detection_list(data[key])

        return []

    def _parse_single_detection(self, item: dict) -> Optional[VLMDetection]:
        """解析单个检测项，兼容多种字段命名"""
        # 类名
        class_name = (item.get("class") or item.get("class_name") or
                      item.get("label") or item.get("category") or "unknown")

        # 边界框 — 支持多种格式
        bbox = (item.get("bbox") or item.get("bounding_box") or
                item.get("box") or item.get("region"))

        if bbox is None:
            # 尝试从独立字段拼装
            keys_sets = [
                ("x_min", "y_min", "x_max", "y_max"),
                ("xmin", "ymin", "xmax", "ymax"),
                ("x1", "y1", "x2", "y2"),
                ("left", "top", "right", "bottom"),
            ]
            for k1, k2, k3, k4 in keys_sets:
                if all(k in item for k in (k1, k2, k3, k4)):
                    bbox = [item[k1], item[k2], item[k3], item[k4]]
                    break

        if bbox is None or len(bbox) < 4:
            return None

        bbox = [int(round(v)) if isinstance(v, float) else int(v) for v in bbox[:4]]

        # 置信度
        confidence = float(item.get("confidence") or item.get("score") or
                           item.get("prob") or 0.5)

        return VLMDetection(
            class_name=str(class_name),
            bbox=bbox,
            confidence=round(confidence, 4),
            raw_text=json.dumps(item, ensure_ascii=False),
        )

    def _regex_extract(self, text: str) -> list[VLMDetection]:
        """最后手段：正则提取坐标"""
        detections = []
        # 匹配形如 class: xxx, bbox: [x1, y1, x2, y2] 的模式
        pattern = (
            r'(?:class|label|object)\s*[:=]\s*["\']?(\w+)["\']?\s*'
            r'.*?(?:bbox|box|region)\s*[:=]\s*\[?\s*'
            r'(\d+)\s*[,\s]+(\d+)\s*[,\s]+(\d+)\s*[,\s]+(\d+)'
        )
        for m in re.finditer(pattern, text, re.IGNORECASE):
            detections.append(VLMDetection(
                class_name=m.group(1),
                bbox=[int(m.group(2)), int(m.group(3)),
                      int(m.group(4)), int(m.group(5))],
                confidence=0.5,
                raw_text=m.group(0),
            ))

        # 匹配 <box> 标签格式（Qwen-VL 常见）
        box_pattern = r'<box>\s*\((\d+),\s*(\d+)\)\s*,\s*\((\d+),\s*(\d+)\)\s*</box>'
        for m in re.finditer(box_pattern, text):
            detections.append(VLMDetection(
                class_name="object",
                bbox=[int(m.group(1)), int(m.group(2)),
                      int(m.group(3)), int(m.group(4))],
                confidence=0.5,
                raw_text=m.group(0),
            ))

        # 匹配 <ref>xxx</ref><box>(x1,y1),(x2,y2)</box> 格式
        ref_box_pattern = (
            r'<ref>\s*(.*?)\s*</ref>\s*'
            r'<box>\s*\((\d+),\s*(\d+)\)\s*,\s*\((\d+),\s*(\d+)\)\s*</box>'
        )
        for m in re.finditer(ref_box_pattern, text):
            detections.append(VLMDetection(
                class_name=m.group(1),
                bbox=[int(m.group(2)), int(m.group(3)),
                      int(m.group(4)), int(m.group(5))],
                confidence=0.5,
                raw_text=m.group(0),
            ))

        return detections
