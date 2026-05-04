"""
Prompt 模板工厂

为不同的座舱场景和检测任务生成结构化的 VLM Prompt。
支持多种提问模式：开放检测、定向检测、框选确认、场景描述。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplate:
    """单条 Prompt 模板"""
    name: str
    task_type: str          # "detection" | "classification" | "description" | "verification"
    system_prompt: str
    user_prompt: str
    target_classes: list    # 期望检测的目标类别


class PromptFactory:
    """Prompt 模板工厂

    根据难例类型和标注需求，自动选择或组合 Prompt 模板。
    """

    # ----------------------------------------------------------------
    # 预定义模板库
    # ----------------------------------------------------------------

    TEMPLATES = {
        # ---- 通用开放检测 ----
        "open_detection": PromptTemplate(
            name="open_detection",
            task_type="detection",
            system_prompt=(
                "You are a visual perception expert for autonomous driving cabin monitoring. "
                "Analyze the given cabin camera image and detect all objects of interest. "
                "Return results as a JSON list."
            ),
            user_prompt=(
                "Detect and output the bounding boxes of any pets, unusual objects, "
                "or hands obscuring the camera in this cabin image. "
                "For each detection, return a JSON object with fields: "
                '"class" (string), "bbox" [x_min, y_min, x_max, y_max] in pixels, '
                '"confidence" (0-1). Return a JSON array of all detections.'
            ),
            target_classes=["pet", "left_object", "hand", "obstruction"],
        ),

        # ---- 宠物专项检测 ----
        "pet_detection": PromptTemplate(
            name="pet_detection",
            task_type="detection",
            system_prompt=(
                "You are an expert at detecting animals inside vehicle cabins. "
                "Focus on identifying any pets (dogs, cats, birds, etc.) in the image."
            ),
            user_prompt=(
                "Carefully examine this cabin camera image. Detect any pets or animals "
                "present in the vehicle. For each animal found, return: "
                '"class" (specific animal type like "dog", "cat"), '
                '"bbox" [x_min, y_min, x_max, y_max] in pixels, '
                '"confidence" (0-1). Return a JSON array.'
            ),
            target_classes=["dog", "cat", "bird", "pet"],
        ),

        # ---- 遗留物品检测 ----
        "left_object_detection": PromptTemplate(
            name="left_object_detection",
            task_type="detection",
            system_prompt=(
                "You are an expert at identifying objects left behind in vehicles. "
                "Focus on bags, phones, wallets, packages, and other personal items."
            ),
            user_prompt=(
                "Examine this cabin image and detect any objects that may have been "
                "left behind by passengers. Common items include bags, phones, wallets, "
                "keys, packages, food containers, and umbrellas. "
                "For each item, return: "
                '"class" (specific object type), '
                '"bbox" [x_min, y_min, x_max, y_max] in pixels, '
                '"confidence" (0-1). Return a JSON array.'
            ),
            target_classes=["bag", "phone", "wallet", "package", "keys", "left_object"],
        ),

        # ---- 摄像头遮挡检测 ----
        "occlusion_detection": PromptTemplate(
            name="occlusion_detection",
            task_type="detection",
            system_prompt=(
                "You are an expert at detecting camera obstructions and occlusions. "
                "Identify any objects or hands blocking the camera view."
            ),
            user_prompt=(
                "Analyze this cabin camera image for any obstructions. Detect hands, "
                "stickers, dirt, or any objects that are partially or fully blocking "
                "the camera view. For each obstruction found, return: "
                '"class" (type of obstruction), '
                '"bbox" [x_min, y_min, x_max, y_max] in pixels, '
                '"confidence" (0-1). Return a JSON array.'
            ),
            target_classes=["hand", "sticker", "dirt", "obstruction"],
        ),

        # ---- 光照异常检测 ----
        "lighting_anomaly": PromptTemplate(
            name="lighting_anomaly",
            task_type="classification",
            system_prompt=(
                "You are an expert at analyzing image quality and lighting conditions."
            ),
            user_prompt=(
                "Analyze the lighting conditions in this cabin camera image. "
                "Determine if the lighting is: normal, overexposed, underexposed, "
                "or has glare/flare artifacts. "
                'Return a JSON object with: "lighting_condition" (string), '
                '"severity" (0-1), "affected_region" [x_min, y_min, x_max, y_max] '
                "covering the most affected area, and "
                '"description" (brief explanation).'
            ),
            target_classes=["overexposed", "underexposed", "glare", "normal"],
        ),

        # ---- 驾驶员行为检测 ----
        "driver_behavior": PromptTemplate(
            name="driver_behavior",
            task_type="detection",
            system_prompt=(
                "You are a driver monitoring system expert. Detect unsafe driver "
                "behaviors from cabin camera images."
            ),
            user_prompt=(
                "Examine this cabin image and detect any of the following driver "
                "behaviors: phone usage, smoking, drowsiness, seatbelt not worn, "
                "distracted driving. For each detected behavior, return: "
                '"class" (behavior type), '
                '"bbox" [x_min, y_min, x_max, y_max] around the relevant region, '
                '"confidence" (0-1). Return a JSON array.'
            ),
            target_classes=["phone_usage", "smoking", "drowsy", "seatbelt_off", "distracted"],
        ),

        # ---- 场景全局描述 ----
        "scene_description": PromptTemplate(
            name="scene_description",
            task_type="description",
            system_prompt=(
                "You are a cabin scene analyst for autonomous driving systems."
            ),
            user_prompt=(
                "Describe the scene in this cabin camera image in detail. Include: "
                "1) Number and position of occupants, "
                "2) Any unusual objects or situations, "
                "3) Lighting conditions, "
                "4) Camera view quality. "
                'Return a JSON object with: "occupant_count" (int), '
                '"description" (string), "anomalies" (list of strings), '
                '"risk_level" ("low"/"medium"/"high").'
            ),
            target_classes=[],
        ),

        # ---- 标注验证/确认 ----
        "verification": PromptTemplate(
            name="verification",
            task_type="verification",
            system_prompt=(
                "You are a quality assurance expert for object detection annotations."
            ),
            user_prompt=(
                "Given the following pre-existing detection in this cabin image: "
                "class={class_name}, bbox={bbox}. "
                "Please verify: 1) Is the detected object correctly classified? "
                "2) Is the bounding box accurate? "
                'Return a JSON object with: "correct_class" (bool), '
                '"correct_bbox" (bool), "suggested_class" (string if reclassified), '
                '"suggested_bbox" [x_min, y_min, x_max, y_max] if adjustment needed, '
                '"confidence" (0-1).'
            ),
            target_classes=[],
        ),
    }

    # ----------------------------------------------------------------
    # 难例类型 → 模板映射
    # ----------------------------------------------------------------

    SCENE_TO_TEMPLATES = {
        "pet_in_rear":      ["pet_detection", "open_detection"],
        "left_object":      ["left_object_detection", "open_detection"],
        "camera_occluded":  ["occlusion_detection", "open_detection"],
        "extreme_lighting": ["lighting_anomaly", "open_detection"],
        "normal":           ["open_detection", "driver_behavior"],
    }

    def __init__(self, custom_templates: Optional[dict] = None):
        self.templates = dict(self.TEMPLATES)
        if custom_templates:
            for name, tmpl in custom_templates.items():
                self.templates[name] = tmpl

    def get_template(self, name: str) -> PromptTemplate:
        """按名称获取模板"""
        if name not in self.templates:
            raise KeyError(f"未知模板: {name}, 可用: {list(self.templates.keys())}")
        return self.templates[name]

    def get_templates_for_scene(self, scene_type: str) -> list[PromptTemplate]:
        """根据场景类型获取推荐的模板列表"""
        names = self.SCENE_TO_TEMPLATES.get(scene_type, ["open_detection"])
        return [self.templates[n] for n in names if n in self.templates]

    def build_prompt(self, template_name: str, image_width: int = 0,
                     image_height: int = 0, **kwargs) -> dict:
        """构建完整的 Prompt 消息（适配多数 VLM API 的消息格式）

        Args:
            template_name: 模板名称
            image_width: 图像宽度（用于坐标说明）
            image_height: 图像高度
            **kwargs: 用于模板中变量替换的参数

        Returns:
            {"system": str, "user": str, "target_classes": list}
        """
        tmpl = self.get_template(template_name)
        user_prompt = tmpl.user_prompt
        # 变量替换（如 verification 模板中的 {class_name}, {bbox}）
        for key, val in kwargs.items():
            user_prompt = user_prompt.replace(f"{{{key}}}", str(val))

        # 追加图像尺寸信息
        if image_width > 0 and image_height > 0:
            user_prompt += (
                f"\n\nNote: Image dimensions are {image_width}x{image_height} pixels. "
                "All bounding box coordinates should be in absolute pixel values."
            )

        return {
            "system": tmpl.system_prompt,
            "user": user_prompt,
            "target_classes": tmpl.target_classes,
            "task_type": tmpl.task_type,
        }

    def list_templates(self) -> list[str]:
        """列出所有可用模板名称"""
        return list(self.templates.keys())
