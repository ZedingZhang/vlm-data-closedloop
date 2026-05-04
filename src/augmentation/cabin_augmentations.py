"""
座舱场景高级数据增强流水线

基于 Albumentations 构建，针对座舱长尾场景专门设计以下增强策略：
  1. 夜间噪点模拟 — ISO 噪声 + 低亮度 + 色温偏移
  2. 强光逆光模拟 — 局部过曝光晕 + 整体曝光偏移
  3. 局部阴影合成 — 多边形/条纹阴影叠加
  4. 摄像头退化模拟 — 模糊 + 色差 + 暗角
  5. 座舱通用增强  — 几何变换 + 颜色抖动
"""

import math
import random
from typing import Optional

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


# ====================================================================
# 自定义增强算子 — 座舱专用
# ====================================================================


class NightNoiseSimulation(ImageOnlyTransform):
    """夜间噪点模拟

    模拟高 ISO 条件下的真实传感器噪声：
      - 高斯亮度噪声（读出噪声）
      - 泊松散粒噪声（光子噪声）
      - 色度噪声（色彩通道独立偏移）
      - 整体亮度压低
    """

    class InitSchema(ImageOnlyTransform.InitSchema):
        brightness_range: tuple = (0.10, 0.40)
        gaussian_noise_range: tuple = (15, 50)
        color_noise_range: tuple = (5, 20)
        enable_poisson: bool = True
        color_temp_shift: tuple = (-25, 10)

    def __init__(
        self,
        brightness_range: tuple = (0.10, 0.40),
        gaussian_noise_range: tuple = (15, 50),
        color_noise_range: tuple = (5, 20),
        enable_poisson: bool = True,
        color_temp_shift: tuple = (-25, 10),
        p: float = 1.0,
        **kwargs,
    ):
        super().__init__(p=p, **kwargs)
        self.brightness_range = brightness_range
        self.gaussian_noise_range = gaussian_noise_range
        self.color_noise_range = color_noise_range
        self.enable_poisson = enable_poisson
        self.color_temp_shift = color_temp_shift

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        result = img.astype(np.float32)

        # 1) 整体亮度压低
        brightness_factor = random.uniform(*self.brightness_range)
        result *= brightness_factor

        # 2) 高斯读出噪声
        sigma = random.uniform(*self.gaussian_noise_range)
        gaussian = np.random.normal(0, sigma, result.shape).astype(np.float32)
        result += gaussian

        # 3) 泊松散粒噪声
        if self.enable_poisson:
            noisy = result.clip(0, 255)
            scale = max(1.0, noisy.max() / 30.0)
            poisson = np.random.poisson(np.maximum(noisy / scale, 0.1)).astype(np.float32) * scale
            result = result * 0.7 + poisson * 0.3

        # 4) 色度噪声 — 每个通道独立偏移
        color_sigma = random.uniform(*self.color_noise_range)
        for c in range(3):
            channel_noise = np.random.normal(0, color_sigma, result.shape[:2]).astype(np.float32)
            result[:, :, c] += channel_noise

        # 5) 色温偏移（夜间偏黄/偏蓝）
        temp_shift = random.uniform(*self.color_temp_shift)
        result[:, :, 0] += temp_shift * 0.6   # B 通道
        result[:, :, 2] -= temp_shift * 0.4   # R 通道

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("brightness_range", "gaussian_noise_range", "color_noise_range",
                "enable_poisson", "color_temp_shift")


class StrongBacklightGlare(ImageOnlyTransform):
    """强光/逆光模拟

    模拟阳光直射或对向车灯造成的局部过曝和光晕：
      - 随机位置放置一个或多个高亮光源
      - 径向渐变光晕扩散
      - 全局曝光偏移
    """

    class InitSchema(ImageOnlyTransform.InitSchema):
        num_sources: tuple = (1, 3)
        intensity_range: tuple = (0.6, 1.0)
        radius_ratio_range: tuple = (0.15, 0.50)
        global_exposure_shift: tuple = (0.0, 0.4)

    def __init__(
        self,
        num_sources: tuple = (1, 3),
        intensity_range: tuple = (0.6, 1.0),
        radius_ratio_range: tuple = (0.15, 0.50),
        global_exposure_shift: tuple = (0.0, 0.4),
        flare_color: Optional[tuple] = None,
        p: float = 1.0,
        **kwargs,
    ):
        super().__init__(p=p, **kwargs)
        self.num_sources = num_sources
        self.intensity_range = intensity_range
        self.radius_ratio_range = radius_ratio_range
        self.global_exposure_shift = global_exposure_shift
        self.flare_color = flare_color

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        result = img.astype(np.float32)
        h, w = result.shape[:2]
        diag = math.sqrt(h * h + w * w)

        n_sources = random.randint(*self.num_sources)

        for _ in range(n_sources):
            cx = random.randint(0, w)
            cy = random.randint(0, int(h * 0.6))

            intensity = random.uniform(*self.intensity_range)
            radius = int(diag * random.uniform(*self.radius_ratio_range))

            if self.flare_color:
                color = np.array(self.flare_color, dtype=np.float32)
            else:
                color = np.array([
                    random.uniform(200, 255),
                    random.uniform(220, 255),
                    random.uniform(230, 255),
                ], dtype=np.float32)

            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
            mask = np.clip(1.0 - dist / radius, 0, 1)
            mask = mask ** 2
            mask = mask[:, :, np.newaxis]

            flare = color[np.newaxis, np.newaxis, :] * mask * intensity * 255
            result = result + flare

        exposure = random.uniform(*self.global_exposure_shift)
        if exposure > 0:
            result = result * (1.0 + exposure)

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("num_sources", "intensity_range", "radius_ratio_range",
                "global_exposure_shift")


class LocalShadowSynthesis(ImageOnlyTransform):
    """局部阴影合成

    模拟座舱内由 A 柱、遮阳板、树影等造成的局部阴影：
      - 随机多边形阴影区域
      - 条纹状阴影（模拟百叶窗/树影）
      - 渐变边缘软化
    """

    class InitSchema(ImageOnlyTransform.InitSchema):
        shadow_type: str = "random"
        num_shadows: tuple = (1, 3)
        darkness_range: tuple = (0.25, 0.65)
        blur_kernel_range: tuple = (15, 45)
        stripe_width_range: tuple = (10, 40)
        stripe_angle_range: tuple = (-60, 60)

    def __init__(
        self,
        shadow_type: str = "random",
        num_shadows: tuple = (1, 3),
        darkness_range: tuple = (0.25, 0.65),
        blur_kernel_range: tuple = (15, 45),
        stripe_width_range: tuple = (10, 40),
        stripe_angle_range: tuple = (-60, 60),
        p: float = 1.0,
        **kwargs,
    ):
        super().__init__(p=p, **kwargs)
        self.shadow_type = shadow_type
        self.num_shadows = num_shadows
        self.darkness_range = darkness_range
        self.blur_kernel_range = blur_kernel_range
        self.stripe_width_range = stripe_width_range
        self.stripe_angle_range = stripe_angle_range

    def _create_polygon_shadow(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)
        n_vertices = random.randint(3, 6)
        points = []
        cx = random.randint(w // 6, 5 * w // 6)
        cy = random.randint(h // 6, 5 * h // 6)
        for _ in range(n_vertices):
            px = cx + random.randint(-w // 3, w // 3)
            py = cy + random.randint(-h // 3, h // 3)
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            points.append([px, py])
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1.0)
        return mask

    def _create_stripe_shadow(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)
        angle = random.uniform(*self.stripe_angle_range)
        stripe_w = random.randint(*self.stripe_width_range)
        gap = stripe_w + random.randint(stripe_w, stripe_w * 3)

        diag = int(math.sqrt(h * h + w * w))
        big_mask = np.zeros((diag * 2, diag * 2), dtype=np.float32)
        y = 0
        while y < diag * 2:
            cv2.rectangle(big_mask, (0, y), (diag * 2, y + stripe_w), 1.0, -1)
            y += gap

        center = (diag, diag)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        big_mask = cv2.warpAffine(big_mask, rot_mat, (diag * 2, diag * 2))

        ox = diag - w // 2
        oy = diag - h // 2
        mask = big_mask[oy:oy + h, ox:ox + w]
        return mask

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        result = img.astype(np.float32)
        h, w = result.shape[:2]
        n = random.randint(*self.num_shadows)

        for _ in range(n):
            st = self.shadow_type
            if st == "random":
                st = random.choice(["polygon", "stripe"])

            if st == "stripe":
                mask = self._create_stripe_shadow(h, w)
            else:
                mask = self._create_polygon_shadow(h, w)

            k = random.randrange(
                self.blur_kernel_range[0] | 1,
                self.blur_kernel_range[1] | 1,
                2
            )
            if k % 2 == 0:
                k += 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

            darkness = random.uniform(*self.darkness_range)
            shadow_factor = 1.0 - mask[:, :, np.newaxis] * darkness
            result = result * shadow_factor

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("shadow_type", "num_shadows", "darkness_range",
                "blur_kernel_range", "stripe_width_range", "stripe_angle_range")


class CameraDegradation(ImageOnlyTransform):
    """摄像头退化模拟

    模拟低质量车载摄像头的图像退化：
      - 暗角效果（边缘变暗）
      - 色差/色散
      - 运动模糊
    """

    class InitSchema(ImageOnlyTransform.InitSchema):
        vignette_strength: tuple = (0.3, 0.7)
        chromatic_shift: tuple = (1, 4)
        motion_blur_kernel: tuple = (3, 9)

    def __init__(
        self,
        vignette_strength: tuple = (0.3, 0.7),
        chromatic_shift: tuple = (1, 4),
        motion_blur_kernel: tuple = (3, 9),
        p: float = 1.0,
        **kwargs,
    ):
        super().__init__(p=p, **kwargs)
        self.vignette_strength = vignette_strength
        self.chromatic_shift = chromatic_shift
        self.motion_blur_kernel = motion_blur_kernel

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        result = img.copy()
        h, w = result.shape[:2]

        # 1) 暗角
        strength = random.uniform(*self.vignette_strength)
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        max_dist = math.sqrt(cx ** 2 + cy ** 2)
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        vignette = 1.0 - strength * (dist / max_dist) ** 2
        vignette = vignette[:, :, np.newaxis]
        result = (result.astype(np.float32) * vignette).clip(0, 255).astype(np.uint8)

        # 2) 色差 — 通道偏移
        shift = random.randint(*self.chromatic_shift)
        if shift > 0:
            M_r = np.float32([[1, 0, shift], [0, 1, 0]])
            M_b = np.float32([[1, 0, -shift], [0, 1, 0]])
            result[:, :, 2] = cv2.warpAffine(result[:, :, 2], M_r, (w, h))
            result[:, :, 0] = cv2.warpAffine(result[:, :, 0], M_b, (w, h))

        # 3) 轻微运动模糊
        k = random.randrange(self.motion_blur_kernel[0], self.motion_blur_kernel[1] + 1, 2)
        if k % 2 == 0:
            k += 1
        if k >= 3:
            angle = random.uniform(0, 180)
            M_blur = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1.0)
            kernel = np.zeros((k, k), dtype=np.float32)
            kernel[k // 2, :] = 1.0 / k
            kernel = cv2.warpAffine(kernel, M_blur, (k, k))
            kernel = kernel / (kernel.sum() + 1e-8)
            result = cv2.filter2D(result, -1, kernel)

        return result

    def get_transform_init_args_names(self):
        return ("vignette_strength", "chromatic_shift", "motion_blur_kernel")


# ====================================================================
# 增强流水线预设
# ====================================================================


def build_night_pipeline(p: float = 1.0) -> A.Compose:
    """夜间场景增强流水线"""
    return A.Compose([
        NightNoiseSimulation(
            brightness_range=(0.08, 0.35),
            gaussian_noise_range=(20, 55),
            color_noise_range=(8, 25),
            color_temp_shift=(-30, 15),
            p=1.0,
        ),
        CameraDegradation(
            vignette_strength=(0.4, 0.8),
            chromatic_shift=(1, 3),
            motion_blur_kernel=(3, 7),
            p=0.6,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.15, 0.05),
            contrast_limit=(-0.2, 0.1),
            p=0.5,
        ),
    ], p=p)


def build_backlight_pipeline(p: float = 1.0) -> A.Compose:
    """强光逆光增强流水线"""
    return A.Compose([
        StrongBacklightGlare(
            num_sources=(1, 3),
            intensity_range=(0.5, 1.0),
            radius_ratio_range=(0.15, 0.55),
            global_exposure_shift=(0.05, 0.5),
            p=1.0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.3),
            contrast_limit=(-0.3, 0.0),
            p=0.5,
        ),
        A.CLAHE(clip_limit=(1, 4), p=0.3),
    ], p=p)


def build_shadow_pipeline(p: float = 1.0) -> A.Compose:
    """局部阴影增强流水线"""
    return A.Compose([
        LocalShadowSynthesis(
            shadow_type="random",
            num_shadows=(1, 4),
            darkness_range=(0.30, 0.70),
            blur_kernel_range=(15, 55),
            p=1.0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=0.4,
        ),
    ], p=p)


def build_camera_degradation_pipeline(p: float = 1.0) -> A.Compose:
    """摄像头退化增强流水线"""
    return A.Compose([
        CameraDegradation(
            vignette_strength=(0.3, 0.7),
            chromatic_shift=(2, 5),
            motion_blur_kernel=(3, 11),
            p=1.0,
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.5),
        A.ImageCompression(quality_range=(30, 70), p=0.4),
    ], p=p)


def build_full_cabin_pipeline(p: float = 1.0) -> A.Compose:
    """座舱综合增强流水线（随机组合所有策略）"""
    return A.Compose([
        # 几何变换（保守，座舱视角变化小）
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.92, 1.08),
            rotate=(-5, 5),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        ),
        A.HorizontalFlip(p=0.3),

        # 颜色抖动
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0
            ),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                          saturation=(0.8, 1.2), hue=(-0.1, 0.1), p=1.0),
        ], p=0.5),

        # 座舱专用增强（互斥选一）
        A.OneOf([
            NightNoiseSimulation(
                brightness_range=(0.10, 0.40),
                gaussian_noise_range=(15, 45),
                p=1.0,
            ),
            StrongBacklightGlare(
                num_sources=(1, 2),
                intensity_range=(0.4, 0.9),
                p=1.0,
            ),
            LocalShadowSynthesis(
                shadow_type="random",
                num_shadows=(1, 3),
                darkness_range=(0.25, 0.60),
                p=1.0,
            ),
            CameraDegradation(
                vignette_strength=(0.3, 0.6),
                p=1.0,
            ),
        ], p=0.7),

        # 通用噪声/模糊
        A.OneOf([
            A.GaussNoise(std_range=(0.04, 0.20), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),

    ], p=p)


def build_cabin_pipeline_with_bboxes(p: float = 1.0) -> A.Compose:
    """带 bbox 标注的座舱增强流水线（YOLO 训练用）

    保留所有增强效果，同时正确变换 bounding box 坐标。
    仅几何变换需要 bbox 感知，像素级变换自动兼容。
    """
    return A.Compose([
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.92, 1.08),
            rotate=(-5, 5),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        ),
        A.HorizontalFlip(p=0.3),

        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0
            ),
            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2),
                          saturation=(0.8, 1.2), hue=(-0.1, 0.1), p=1.0),
        ], p=0.5),

        A.OneOf([
            NightNoiseSimulation(p=1.0),
            StrongBacklightGlare(p=1.0),
            LocalShadowSynthesis(p=1.0),
            CameraDegradation(p=1.0),
        ], p=0.7),

        A.OneOf([
            A.GaussNoise(std_range=(0.04, 0.20), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),

    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ), p=p)


# ====================================================================
# 便捷接口
# ====================================================================

PIPELINE_REGISTRY = {
    "night":             build_night_pipeline,
    "backlight":         build_backlight_pipeline,
    "shadow":            build_shadow_pipeline,
    "camera_degradation": build_camera_degradation_pipeline,
    "full":              build_full_cabin_pipeline,
    "full_with_bboxes":  build_cabin_pipeline_with_bboxes,
}


def get_pipeline(name: str, p: float = 1.0) -> A.Compose:
    """按名称获取增强流水线"""
    if name not in PIPELINE_REGISTRY:
        raise KeyError(f"未知流水线: {name}, 可用: {list(PIPELINE_REGISTRY.keys())}")
    return PIPELINE_REGISTRY[name](p=p)


def list_pipelines() -> list[str]:
    """列出所有可用的增强流水线名称"""
    return list(PIPELINE_REGISTRY.keys())
