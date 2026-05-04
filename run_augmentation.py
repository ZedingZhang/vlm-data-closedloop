"""
数据增强批处理脚本

对难例库或标注数据集批量应用增强，生成扩充后的训练样本。
同时处理图像和对应的 YOLO 标注文件。

用法:
    python run_augmentation.py                                    # 默认配置
    python run_augmentation.py --pipeline night --multiplier 3    # 夜间增强×3
    python run_augmentation.py --pipeline full --input data/annotations/yolo/images
    python run_augmentation.py --all-pipelines                    # 每种风格各生成一份
"""

import os
import sys
import glob
import argparse
import shutil
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.augmentation.cabin_augmentations import (
    get_pipeline, list_pipelines, build_cabin_pipeline_with_bboxes,
)
from src.utils.helpers import load_config, setup_logger


def read_yolo_labels(label_path: str) -> tuple[list, list]:
    """读取 YOLO 标注文件

    Returns:
        (bboxes, class_labels): bboxes 为 [[xc, yc, w, h], ...] 归一化坐标
    """
    bboxes = []
    class_labels = []
    if not os.path.isfile(label_path):
        return bboxes, class_labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                bboxes.append([xc, yc, w, h])
                class_labels.append(cls_id)
    return bboxes, class_labels


def write_yolo_labels(label_path: str, bboxes: list, class_labels: list):
    """写入 YOLO 标注文件"""
    with open(label_path, "w") as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def run_augmentation(
    input_dir: str,
    output_dir: str,
    pipeline_name: str = "full",
    multiplier: int = 3,
    with_labels: bool = True,
    config_path: str = None,
):
    """执行批量数据增强

    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录
        pipeline_name: 增强流水线名称
        multiplier: 每张图像生成的增强副本数
        with_labels: 是否同时处理 YOLO 标注
        config_path: 配置文件路径
    """
    config = {}
    if config_path and os.path.isfile(config_path):
        config = load_config(config_path)

    log_dir = config.get("storage", {}).get("log_dir", "data/logs")
    logger = setup_logger("augmentation", log_dir)

    logger.info("=" * 60)
    logger.info("数据增强流水线启动")
    logger.info("=" * 60)
    logger.info(f"输入: {os.path.abspath(input_dir)}")
    logger.info(f"输出: {os.path.abspath(output_dir)}")
    logger.info(f"流水线: {pipeline_name}")
    logger.info(f"增强倍数: {multiplier}")
    logger.info(f"处理标注: {with_labels}")

    # 收集图像
    exts = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths.sort()

    if not image_paths:
        logger.warning(f"输入目录中未找到图像: {input_dir}")
        return

    logger.info(f"待处理图像: {len(image_paths)}")

    # 推断 labels 目录
    labels_dir = None
    if with_labels:
        # 常见结构: .../images/ 对应 .../labels/
        parent = os.path.dirname(input_dir.rstrip("/"))
        candidate = os.path.join(parent, "labels")
        if os.path.isdir(candidate):
            labels_dir = candidate
            logger.info(f"标注目录: {labels_dir}")
        else:
            logger.info("未找到标注目录，仅增强图像")
            with_labels = False

    # 创建输出目录
    out_images = os.path.join(output_dir, "images")
    out_labels = os.path.join(output_dir, "labels")
    os.makedirs(out_images, exist_ok=True)
    if with_labels:
        os.makedirs(out_labels, exist_ok=True)

    # 构建增强流水线
    if with_labels and labels_dir:
        pipeline = build_cabin_pipeline_with_bboxes()
        logger.info("使用 bbox 感知增强流水线")
    else:
        pipeline = get_pipeline(pipeline_name)

    logger.info("-" * 60)

    total_generated = 0
    t_start = time.time()

    for idx, img_path in enumerate(image_paths):
        fname = os.path.basename(img_path)
        name_no_ext = os.path.splitext(fname)[0]
        ext = os.path.splitext(fname)[1]

        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"无法读取: {img_path}")
            continue

        # 读取标注
        bboxes, class_labels = [], []
        if with_labels and labels_dir:
            label_path = os.path.join(labels_dir, name_no_ext + ".txt")
            bboxes, class_labels = read_yolo_labels(label_path)

        # 复制原图
        shutil.copy2(img_path, os.path.join(out_images, fname))
        if with_labels and labels_dir:
            src_label = os.path.join(labels_dir, name_no_ext + ".txt")
            if os.path.isfile(src_label):
                shutil.copy2(src_label, os.path.join(out_labels, name_no_ext + ".txt"))
        total_generated += 1

        # 生成增强副本
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for aug_idx in range(multiplier):
            try:
                if with_labels and bboxes:
                    transformed = pipeline(
                        image=image_rgb,
                        bboxes=bboxes,
                        class_labels=class_labels,
                    )
                    aug_image = transformed["image"]
                    aug_bboxes = transformed["bboxes"]
                    aug_labels = transformed["class_labels"]
                else:
                    transformed = pipeline(image=image_rgb)
                    aug_image = transformed["image"]
                    aug_bboxes, aug_labels = bboxes, class_labels

                # 保存增强图像
                aug_name = f"{name_no_ext}_aug{aug_idx}{ext}"
                aug_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_images, aug_name), aug_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

                # 保存增强标注
                if with_labels and aug_bboxes:
                    aug_label_name = f"{name_no_ext}_aug{aug_idx}.txt"
                    write_yolo_labels(
                        os.path.join(out_labels, aug_label_name),
                        aug_bboxes, aug_labels,
                    )

                total_generated += 1

            except Exception as e:
                logger.warning(f"增强失败 [{fname} aug{aug_idx}]: {e}")

        if (idx + 1) % 30 == 0 or idx == len(image_paths) - 1:
            logger.info(f"[进度] {idx + 1}/{len(image_paths)} 图像处理完毕, "
                        f"已生成 {total_generated} 样本")

    elapsed = time.time() - t_start

    # 统计
    final_images = len(glob.glob(os.path.join(out_images, "*")))
    final_labels = len(glob.glob(os.path.join(out_labels, "*.txt"))) if with_labels else 0

    logger.info("=" * 60)
    logger.info("增强完成 — 统计报告")
    logger.info("=" * 60)
    logger.info(f"原始图像: {len(image_paths)}")
    logger.info(f"增强倍数: {multiplier}")
    logger.info(f"生成样本总数: {final_images} 张图像, {final_labels} 个标注")
    logger.info(f"扩增倍率: {final_images / max(len(image_paths), 1):.1f}x")
    logger.info(f"耗时: {elapsed:.2f}s")
    logger.info(f"输出路径: {os.path.abspath(output_dir)}")
    logger.info("=" * 60)


def run_all_pipelines(input_dir: str, output_base: str, multiplier: int = 1,
                      config_path: str = None):
    """对每种增强风格各生成一份"""
    for name in ["night", "backlight", "shadow", "camera_degradation"]:
        out_dir = os.path.join(output_base, f"aug_{name}")
        run_augmentation(
            input_dir=input_dir,
            output_dir=out_dir,
            pipeline_name=name,
            multiplier=multiplier,
            with_labels=False,
            config_path=config_path,
        )


def main():
    parser = argparse.ArgumentParser(
        description="座舱场景数据增强 — 基于 Albumentations 的高级增强流水线"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "pipeline_config.yaml"),
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="输入图像目录（默认: data/annotations/yolo/images）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出目录（默认: data/augmented）"
    )
    parser.add_argument(
        "--pipeline", type=str, default="full",
        choices=list_pipelines(),
        help="增强流水线名称"
    )
    parser.add_argument(
        "--multiplier", type=int, default=3,
        help="每张图像生成的增强副本数"
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="不处理标注文件"
    )
    parser.add_argument(
        "--all-pipelines", action="store_true",
        help="对每种增强风格各生成一份"
    )
    args = parser.parse_args()

    default_input = os.path.join(PROJECT_ROOT, "data", "annotations", "yolo", "images")
    default_output = os.path.join(PROJECT_ROOT, "data", "augmented")
    input_dir = args.input or default_input
    output_dir = args.output or default_output

    if args.all_pipelines:
        run_all_pipelines(input_dir, output_dir, args.multiplier, args.config)
    else:
        run_augmentation(
            input_dir=input_dir,
            output_dir=output_dir,
            pipeline_name=args.pipeline,
            multiplier=args.multiplier,
            with_labels=not args.no_labels,
            config_path=args.config,
        )


if __name__ == "__main__":
    main()
