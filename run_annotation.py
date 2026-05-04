"""
VLM 自动标注管线

读取难例库中的图像，调用 VLM 进行自动标注，
将结果转换为 YOLO / COCO 标准格式输出。

用法:
    python run_annotation.py                                  # 默认配置
    python run_annotation.py --format yolo                    # 仅输出 YOLO 格式
    python run_annotation.py --format coco                    # 仅输出 COCO 格式
    python run_annotation.py --input data/hard_examples       # 指定输入目录
"""

import os
import sys
import argparse
import glob
import time
import json

import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.annotation.prompt_factory import PromptFactory
from src.annotation.vlm_backends import create_vlm_backend, VLMResponse
from src.annotation.response_parser import VLMResponseParser
from src.annotation.format_converter import YOLOFormatConverter, COCOFormatConverter
from src.utils.helpers import load_config, setup_logger


# 难例子目录 → 推荐场景类型映射
REASON_TO_SCENE = {
    "low_confidence": "normal",
    "bbox_jitter":    "normal",
    "combined":       "normal",
    "pet_in_rear":    "pet_in_rear",
    "left_object":    "left_object",
    "camera_occluded": "camera_occluded",
    "extreme_lighting": "extreme_lighting",
}


def infer_scene_from_metadata(meta_path: str) -> str:
    """从难例元数据 JSON 推断场景类型"""
    if not os.path.isfile(meta_path):
        return "normal"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 检查检测类别
        details = meta.get("flag_details", {})
        cls = details.get("class", "")
        if cls in ("pet", "dog", "cat"):
            return "pet_in_rear"
        if cls in ("left_object", "bag", "phone"):
            return "left_object"
        if cls in ("obstruction", "hand"):
            return "camera_occluded"
        # 检查 reason
        reason = meta.get("reason", "")
        return REASON_TO_SCENE.get(reason, "normal")
    except Exception:
        return "normal"


def collect_images(input_dir: str) -> list[dict]:
    """收集难例库中的所有图像及元数据"""
    entries = []
    for root, dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(root, fname)
            meta_path = os.path.splitext(img_path)[0] + ".json"
            # 从父目录名推断 reason
            parent = os.path.basename(root)
            scene = infer_scene_from_metadata(meta_path)
            entries.append({
                "image_path": img_path,
                "meta_path": meta_path,
                "image_name": fname,
                "reason": parent,
                "scene_type": scene,
            })
    return entries


def run_annotation(config_path: str, input_dir: str = None,
                   output_format: str = "both", force: bool = False):
    """执行 VLM 自动标注管线

    Args:
        force: 如果为 True，即使输出文件已存在也重新处理
    """
    config = load_config(config_path)
    log_dir = config["storage"]["log_dir"]
    logger = setup_logger("annotation", log_dir)

    logger.info("=" * 60)
    logger.info("VLM 自动标注管线启动")
    logger.info("=" * 60)

    # 确定输入目录
    if input_dir is None:
        input_dir = config["storage"]["output_dir"]
    logger.info(f"难例库路径: {os.path.abspath(input_dir)}")

    # 收集图像
    entries = collect_images(input_dir)
    if not entries:
        logger.warning("难例库中未找到图像，请先运行 run_pipeline.py 生成难例")
        return

    # 断点续跑：收集已完成的图像名
    processed = set()
    annotation_output = os.path.join(os.path.dirname(input_dir), "annotations")
    if not force:
        if output_format in ("yolo", "both"):
            prev_labels_dir = os.path.join(annotation_output, "yolo", "labels")
            if os.path.isdir(prev_labels_dir):
                for f in os.listdir(prev_labels_dir):
                    if f.endswith(".txt"):
                        processed.add(os.path.splitext(f)[0] + ".jpg")
        logger.info(
            f"断点续跑: 已完成 {len(processed)} 张, "
            f"待处理 {len(entries)} 张"
            if processed else f"待标注图像: {len(entries)} 张"
        )
    else:
        logger.info(f"强制模式: 待标注图像 {len(entries)} 张")

    # 初始化组件
    vlm_backend = create_vlm_backend(config)
    prompt_factory = PromptFactory()
    logger.info(f"VLM 后端: {type(vlm_backend).__name__}")
    logger.info(f"可用 Prompt 模板: {prompt_factory.list_templates()}")

    # 标注类别（合并配置类别 + VLM 可能检测到的类别）
    base_classes = config["inference"]["classes"]
    extra_classes = ["dog", "cat", "bag", "phone", "wallet", "hand",
                     "obstruction", "sticker", "dirt"]
    all_classes = list(dict.fromkeys(base_classes + extra_classes))

    # 初始化格式转换器
    yolo_converter = None
    coco_converter = None

    if output_format in ("yolo", "both"):
        yolo_dir = os.path.join(annotation_output, "yolo")
        yolo_converter = YOLOFormatConverter(all_classes, yolo_dir)
        logger.info(f"YOLO 输出: {os.path.abspath(yolo_dir)}")

    if output_format in ("coco", "both"):
        coco_dir = os.path.join(annotation_output, "coco")
        coco_converter = COCOFormatConverter(all_classes, coco_dir)
        logger.info(f"COCO 输出: {os.path.abspath(coco_dir)}")

    logger.info(f"类别数: {len(all_classes)}")
    logger.info("-" * 60)

    # 逐张标注
    total = len(entries)
    success = 0
    empty = 0
    skipped = 0
    t_start = time.time()

    for idx, entry in enumerate(entries):
        # 断点续跑：跳过已有输出的图像
        if entry["image_name"] in processed and not force:
            skipped += 1
            if yolo_converter and coco_converter:
                # COCO: 仍需将跳过项加入数据集
                image = cv2.imread(entry["image_path"])
                if image is not None:
                    h, w = image.shape[:2]
                    coco_converter.add_image(entry["image_name"], [], w, h)
            if (idx + 1) % 50 == 0:
                logger.info(f"[进度] {idx + 1}/{total} | 成功: {success} | 跳过: {skipped}")
            continue
        img_path = entry["image_path"]
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"无法读取图像: {img_path}")
            continue

        h, w = image.shape[:2]
        scene_type = entry["scene_type"]

        # 选择 Prompt 模板（优先使用场景专属模板）
        templates = prompt_factory.get_templates_for_scene(scene_type)
        template = templates[0] if templates else prompt_factory.get_template("open_detection")
        prompt = prompt_factory.build_prompt(template.name, image_width=w, image_height=h)

        # VLM 推理
        try:
            response = vlm_backend.infer(image, prompt)
        except Exception as e:
            logger.error(f"VLM 推理失败 [{entry['image_name']}]: {e}")
            continue

        detections = response.detections
        if not detections:
            empty += 1
            if (idx + 1) % 20 == 0:
                logger.info(f"[进度] {idx + 1}/{total} | 成功: {success} | 空: {empty}")
            continue

        # 保存 YOLO 标签
        if yolo_converter:
            # 复制图像到 YOLO images 目录
            dst_img = os.path.join(yolo_converter.images_dir, entry["image_name"])
            if not os.path.exists(dst_img):
                cv2.imwrite(dst_img, image)
            yolo_converter.convert(entry["image_name"], detections, w, h)

        # 添加到 COCO 数据集
        if coco_converter:
            coco_converter.add_image(entry["image_name"], detections, w, h)

        success += 1

        # 日志
        det_summary = ", ".join(
            f"{d.class_name}({d.confidence:.2f})" for d in detections[:5]
        )
        if len(detections) > 5:
            det_summary += f" +{len(detections) - 5} more"
        logger.info(
            f"[{idx + 1}/{total}] {entry['image_name']} | "
            f"场景={scene_type} | 检测={len(detections)} | {det_summary}"
        )

    # 保存 COCO JSON
    if coco_converter:
        coco_path = coco_converter.save()
        logger.info(f"COCO 标注已保存: {coco_path}")

    # 生成 YOLO data.yaml
    if yolo_converter:
        yaml_path = yolo_converter.generate_data_yaml("cabin_hard_examples")
        logger.info(f"YOLO data.yaml 已保存: {yaml_path}")

    elapsed = time.time() - t_start

    # 最终报告
    logger.info("=" * 60)
    logger.info("标注完成 — 统计报告")
    logger.info("=" * 60)
    logger.info(f"总图像数: {total}")
    logger.info(f"成功标注: {success}")
    logger.info(f"空检测: {empty}")
    logger.info(f"跳过(已完成): {skipped}")
    logger.info(f"耗时: {elapsed:.2f}s")

    if yolo_converter:
        yolo_labels = len(glob.glob(os.path.join(yolo_converter.labels_dir, "*.txt")))
        logger.info(f"YOLO labels: {yolo_labels} 个文件")

    if coco_converter:
        stats = coco_converter.get_statistics()
        logger.info(f"COCO: {stats['total_images']} images, "
                    f"{stats['total_annotations']} annotations")
        logger.info(f"COCO 类别分布: {stats['category_counts']}")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="VLM 自动标注管线 — 对难例库图像进行 VLM 标注并转换为标准格式"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "pipeline_config.yaml"),
        help="配置文件路径"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="难例库输入路径（默认从配置中读取）"
    )
    parser.add_argument(
        "--format", type=str, default="both",
        choices=["yolo", "coco", "both"],
        help="输出格式"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新处理所有图像（忽略已有输出）"
    )
    args = parser.parse_args()
    run_annotation(args.config, input_dir=args.input, output_format=args.format, force=args.force)


if __name__ == "__main__":
    main()
