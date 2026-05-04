"""
VLM 数据闭环主管线

将视频推流 → 模型推理 → 不确定性采样 → 难例存储 串联为完整的闭环管线。

用法:
    python run_pipeline.py                              # 使用默认配置
    python run_pipeline.py --config configs/custom.yaml # 指定配置
    python run_pipeline.py --fast                       # 不限帧率快速处理
"""

import os
import sys
import argparse
import time

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.simulator.video_streamer import CabinVideoSimulator
from src.inference.engine import create_engine, SimulatedInferenceEngine
from src.sampling.uncertainty import UncertaintySampler
from src.storage.hard_example_store import HardExampleStorage
from src.utils.helpers import load_config, setup_logger


def run_pipeline(config_path: str, fast_mode: bool = False, visualize: bool = False):
    """执行完整的数据闭环管线

    流程:
        1. 模拟器逐帧生成座舱视频帧
        2. 推理引擎对每帧执行目标检测/分类
        3. 不确定性采样器评估每帧是否为难例
        4. 满足条件的帧及元数据存入难例库
    """
    # 加载配置
    config = load_config(config_path)
    log_dir = config.get("storage", {}).get("log_dir", "data/logs")
    logger = setup_logger("pipeline", log_dir)

    logger.info("=" * 60)
    logger.info("VLM 数据闭环管线启动")
    logger.info("=" * 60)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"模式: {'快速(不限帧率)' if fast_mode else '实时模拟'}")

    # 初始化各组件
    simulator = CabinVideoSimulator(config)
    engine = create_engine(config)
    sampler = UncertaintySampler(config)
    storage = HardExampleStorage(config)

    logger.info(f"推理引擎: {type(engine).__name__}")
    logger.info(f"模拟视频: {simulator.width}x{simulator.height} @ {simulator.fps}fps, "
                f"共 {simulator.total_frames} 帧")
    logger.info(f"长尾场景概率: {simulator.longtail_prob}")
    logger.info(f"不确定性采样 - 置信度阈值: {sampler.conf_threshold}, "
                f"IoU抖动阈值: {sampler.iou_threshold}")
    logger.info(f"难例库: {storage.output_dir}")
    logger.info("-" * 60)

    # 选择推流模式
    stream = simulator.stream_fast() if fast_mode else simulator.stream()

    # 统计
    total_frames = 0
    total_hard = 0
    scene_counts = {}
    t_start = time.time()

    try:
        for frame, meta in stream:
            total_frames += 1
            scene_counts[meta.scene_type] = scene_counts.get(meta.scene_type, 0) + 1

            # Step 2: 推理
            if isinstance(engine, SimulatedInferenceEngine):
                result = engine.infer(frame, frame_id=meta.frame_id,
                                      gt_objects=meta.objects)
            else:
                result = engine.infer(frame, frame_id=meta.frame_id)

            # Step 3: 不确定性评估
            flags = sampler.evaluate(result)

            # Step 4: 保存难例
            if flags:
                for flag in flags:
                    saved_path = storage.save(frame, flag, result)
                    if saved_path:
                        total_hard += 1
                        logger.info(
                            f"[难例] Frame={meta.frame_id:05d} | "
                            f"场景={meta.scene_type} | "
                            f"原因={flag.reason} | "
                            f"分数={flag.score:.3f} | "
                            f"类别={flag.details.get('class', 'N/A')} | "
                            f"保存={os.path.basename(saved_path)}"
                        )

            # 可视化（可选）
            if visualize:
                import cv2
                display = frame.copy()
                for det in result.detections:
                    x1, y1, x2, y2 = det.bbox
                    color = (0, 255, 0) if det.confidence >= sampler.conf_threshold else (0, 0, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    label = f"{det.class_name}:{det.confidence:.2f}"
                    cv2.putText(display, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                if flags:
                    cv2.putText(display, "!! HARD EXAMPLE !!", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("VLM Data Closedloop", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("用户按 Q 退出")
                    break

            # 进度打印
            if total_frames % 50 == 0:
                elapsed = time.time() - t_start
                fps_actual = total_frames / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[进度] {total_frames}/{simulator.total_frames} 帧 | "
                    f"难例: {total_hard} | "
                    f"实际FPS: {fps_actual:.1f}"
                )

    except KeyboardInterrupt:
        logger.info("用户中断 (Ctrl+C)")

    if visualize:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # 打印最终报告
    elapsed = time.time() - t_start
    stats = storage.get_statistics()

    logger.info("=" * 60)
    logger.info("管线执行完毕 — 统计报告")
    logger.info("=" * 60)
    logger.info(f"总帧数: {total_frames}")
    logger.info(f"耗时: {elapsed:.2f}s ({total_frames / elapsed:.1f} fps)")
    logger.info(f"场景分布: {scene_counts}")
    logger.info(f"难例总数: {stats['total_saved']}")
    logger.info(f"去重跳过: {stats['total_skipped_dedup']}")
    logger.info(f"难例分类: {stats['category_counts']}")
    logger.info(f"难例库路径: {os.path.abspath(stats['output_dir'])}")
    logger.info("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="VLM 数据闭环系统 — 面向长尾座舱场景的自动化难例挖掘"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "pipeline_config.yaml"),
        help="管线配置文件路径"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="快速模式（不限帧率）"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="可视化模式（弹出 OpenCV 窗口显示实时检测）"
    )
    args = parser.parse_args()

    run_pipeline(args.config, fast_mode=args.fast, visualize=args.visualize)


if __name__ == "__main__":
    main()
