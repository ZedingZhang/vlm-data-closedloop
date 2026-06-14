# VLM Data Closedloop

面向自动驾驶座舱场景的 **VLM 数据闭环系统**，通过"模拟推流 → 模型推理 → 困难样本挖掘 → VLM 自动标注 → 数据增强"链路，自动发现并对长尾场景进行标注，持续扩充训练数据集。

## 30-second verification

快速确认核心工具函数、响应解析和不确定性采样逻辑：

```bash
git clone https://github.com/ZedingZhang/vlm-data-closedloop.git
cd vlm-data-closedloop
python -m pip install -r requirements.txt
python -m pytest -q
```

## 架构

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 模拟推流   │───▶│ 模型推理   │───▶│ 不确定性   │───▶│ 难例存储   │
│ Simulator │    │Inference │    │ Sampling │    │ Storage  │
└──────────┘    └──────────┘    └──────────┘    └─────┬────┘
                                                     │
                    ┌────────────────────────────────┘
                    ▼
              ┌──────────┐    ┌──────────┐
              │VLM 自动标注 │───▶│ 数据增强   │
              │Annotation │    │Augment.  │
              └──────────┘    └──────────┘
```

## 项目结构

```
vlm-data-closedloop/
  configs/
    pipeline_config.yaml         # 全管线 YAML 配置
  src/
    simulator/                   # 合成座舱视频生成（注入长尾场景）
    inference/                   # 推理引擎（YOLOv8 / 模拟后端）
    sampling/                    # 困难样本挖掘（低置信度 + 边界框抖动）
    storage/                     # 困难样本落盘与去重
    annotation/                  # VLM 自动标注（提示词/解析/格式转换）
    augmentation/                # 座舱场景数据增强
    utils/                       # 配置加载、日志、IoU 等工具
  data/                          # 运行时产出（不入版本库）
  run_pipeline.py                # 入口：主数据闭环管线
  run_annotation.py              # 入口：VLM 自动标注
  run_augmentation.py            # 入口：数据增强批量脚本
  requirements.txt
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 运行主闭环管线

```bash
# 使用模拟推理后端（无需 GPU）
python run_pipeline.py

# 可视化模式
python run_pipeline.py --visualize

# 快速模式（不限帧率）
python run_pipeline.py --fast

# 使用自定义配置
python run_pipeline.py --config configs/custom.yaml
```

管线会依次执行：**视频模拟 → 推理 → 不确定性采样 → 困难样本存储**，输出到 `data/hard_examples/`。

### VLM 自动标注

```bash
# 对困难样本进行自动标注
python run_annotation.py

# 仅输出 YOLO 格式
python run_annotation.py --format yolo

# 指定输入目录
python run_annotation.py --input data/hard_examples/combined
```

支持三种 VLM 后端（在 `pipeline_config.yaml` 中配置）：
- **Simulated** — 基于图像分析的模拟 VLM，无需 GPU
- **Qwen-VL** — Qwen2.5-VL-7B-Instruct，支持本地推理/OpenAI 兼容 API
- **Grounding-DINO** — 开放词汇检测，支持本地/API 模式

### 数据增强

```bash
# 夜间增强，每张图扩增 3 倍
python run_augmentation.py --pipeline night --multiplier 3

# 使用完整增强管线
python run_augmentation.py --pipeline full

# 所有风格各生成一份
python run_augmentation.py --all-pipelines
```

四种增强策略：`night`（夜间噪声）、`backlight`（强背光）、`shadow`（局部阴影）、`camera_degradation`（摄像头退化）。

## 管线配置

编辑 `configs/pipeline_config.yaml` 可配置所有环节：

| 模块 | 关键参数 |
|------|---------|
| **simulator** | 分辨率、FPS、总帧数、长尾场景概率（默认 15%）、场景类型 |
| **inference** | 模型类型（simulated/yolo）、YOLO 权重路径、置信度阈值、设备 |
| **sampling** | 置信度阈值（默认 0.4）、IoU 抖动阈值、滑动窗口大小、融合策略 |
| **storage** | 输出目录、保存格式、去重间隔 |
| **vlm** | 后端选择、Qwen/Grounding-DINO 的 API 地址与参数 |
| **annotation** | 输出格式（yolo/coco/both）、输出目录 |

模拟推理后端会在真实标注上叠加高斯噪声、边界框抖动、随机漏检/误检，用于在无 GPU 环境下验证管线逻辑。

## 长尾场景类型

模拟器以可配置概率随机注入以下场景：

- **pet_in_rear** — 后排出现宠物
- **left_object** — 遗留物品
- **extreme_lighting** — 极端光照
- **camera_occluded** — 摄像头被遮挡

## 困难样本挖掘策略

1. **低置信度** — 检测置信度低于阈值（默认 0.4）即标记
2. **边界框抖动** — 连续帧间 IoU 波动过大则标记抖动
3. **融合策略** — `any`（任一触发）或 `all`（同时触发）

## 依赖

- Python 3.10+
- PyTorch ≥ 2.0, torchvision
- Ultralytics YOLOv8
- Albumentations ≥ 1.3
- OpenCV, NumPy, Pillow
- PyYAML, tqdm

完整依赖见 [requirements.txt](requirements.txt)。
