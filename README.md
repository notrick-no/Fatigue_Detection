# Fatigue_Detection — 疲劳驾驶实时检测系统

> 基于 RetinaFace + 自定义 CNN + PERCLOS 算法的三级疲劳检测管线。

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-export-brightgreen)](https://onnx.ai)

## 概述

深度学习疲劳驾驶检测系统，用于车载场景下的驾驶员状态实时监控。通过三级管线实现从人脸检测到疲劳判定的全流程。

## 架构

```
摄像头 → 人脸检测 (RetinaFace) → 疲劳特征提取 (自定义 CNN) → 疲劳判定 (PERCLOS)
```

### 三级检测管线

| 阶段 | 模型 | 输出 |
|------|------|------|
| 1. 人脸检测 | RetinaFace (mobile0.25/Slim/RFB) | 人脸框 + 5 关键点 |
| 2. 特征提取 | 自定义 CNN（SeparableConv2d + ResidualBlock） | 眼睛开闭 + 嘴巴开闭 |
| 3. 疲劳判定 | PERCLOS 算法 + 多维度加权 | 疲劳/清醒 + 置信度 |

## 技术栈

| 组件 | 技术 |
|------|------|
| 人脸检测 | RetinaFace 全流程手写（config/data/prior box/decode/NMS） |
| 特征提取 | 自定义轻量 CNN（SeparableConv2d + ResidualBlock，单通道输入） |
| 疲劳判定 | PERCLOS + 打哈欠检测 + 离位检测 → 加权判定 |
| 可视化 | Grad-CAM 热力图 + ROC 曲线 |
| 模型优化 | ONNX 导出 + FLOPs 计算 |
| GUI | wxPython 实时检测界面 |

## 快速开始

```bash
pip install -r requirements.txt
cd Fatigue_Detector
python FatigueDetector.py
```

## 项目结构

```
Fatigue_Detector/
├── FatigueDetector.py       # 主入口 + wxPython GUI
├── model.py                 # 模型组装
├── train.py                 # RetinaFace 训练
├── test.py                  # WIDERFace 测试
├── test_widerface.py        # WIDERFace 评估
├── convert_to_onnx.py       # ONNX 导出
├── calculate_paremeter_flop.py  # FLOPs 计算
├── plot_roc_curves.py       # ROC 评估
├── data/                    # 数据处理
│   ├── config.py            # 配置管理
│   ├── data_augment.py      # 数据增强
│   └── wider_face.py        # WIDERFace 数据集
├── layers/
│   ├── functions/prior_box.py   # PriorBox 生成
│   └── modules/multibox_loss.py # MultiBoxLoss
├── load_model/              # MXNet → PyTorch 权重迁移
│   ├── MainModel.py
│   ├── mxnet_loader.py
│   ├── mxnet_model_structure.py
│   └── pytorch_loader.py
├── models/                  # RetinaFace 变体
│   ├── retinaface.py        # 基础版
│   ├── net.py               # MobileNet0.25
│   ├── net_slim.py          # Slim 轻量版
│   └── net_rfb.py           # RFB 版（Receptive Field Block）
├── recognition/             # 疲劳识别 CNN
│   ├── train.py             # CNN 训练
│   ├── check.py             # 模型验证
│   └── grad_cam.py          # Grad-CAM 可视化
└── utils/                   # 工具函数
    ├── anchor_generator.py  # Anchor 生成
    ├── anchor_decode.py     # Anchor 解码
    ├── box_utils.py         # BBox 工具
    ├── nms.py               # GPU NMS
    ├── py_cpu_nms.py        # CPU NMS
    └── timer.py             # 计时工具

pytorch-eyeblink-detection/  # 眨眼检测模块
└── src/ (model.py, train.py, check.py, grad_cam.py)

pytorch-mouth-detection/     # 嘴部检测模块
└── src/ (model.py, train.py, check.py, grad_cam.py, mouth_check.py)
```

## RetinaFace 实现要点

本项目是 RetinaFace 的**全流程手写实现**（非 API 调包）：

- **PriorBox 生成**：多尺度 anchor，针对人脸比例优化
- **MultiBoxLoss**：分类 + 回归 + 关键点联合损失
- **Decode & NMS**：anchor 解码 + 非极大值抑制后处理
- **多模型变体**：MobileNet0.25 / Slim / RFB 三种 backbone
- **权重迁移**：MXNet → PyTorch 权重转换脚本

## 疲劳判定逻辑

### PERCLOS (Percentage of Eye Closure)

```
PERCLOS = 闭眼帧数 / 总帧数 × 100%
疲劳阈值：PERCLOS > 20%
```

### 多维度加权判定

| 指标 | 权重 | 说明 |
|------|------|------|
| PERCLOS | 高 | 核心指标，眼睑闭合比例 |
| 打哈欠频率 | 中 | 嘴部开合检测 |
| 离位检测 | 低 | 驾驶员面部偏离画面 |

## 模型变体

| 模型 | Backbone | 适用场景 |
|------|----------|---------|
| mobile0.25 | MobileNet 0.25× | 移动端 / 嵌入式 |
| Slim | 轻量化设计 | 边缘设备 |
| RFB | Receptive Field Block | 精度优先 |

支持 ONNX 导出用于生产部署。

## License

MIT
