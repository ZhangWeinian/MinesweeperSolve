# MinesweeperSolve

基于 CNN 图像识别 + 数学概率求解的扫雷全自动 AI，支持训练自己的识别模型并在 Windows 上实时自动操控游戏。

---

## 功能

- **自动运行**（`--using`）：截图识别棋盘，通过概率/约束求解器自动点击与标雷，触发物理急停（鼠标移至角落）或 `ESC` 可随时中止。
- **训练模型**（`--train`）：用 CNN 对棋盘格图片进行分类训练（1~8、flag），并可验证精度。
- **数据采集**：自动运行时同步将识别到的格子图片存入 `src/dataset/train/`，超过阈值后停止采集。

---

## 目录结构

```text
MinesweeperSolve/
├── main.py                  # 统一入口
├── src/
│   ├── app/                 # 自动运行模块
│   │   └── manager/
│   │       ├── BusController.py      # 主循环调度
│   │       ├── MathematicalSolver.py # 概率/约束求解器
│   │       ├── MouseController.py    # 鼠标/键盘控制
│   │       ├── Screenshot.py         # 截图与格子裁切
│   │       └── img2num/              # 图像识别 + 数据采集
│   ├── cnn/                 # CNN 训练 & 测试模块
│   │   ├── minesweeper_ocr/ # 数据集、训练器、预测器
│   │   ├── dataset/         # 训练/测试图片
│   │   ├── result/          # 输出的 .pth 与 meta.json
│   │   └── test/            # 精度验证
│   ├── export/              # 共享模型定义 MinesweeperCNN
│   └── dataset/             # app 运行时采集的样本
│       ├── train/
│       └── error/
└── model/                   # app 运行时加载的模型（从 result/ 复制）
    ├── minesweeper_cnn.pth
    └── minesweeper_meta.json
```

---

## 环境要求

- Python ≥ 3.13
- CUDA 13.0（可选，CPU 也可运行）
- [uv](https://github.com/astral-sh/uv) 包管理器

---

## 快速开始

```bash
# 安装依赖
uv sync

# 训练 CNN 模型
uv run main.py --train

# 仅测试已有模型（N = 测试轮数，1~100）
uv run main.py --train --test N

# 启动自动扫雷（Windows，默认 16×30 专家模式）
uv run main.py --using

# 自定义棋盘尺寸
uv run main.py --using --rows 9 --cols 9 --mines 10

# 省略模式参数时，Windows 自动选 --using，Linux/WSL 自动选 --train
uv run main.py
```

> **安全停止**：将鼠标快速移至屏幕角落（PyAutoGUI FailSafe），或按 `ESC`。

---

## 工作流程

```text
训练阶段
  uv run main.py --train
      └─ 读取 src/cnn/dataset/train/
      └─ 训练 MinesweeperCNN（ResNet 轻量版）
      └─ 输出 src/cnn/result/minesweeper_cnn.pth + meta.json

运行阶段（将 result/ 中的文件复制到 model/ 后）
  uv run main.py --using
      └─ 截图识别棋盘格 → CNN 分类
      └─ 概率求解器决策 → 自动点击/标雷
      └─ 同步采集新样本至 src/dataset/train/
```

---

## 命令行参数

### `--using` 模式

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `--rows` | `16` | 棋盘行数 |
| `--cols` | `30` | 棋盘列数 |
| `--mines` | `99` | 地雷总数 |
| `--model-path` | `model/minesweeper_cnn.pth` | 模型权重路径 |
| `--meta-path` | `model/minesweeper_meta.json` | 元数据路径 |

### `--train` 模式

| 参数 | 说明 |
| ---- | ---- |
| `--test [N]` | 跳过训练，直接测试 N 轮（省略 N 默认 1 次） |
