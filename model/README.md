# 扫雷方块 CNN 视觉识别模型 (RGB)

本目录包含了训练好的扫雷方块数字和旗帜识别模型权重（`minesweeper_cnn.pth`）以及标签映射文件（`minesweeper_meta.json`）。该模型采用深度学习 CNN 架构，旨在作为整个扫雷视觉流水线的 **Stage 2（精确特征分类识别）**，建议配合 **Stage 1（OpenCV 纯色及噪声过滤）** 的策略级联使用。

## 标签映射（Label Mapping）

模型的输出张量映射逻辑如下：

- `0` => `flag` (红旗雷标)
- `1` => `1`
- `2` => `2`
- `3` => `3`
- `4` => `4`
- `5` => `5`
- `6` => `6`
- `7` => `7`
- `8` => `8`

## 使用大纲与架构建议

在真正的扫雷解析中，强烈建议采取以下的 **二级判断架构**：

1. **Stage 1 (OpenCV 预处理)**：使用 OpenCV 遍历裁剪出的格子阵列。通过判断格子中心区域像素的色彩对比度或方差，快速筛选出“盲区（未翻出）”和“空白（数字0）”。这一步由纯图像算法过滤以大幅节省计算开销。
2. **Stage 2 (CNN 精准识别)**：当第一步发现格子中存在突出的颜色、或复杂的非单色像素堆叠时，将其保留并输入本目录下的 CNN 模型，通过 RGB 色彩通道与形状的双重特征，精确输出 1~8 数字或 flag。

## 示例调用代码 (Python)

使用前需要安装 PyTorch 和 Pillow 并引入本项目中的 `minesweeper_visual_recognition` 相关依赖。

```python
import torch
from PIL import Image
from torchvision import transforms

# 1. 确保定义或引入了正确的 MinesweeperCNN 类 (3通道输入)
from minesweeper_visual_recognition.model import MinesweeperCNN

# 2. 实例化模型实例并加载权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MinesweeperCNN(num_classes=9).to(device)

model_weights_path = "result/minesweeper_cnn.pth"
model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True))
model.eval()  # 切记切换至评估模式，关闭 Dropout 和 BatchNorm 的变动

# 3. 定义与训练时一致的图像变换 (注意没有 Grayscale，需要三通道 RGB 输入)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # RGB 3 通道的标准化
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 4. 读取图片并进行预测
# 注意：传入前请确保图片被转换为 RGB 格式，否则单通道送入模型会报错
image_path = "path/to/your/cell.png"
raw_image = Image.open(image_path).convert("RGB")
input_tensor = transform(raw_image).unsqueeze(0).to(device)

# 5. 模型推理
with torch.no_grad():
    outputs = model(input_tensor)
    predicted_idx = torch.argmax(outputs, dim=1).item()

# 6. 获取映射结果
IDX_TO_CLASS = {
    0: "flag", 1: "1", 2: "2", 3: "3",
    4: "4", 5: "5", 6: "6", 7: "7", 8: "8"
}
predicted_label = IDX_TO_CLASS[predicted_idx]
print(f"预测结果: {predicted_label}")
```

## 注意事项

- **必须使用RGB**：训练输入和预测输入均必须为 `PIL` 的 `RGB` 3通道模式，并且经过 `(0.5, 0.5, 0.5)` 标准化。如果你在预测前使用了 `cv2` 去读取图像，务必转化为 RGB，即 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 然后再转成 `PIL Image` 或者相应张量。
- **图像大小**：目前的模型第一层特征匹配依然基于 `64x64` 设置。如果输入格子大小不等于 `64x64`，由 transforms 内置的 Resize 来拉伸或缩小处理，一般能够保证原形。
