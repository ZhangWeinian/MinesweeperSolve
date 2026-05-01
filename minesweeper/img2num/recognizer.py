import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model import MinesweeperCNN
from .preprocessor import binarize_cell

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class CellRecognizer:
    """基于 CNN 的格子识别器。

    识别流程：
    1. binarize_cell 预判状态（hidden / blank / 有内容）
    2. 仅对"有内容"的已翻开格子调用 CNN 推理
    """

    def __init__(self, model_path, meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        self.idx_to_class: dict[int, str] = {int(k): v for k, v in meta.items()}
        num_classes = len(self.idx_to_class)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MinesweeperCNN(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ [CNN 识别引擎] 模型已加载，设备: {self.device}，类别数: {num_classes}")

    def _cnn_predict(self, cell_img_bgr: np.ndarray) -> str:
        """对 BGR numpy 格子图像运行 CNN 推理。

        Returns:
            类别字符串，"1"-"8" 或 "flag"
        """

        # BGR → RGB → PIL Image
        rgb = cell_img_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb.astype(np.uint8))
        tensor = _TRANSFORM(pil_img).unsqueeze(0).to(self.device)  # type: ignore[assignment]

        with torch.no_grad():
            _, idx = torch.max(self.model(tensor), 1)

        return self.idx_to_class[int(idx.item())]

    def identify(self, cell_img: np.ndarray):
        """识别单个 64×64 格子。

        Args:
            cell_img: BGR numpy 数组 (64×64)

        Returns:
            int (0-8) | "F" | -1 (未翻开)
        """

        shape, is_opened = binarize_cell(cell_img)

        if not is_opened:
            if shape is not None:
                return "F"
            return -1

        if shape is None:
            return 0

        label = self._cnn_predict(cell_img)
        if label == "flag":
            return "F"
        else:
            return int(label)  # "1"-"8" → 1-8
