import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.app.manager.img2num.Preprocessor import binarize_cell
from src.export import MinesweeperCNN

_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class CellRecognizer:
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

    def _cnn_predict_batch(self, cell_imgs_bgr: list[np.ndarray]) -> tuple[list[str], list[float]]:
        """
        批量 CNN 推理。
        返回: (类别列表, 置信度列表)
        注意：这里绝不作拦截，置信度仅作为上层 ConsistencyChecker 的“3帧决胜”打分依据
        """

        tensors = []
        for img in cell_imgs_bgr:
            rgb = img[:, :, ::-1].copy()
            pil_img = Image.fromarray(rgb.astype(np.uint8))
            tensors.append(_TRANSFORM(pil_img))

        batch_tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
            max_probs, indices = torch.max(probs, dim=1)

        labels = [self.idx_to_class[int(idx)] for idx in indices.cpu().numpy()]
        confidences = max_probs.cpu().numpy().tolist()
        return labels, confidences

    def analyze_row(self, cell_images: list[np.ndarray]) -> list[tuple[int | str, float]]:
        """
        分析一行图像。
        返回: [(value, confidence), ...]
        confidence 为 -1.0 表示盲区/空白，不参与时序校验，直接采信
        """

        results: list[tuple[int | str, float]] = [(-1, -1.0)] * len(cell_images)
        cnn_indices = []

        for i, img in enumerate(cell_images):
            shape, is_opened = binarize_cell(img)
            if not is_opened:
                val = "F" if shape is not None else -1
                results[i] = (val, -1.0)
            elif shape is None:
                results[i] = (0, -1.0)
            else:
                cnn_indices.append(i)

        if cnn_indices:
            cnn_imgs = [cell_images[i] for i in cnn_indices]
            labels, confs = self._cnn_predict_batch(cnn_imgs)

            for idx_in_batch, i in enumerate(cnn_indices):
                lbl = labels[idx_in_batch]
                conf = confs[idx_in_batch]
                final_val = "F" if lbl == "flag" else int(lbl)
                results[i] = (final_val, conf)

        return results
