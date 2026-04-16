import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from ..config import (
    COLOR_FALLBACK_THRESHOLD,
    FORCE_THRESHOLD,
    MATCH_PAD,
    NORMAL_THRESHOLD,
    SSIM_WEIGHT,
    TEMPLATE_DIR,
    TEMPLATE_SIZE,
    TM_WEIGHT,
)
from .preprocessor import KERNEL_3x3, binarize_cell, get_digit_color_label

VALID_LABELS = frozenset(["1", "2", "3", "4", "5", "6", "7", "8", "F", "Q"])


class CellRecognizer:
    """模板匹配 + SSIM 结构相似度 + 颜色分类的多通道识别器。"""

    def __init__(self):
        self.templates: dict[str, list[np.ndarray]] = {}
        self.load_templates()

    def load_templates(self):
        self.templates.clear()
        template_dir = TEMPLATE_DIR
        if not template_dir.exists():
            template_dir.mkdir(parents=True)

        count = 0
        for path in template_dir.iterdir():
            if path.suffix.lower() != ".png":
                continue
            base_label = path.stem.split("_")[0].upper()
            if base_label not in VALID_LABELS:
                continue

            tpl_bw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if tpl_bw is None:
                continue

            resized = cv2.resize(tpl_bw, TEMPLATE_SIZE)
            # 统一二值化 + 形态学处理，与 binarize_cell 管线一致
            _, clean = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, KERNEL_3x3)
            self.templates.setdefault(base_label, []).append(clean)
            count += 1

        print(
            f"✅ [视觉特征库] 成功挂载 {count} 个 {TEMPLATE_SIZE[0]}×{TEMPLATE_SIZE[1]} 模板。"
        )

    def identify(self, cell_img, force_guess=False):
        """识别单个 64×64 格子。

        Returns:
            int (0-8) | "F" | "Q" | -1 (未翻开) | "?" (无法识别)
        """
        shape, is_opened = binarize_cell(cell_img)

        if shape is None:
            return 0 if is_opened else -1

        if not self.templates:
            return "?"

        # ── 滑动窗口 + SSIM 混合匹配 ────────────────────
        pad = MATCH_PAD
        h, w = shape.shape
        padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
        padded[pad : pad + h, pad : pad + w] = shape

        # 第一轮：模板匹配，取每个 label 的最佳得分
        label_best: dict[str, tuple[float, tuple, np.ndarray]] = {}
        for label, tpl_list in self.templates.items():
            for tpl in tpl_list:
                res = cv2.matchTemplate(padded, tpl, cv2.TM_CCOEFF_NORMED)
                _, tm_score, _, max_loc = cv2.minMaxLoc(res)
                if label not in label_best or tm_score > label_best[label][0]:
                    label_best[label] = (tm_score, max_loc, tpl)

        # 按 TM 得分排序
        ranked = sorted(label_best.items(), key=lambda x: x[1][0], reverse=True)

        # 第二轮：对 Top-2 候选计算 SSIM 精细评分
        refined = []
        for label, (tm_score, max_loc, tpl) in ranked[:2]:
            mx, my = max_loc
            th, tw = tpl.shape
            aligned = padded[my : my + th, mx : mx + tw]
            ssim_val = ssim(aligned, tpl, data_range=255)
            combined = TM_WEIGHT * tm_score + SSIM_WEIGHT * ssim_val
            refined.append((label, combined))

        # 剩余候选保留原始 TM 得分
        for label, (tm_score, _, _) in ranked[2:]:
            refined.append((label, tm_score))

        refined.sort(key=lambda x: x[1], reverse=True)

        best_label, best_score = refined[0]
        second_label = refined[1][0] if len(refined) > 1 else None
        second_score = refined[1][1] if len(refined) > 1 else -1.0

        # ── 颜色辅助识别（仅对已翻开的数字格生效）──────
        color_label = None
        if is_opened and best_label not in ("F", "Q"):
            color_label = get_digit_color_label(cell_img, shape)

        threshold = FORCE_THRESHOLD if force_guess else NORMAL_THRESHOLD

        if best_score > threshold and best_label is not None:
            # 模板 + 颜色一致 → 高置信度
            if color_label is None or best_label == color_label:
                return best_label if best_label in ("F", "Q") else int(best_label)

            # 颜色与模板矛盾，但第二名恰好是颜色候选且分差极小 → 信任颜色
            if second_label == color_label and (best_score - second_score) < 0.12:
                return int(color_label)

            # 模板分数足够高，仍信任模板
            if best_score > 0.70:
                return best_label if best_label in ("F", "Q") else int(best_label)

            # 低置信度区间，优先颜色
            if color_label:
                return int(color_label)

            return best_label if best_label in ("F", "Q") else int(best_label)

        # 低于阈值 → 尝试用颜色兜底
        if color_label and best_score > COLOR_FALLBACK_THRESHOLD:
            return int(color_label)

        return 0 if force_guess else "?"
