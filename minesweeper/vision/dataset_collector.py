import threading
from pathlib import Path

import cv2

from ..config import PROJECT_ROOT

DATASET_TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
DATASET_ERROR_DIR = PROJECT_ROOT / "dataset" / "error"
MAX_TRAIN_PER_CLASS = 1000


def _label_to_folder(label) -> str:
    """将识别结果转换为数据集子目录名。"""
    if label == "F" or label == "flag":
        return "flag"
    return str(label)


class DatasetCollector:
    """线程安全的训练/错误样本采集器。

    职责：
    - 将已识别格子图像保存至 dataset/train/<class>/，每类最多 500 张。
    - 将疑似误识别格子图像保存至 dataset/error/<class>/。
    - 会话内通过 (label, r, c) 三元组去重，避免同一格子重复写入。
    """

    def __init__(self):
        self._lock = threading.Lock()
        # label_str → 磁盘上已有文件数
        self._train_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}
        # 本次运行中已保存训练样本的位置集合，格式 (label_str, r, c)
        self._saved_positions: set[tuple] = set()

        self._init_counts()

    def _init_counts(self) -> None:
        """从磁盘读取各类已有图片数量。"""
        DATASET_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_ERROR_DIR.mkdir(parents=True, exist_ok=True)

        for folder in DATASET_TRAIN_DIR.iterdir():
            if folder.is_dir():
                self._train_counts[folder.name] = len(list(folder.glob("*.png")))

        for folder in DATASET_ERROR_DIR.iterdir():
            if folder.is_dir():
                self._error_counts[folder.name] = len(list(folder.glob("*.png")))

    def reset_session(self) -> None:
        """新局开始时调用，清除位置去重缓存（允许新局重新采集相同位置）。"""
        with self._lock:
            self._saved_positions.clear()

    def try_save_train(self, cell_img_bgr, label, pos: tuple) -> bool:
        """尝试保存一张训练样本。

        Args:
            cell_img_bgr: BGR numpy 格子图像
            label: 识别结果（int 1-8 或 "F"）
            pos: (row, col) 格子坐标，用于会话内去重

        Returns:
            True 表示成功保存，False 表示已跳过（达上限 / 本局已保存）
        """
        folder_name = _label_to_folder(label)
        key = (folder_name, pos[0], pos[1])

        with self._lock:
            if key in self._saved_positions:
                return False
            count = self._train_counts.get(folder_name, 0)
            if count >= MAX_TRAIN_PER_CLASS:
                return False

            save_dir = DATASET_TRAIN_DIR / folder_name
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{count:04d}.png"
            cv2.imwrite(str(out_path), cell_img_bgr)

            self._train_counts[folder_name] = count + 1
            self._saved_positions.add(key)
            return True

    def save_error(self, cell_img_bgr, label) -> None:
        """保存一张疑似误识别的格子图像到 dataset/error/<label>/。

        Args:
            cell_img_bgr: BGR numpy 格子图像
            label: CNN 返回的（可能错误的）识别结果
        """
        folder_name = _label_to_folder(label)

        with self._lock:
            count = self._error_counts.get(folder_name, 0)
            save_dir = DATASET_ERROR_DIR / folder_name
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{count:04d}.png"
            cv2.imwrite(str(out_path), cell_img_bgr)
            self._error_counts[folder_name] = count + 1
