import threading
from pathlib import Path

import cv2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_TRAIN_DIR = _PROJECT_ROOT / "dataset" / "train"
DATASET_ERROR_DIR = _PROJECT_ROOT / "dataset" / "error"
MAX_TRAIN_PER_CLASS = 1000


def _label_to_folder(label) -> str:
    """将识别结果转换为数据集子目录名。"""
    if label == "F" or label == "flag":
        return "flag"
    return str(label)


class DatasetCollector:
    """线程安全的训练/错误样本采集器。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._train_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}
        self._saved_positions: set[tuple] = set()

        self._init_counts()

    def _init_counts(self) -> None:
        DATASET_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        DATASET_ERROR_DIR.mkdir(parents=True, exist_ok=True)

        for folder in DATASET_TRAIN_DIR.iterdir():
            if folder.is_dir():
                self._train_counts[folder.name] = len(list(folder.glob("*.png")))

        for folder in DATASET_ERROR_DIR.iterdir():
            if folder.is_dir():
                self._error_counts[folder.name] = len(list(folder.glob("*.png")))

    def reset_session(self) -> None:
        with self._lock:
            self._saved_positions.clear()

    def try_save_train(self, cell_img_bgr, label, pos: tuple) -> bool:
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
        folder_name = _label_to_folder(label)

        with self._lock:
            count = self._error_counts.get(folder_name, 0)
            save_dir = DATASET_ERROR_DIR / folder_name
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{count:04d}.png"
            cv2.imwrite(str(out_path), cell_img_bgr)
            self._error_counts[folder_name] = count + 1
